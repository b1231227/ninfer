"""Direct dense-GGUF -> .ninfer conversion (native formats, table-driven).

`qwen3_5_direct_gguf` converts a dense Qwen3.5/3.8-27B-family GGUF straight
to native .ninfer formats driven by a final-format table (see
`direct_gguf_table.py`), without an intermediate re-quantized GGUF. The
wiring replicates `gguf_blocks.qwen3_8_27b_gguf` one-for-one (the same
`text_sources` calls, the same grouping), with two deliberate differences:

1. The file's `output.weight` may be stored as BF16 (untied head): the
   official recipe's validate() accepts stored block types only; here a
   BF16 matrix source supplies `text/output_head` instead (an in-memory
   intercept of the single block_source call, restored in a finally).
2. Native-path ssm_out gather: every `blk.N.ssm_out.weight` source is
   physically pre-gathered
   on the column axis (`ssm_out_column_gather`), and the recipe emits no
   `input_columns` auxiliary. The native linear kernels never consume that
   auxiliary, so a native artifact must already store the matrix in the
   runtime's layout; the GGUF container path (qwen3_8_27b_gguf,
   import_encoded) keeps the original order + auxiliary and is unaffected.

Usage: the table is data, so set it before the recipe runs. The simplest
pattern is a small wrapper file passed as `--recipe FILE.py`:

    # my_direct_recipe.py
    from pathlib import Path
    import tools.convert.direct_gguf_table as direct_gguf_table
    direct_gguf_table.TABLE = Path(__file__).resolve().parent / "my_table.tsv"
    from tools.convert.direct_gguf import qwen3_5_direct_gguf as configure  # noqa: E402

    python -m tools.convert --model CONFIG_DIR --recipe my_direct_recipe.py \
      --components text,vision,mtp --source gguf=MODEL.gguf \
      --source vision=MMPROJ.gguf --device cpu --out OUT.ninfer
"""
from __future__ import annotations

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE
from tools.convert import gguf_blocks as _gb
from tools.convert.gguf_blocks import (
    EXPECTED_HEADER,
    GDN_VALUE_DIM,
    GGUFFile,
    HIDDEN,
    LAYERS,
    MTP_BLOCK,
    VOCABULARY,
    _companions,
    _group_same_format,
    expected_tensors,
    full_attention,
    has_mtp,
    tiled_input_columns,
    text_sources as _official_text_sources,
)
from tools.convert.methods import cast_direct, import_encoded
from tools.convert.sources.logical import LogicalSource

from .direct_gguf_table import TABLE, apply_table


def validate_with_bf16_output(gguf: GGUFFile) -> None:
    """gguf_blocks.validate() that also accepts a BF16 stored output.weight."""

    for key, expected in EXPECTED_HEADER.items():
        if gguf.kv.get(key) != expected:
            raise ValueError(
                f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}"
            )
    mtp = has_mtp(gguf)
    blocks = gguf.kv.get("qwen35.block_count")
    if blocks != LAYERS + (1 if mtp else 0):
        raise ValueError(f"{gguf.path}: {blocks} blocks, expected {LAYERS} (+1 for MTP)")
    expected = expected_tensors(mtp)
    if set(gguf.tensors) != set(expected):
        missing = sorted(set(expected) - set(gguf.tensors))[:5]
        extra = sorted(set(gguf.tensors) - set(expected))[:5]
        raise ValueError(
            f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})"
        )
    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        if kind == "blocks":
            accepted = info.type_id in GGUF_FORMATS_BY_TYPE
            if not accepted and name == "output.weight" and info.type_name == "BF16":
                accepted = True
        else:
            accepted = info.type_name in ((kind,) if isinstance(kind, str) else kind)
        if info.shape != shape or not accepted:
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")


def bf16_matrix_source(gguf: GGUFFile, tensor: str, shape: tuple[int, int]) -> LogicalSource:
    """Whole BF16 matrix read once; fp32 row slices on request."""

    n, k = shape
    info = gguf.info(tensor)
    if info.type_name != "BF16":
        raise ValueError(f"{tensor}: expected BF16, got {info.type_name}")
    if tuple(int(x) for x in info.shape) != (n, k):
        raise ValueError(f"{tensor}: logical shape {list(info.shape)} != expected {(n, k)}")
    words = gguf.read_bf16_words(tensor)  # uint16 in the logical row-major (n, k) shape
    values = torch.from_numpy(np.ascontiguousarray(words, dtype=np.int16)).view(
        torch.bfloat16
    )
    if tuple(values.shape) != (n, k):
        raise ValueError(f"{tensor}: unexpected matrix shape {tuple(values.shape)}")
    flat = values.float().contiguous().view(-1)

    def read(begin: int, end: int) -> torch.Tensor:
        return flat[begin:end]

    return LogicalSource((n, k), f"{tensor}[bf16]{list(shape)}", read)


_ORIG_BLOCK_SOURCE = _gb.block_source


def ssm_out_column_gather(source: LogicalSource) -> LogicalSource:
    """Physically gather the ssm_out column axis for the native path.

    The native linear kernels never consume `input_columns` (zero
    references in src/ops/linear/*), so a native artifact must store
    the matrix
    pre-gathered: stored column i holds the source weight of grouped
    activation element i (the official native build convention; the GGUF
    bridge path keeps the original order and the auxiliary instead).

    `tiled_input_columns()` maps each stored (source-tiled) column c to
    the grouped element it should multiply, i.e. it is the forward
    permutation sigma; the gather that produces the native layout is its
    INVERSE: stored[i] <- source[argsort(input_columns)[i]]  (== untile
    applied on the column axis). Do not use input_columns[i] as the source
    index directly - that would apply sigma instead of sigma**-1.
    """

    k = int(source.shape[1])
    if k != GDN_VALUE_DIM:
        raise ValueError(f"ssm_out gather: source width {k} != GDN_VALUE_DIM {GDN_VALUE_DIM}")
    columns = tiled_input_columns().astype(np.intp)
    inverse = np.argsort(columns, kind="stable")
    assert (columns[inverse] == np.arange(k)).all()  # sanity: true inverse
    inv = torch.from_numpy(inverse)

    def read_values(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        block = source.read_values(first * k, last * k).reshape(last - first, k)
        gathered = block[:, inv]
        return gathered.reshape(-1)[begin - first * k : end - first * k]

    def read_encoded(begin: int, end: int):
        # The default-assignment probe reads only .format; the final-format
        # table reassigns every ssm_out object to grouped_absmax (the values
        # path above), so the byte-copy path never materializes these objects
        # and can never emit an un-gathered layout.
        return source.read_encoded(begin, end)

    return LogicalSource(
        source.shape,
        f"{source.label.split('[', 1)[0]}[colgather]",
        read_values,
        read_encoded,
        weight_divisor=source.weight_divisor,
        input_divisor=source.input_divisor,
    )


def text_sources_bf16(
    gguf: GGUFFile, *, mtp: bool
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Official text_sources() with two changes.

    1. When the file stores output.weight as BF16 (untied), the single
       block_source("output.weight", ...) call inside the official function
       is intercepted and answered with a bf16_matrix_source.
    2. Every block_source("blk.N.ssm_out.weight", ...) call is wrapped with
       ssm_out_column_gather: the stored matrix is physically
       pre-gathered on the column axis for the native runtime.

    All other calls are delegated to the original block_source unchanged.
    """

    head = None
    if gguf.info("output.weight").type_name == "BF16":
        head = bf16_matrix_source(gguf, "output.weight", (VOCABULARY, HIDDEN))

    def _intercept(gguf_, tensor: str, shape_, select):
        if tensor == "output.weight" and head is not None:
            return head
        source = _ORIG_BLOCK_SOURCE(gguf_, tensor, shape_, select)
        if tensor.endswith(".ssm_out.weight"):
            return ssm_out_column_gather(source)
        return source

    _gb.block_source = _intercept
    try:
        return _official_text_sources(gguf, mtp=mtp)
    finally:
        _gb.block_source = _ORIG_BLOCK_SOURCE


def qwen3_5_direct_gguf(model, recipe, sources) -> None:
    """qwen3_8_27b_gguf wiring + BF16 output head, then the table."""

    config = model.config
    if (
        "num_experts" in config
        or config["hidden_size"] != HIDDEN
        or config["num_hidden_layers"] != LAYERS
    ):
        raise ValueError("the direct GGUF recipe requires the Qwen3.5/3.8-27B Dense geometry")
    gguf = sources["gguf"]
    if not isinstance(gguf, GGUFFile):
        raise ValueError("--source gguf must name the model's .gguf file")
    validate_with_bf16_output(gguf)
    mtp = "mtp" in model.components
    if mtp and not has_mtp(gguf):
        raise ValueError(
            f"{gguf.path}: the MTP component needs an -mtp GGUF (blk.{MTP_BLOCK}.nextn.*)"
        )
    _companions(model, recipe, sources)
    encoded, direct = text_sources_bf16(gguf, mtp=mtp)

    output_bf16 = gguf.info("output.weight").type_name == "BF16"
    formats: dict[str, str] = {}
    for name, source in encoded.items():
        if name == "text/output_head" and output_bf16:
            formats[name] = "bf16"
            recipe.assign(
                name, format="bf16", method=cast_direct, source=source,
                activation_policy="AllowA8",
            )
            continue
        format = source.read_encoded(0, 1).format
        formats[name] = format
        recipe.assign(
            name, format=format, method=import_encoded, source=source,
            activation_policy="AllowA8",
        )
    for name, source in direct.items():
        recipe.assign(
            name, format=model.parameters[name].direct_format, method=cast_direct,
            source=source,
        )
    prefixes = [f"text/layers/{layer}/" for layer in range(LAYERS)]
    if mtp:
        prefixes.append("mtp/layers/0/")
    for p in prefixes:
        if p.startswith("text/") and not full_attention(int(p.split("/")[2])):
            n = p + "gdn/"
            qkv = [n + "query", n + "key", n + "value"]
            recipe.group(
                qkv + [n + "z"] if formats[n + "z"] == formats[n + "query"] else qkv
            )
            recipe.separate([n + "a_projection", n + "b_projection"])
        else:
            a = p + "attention/"
            _group_same_format(
                recipe, [a + "query", a + "key", a + "gate", a + "value"], formats
            )
        _group_same_format(recipe, [p + "mlp/gate", p + "mlp/up"], formats)
    # The input_columns auxiliary is NOT emitted on this native path.
    # The ssm_out matrices were physically pre-gathered on the column axis by
    # ssm_out_column_gather (see above), so the runtime needs no gather table;
    # the native linear kernels never consume one. The GGUF container path
    # (qwen3_8_27b_gguf, import_encoded) keeps its own auxiliary and is
    # unaffected.

    print(
        "[direct_gguf] dense GGUF wiring complete "
        f"(output.weight stored {gguf.info('output.weight').type_name}); "
        "applying final-format table",
        flush=True,
    )
    if TABLE is None:
        raise ValueError(
            "qwen3_5_direct_gguf needs the final-format table: set "
            "tools.convert.direct_gguf_table.TABLE to a table path before the "
            "recipe runs (see the module docstring)"
        )
    apply_table(model, recipe, sources)


RECIPES = {"qwen3_5_direct_gguf": qwen3_5_direct_gguf}

__all__ = [
    "RECIPES",
    "qwen3_5_direct_gguf",
    "ssm_out_column_gather",
    "validate_with_bf16_output",
    "TABLE",
]
