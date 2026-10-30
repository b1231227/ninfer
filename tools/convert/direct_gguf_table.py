"""Final-format table reassignment for the direct dense-GGUF conversion.

Given a base dense-GGUF model (the `qwen3_5_direct_gguf` recipe in
`direct_gguf.py`) and a final-format table, `apply_table` reassigns every
parameter to its table-mandated target format and method, replaces the dense
wiring's default grouping with the runtime's contiguous-object groups (SwiGLU
mlp, 4-way attention/GDN projections, vision qkv+bias), and keeps the
parameters the runtime requires separate (MTP attention, GDN a/b) separate.

The table is a plain-text 13-column TSV (see `_HEADER`): one row per source
tensor. Text rows are keyed by GGUF tensor name (split rows additionally by a
role name in the 13th column); vision rows by mmproj tensor name, prefixed
with the mmproj file name as stored in the table. It is data, not code:
derive it for your own model from a reference conversion (per-object
format/shape lines in the conversion log, joined with the `.conversion.json`
sidecar) and keep it under version control with your project.

Before running the recipe, point `TABLE` at the file:

    import tools.convert.direct_gguf_table as direct_gguf_table
    direct_gguf_table.TABLE = Path("my_final_format_table.tsv")
"""
from __future__ import annotations

import re
from collections import Counter
from math import prod
from pathlib import Path

import torch

from tools.artifact.formats import get_format
from tools.artifact.layouts import row_split_geometry
from tools.convert.methods import _preflight, cast_direct, grouped_absmax
from tools.convert.quantization.groupwise import (
    QuantizedMatrix,
    _canonical_scale_words,
    pick_device,
)
from tools.convert.sources.gguf_source import qwen35_vision_name_map

# The final-format table (TSV, see module docstring). The recipe refuses to
# run while this is None.
TABLE: Path | None = None

Q2 = "q2_g64_fp16"
COMPOSITE_HF = "model.visual.patch_embed.proj.weight"
COMPOSITE_PARTS = ("v.patch_embd.weight", "v.patch_embd.weight.1")

_HEADER = [
    "source_tensor", "ud_struct_type", "ud_struct_bpw", "new_format", "new_bpw",
    "delta_bpw", "runner_up", "method", "elems", "struct_bytes", "new_bytes",
    "delta_bytes", "param",
]
_EXPECTED_FORMATS = {
    "fp32", "bf16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16",
    "q2_g64_fp16", "t2_g128_fp16",
}
_EXPECTED_METHODS = {
    "cast_direct", "grouped_absmax", "custom_q2_encoder(bypass guard)"
}


def load_table(vision_prefix: str | None = None) -> tuple[dict, dict]:
    """Read TABLE -> (text_rows, vision_rows), validated.

    `vision_prefix` is the mmproj file name + ':' exactly as the table
    stores vision rows; vision rows are keyed by their mm tensor name.

    text_rows:  (GGUF tensor name, param) -> (new_format, method, elems)
                param is "" for whole-tensor rows and a role name
                (query/key/value/gate) for split rows (13th column).
    vision_rows: mm.N tensor name -> (new_format, method, elems)
    """
    text_rows: dict[tuple[str, str], tuple[str, str, int]] = {}
    vision_rows: dict[str, tuple[str, str, int]] = {}
    with TABLE.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        if header != _HEADER:
            raise ValueError(f"table header changed: {header}")
        for line in stream:
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) != 13:
                raise ValueError(f"table row has {len(cols)} columns: {line!r}")
            source, new_format, method, elems = cols[0], cols[3], cols[7], int(cols[8])
            param = cols[12]
            if new_format not in _EXPECTED_FORMATS:
                raise ValueError(f"unknown new_format {new_format!r} in {source}")
            if method not in _EXPECTED_METHODS:
                raise ValueError(f"unknown method {method!r} in {source}")
            row = (new_format, method, elems)
            if vision_prefix is not None and source.startswith(vision_prefix):
                if param:
                    raise ValueError(f"vision row {source} must have empty param")
                key = source[len(vision_prefix):]
                if key in vision_rows:
                    raise ValueError(f"duplicate vision row {key}")
                vision_rows[key] = row
            else:
                if param not in ("", "query", "key", "value", "gate"):
                    raise ValueError(f"unknown param {param!r} in {source}")
                tkey = (source, param)
                if tkey in text_rows:
                    raise ValueError(f"duplicate text row {tkey}")
                text_rows[tkey] = row
    return text_rows, vision_rows


def table_key(name: str, source_name: str) -> tuple[str, str]:
    """Resolve a logical parameter's table key (source_tensor, param).

    Only text-layer attention query/gate on their split 2q source and GDN
    query/key/value on their split qkv source take a role key; every other
    parameter (including MTP, whose attn_q row stays unsplit) uses the bare
    source name.
    """
    if name.startswith("text/"):
        role = name.rsplit("/", 1)[1]
        if role in ("query", "gate") and source_name.endswith("attn_q.weight"):
            return (source_name, role)
        if role in ("query", "key", "value") and source_name.endswith("attn_qkv.weight"):
            return (source_name, role)
    return (source_name, "")


def quantize_matrix_q2(weight: torch.Tensor, *, device=None) -> QuantizedMatrix:
    """FP [N,K] -> q2_g64_fp16 codes/scales (untrained absmax encoder).

    Same canonical algorithm as tools.convert.quantization.groupwise.quantize_matrix
    applied to the registered q2 spec (qmin=-1, qmax=2, group_size=64, code_bias=1):
      scale    = binary16(max_abs / qmax)
      code     = clamp(round(x * binary32(1/scale)), qmin, qmax)   (int8)
    The methods.py q2 guard is a policy (it routes q2 to import_encoded of trained
    GSQ Q2_0 codes), not a kernel limit: the decoding contract is shared and is
    implemented verbatim by both the Python reference decoder
    (tools/artifact/codecs/row_split.py::_dequant_q2_g64) and the C++ runtime
    (src/ops/moe_experts/cpu_projection.h): stored unsigned u = code + 1 in
    {0,1,2,3}, weight = (u - 1) * scale, one binary16 scale per 64 columns.
    """
    spec = get_format(Q2)
    if weight.dim() != 2:
        raise ValueError(f"q2 quantization requires rank 2, got {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical
    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    max_abs = grouped.abs().amax(dim=2)
    scales, reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    codes = torch.clamp(
        torch.round(grouped * reciprocal.to(target).unsqueeze(-1)),
        spec.qmin,
        spec.qmax,
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales.to(target))


def custom_q2(request):
    """Custom FP -> q2_g64_fp16 encoder (untrained absmax; bypasses the methods.py q2 policy guard)."""
    if request.target.format != Q2 or len(request.target.shape) != 2:
        raise ValueError("custom_q2 requires a q2_g64_fp16 matrix target")
    if request.parameters:
        raise ValueError("custom_q2 accepts no numerical parameters")
    _preflight(request)
    n, k = request.target.shape

    def produce(output) -> None:
        for begin in range(0, n, request.rows_per_chunk):
            end = min(begin + request.rows_per_chunk, n)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError("custom_q2 source must provide floating-point values")
            encoded = quantize_matrix_q2(values, device=request.device)
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def _source_label_name(label: str, param: str) -> str:
    """Extract the weight-file tensor name from a recipe source label."""
    if param.startswith("vision/"):
        match = re.search(r"(model\.visual\.[A-Za-z0-9_.]+)", label)
        if match is None:
            raise ValueError(f"{param}: no HF vision name in source label {label!r}")
        return match.group(1)
    name = label.split("[", 1)[0]
    if ":" in name:  # tensor_source-style "<path>:<name>"; rsplit (drive-letter colon)
        name = name.rsplit(":", 1)[1]
    return name


def apply_table(model, recipe, sources) -> None:
    """Reassign every parameter per the final conversion table."""
    if TABLE is None:
        raise ValueError(
            "direct_gguf_table.TABLE must point at the final-format table "
            "(see the module docstring)"
        )
    has_vision = "vision" in model.components
    vision_prefix = None
    if has_vision:
        vision_prefix = f"{Path(sources['vision'].path).name}:"
    text_rows, vision_rows = load_table(vision_prefix)
    if has_vision and not vision_rows:
        raise ValueError("the model has a vision component but the table has no vision rows")
    if not has_vision and vision_rows:
        raise ValueError("the table has vision rows but the model has no vision component")

    if has_vision:
        vcfg = model.components["vision"]["config"]
        names_map, _transforms, _shapes, composites = qwen35_vision_name_map(
            vcfg["depth"],
            hidden_size=vcfg["hidden_size"],
            in_channels=3,
            temporal_patch_size=vcfg["temporal_patch_size"],
            patch_size=vcfg["patch_size"],
        )
        hf_to_mm = dict(names_map)
        if set(composites) != {COMPOSITE_HF}:
            raise ValueError(f"unexpected vision composites: {sorted(composites)}")
        fmt_a = vision_rows[COMPOSITE_PARTS[0]][0]
        fmt_b = vision_rows[COMPOSITE_PARTS[1]][0]
        if fmt_a != fmt_b:
            raise ValueError(
                f"composite patch embedding table rows disagree: {fmt_a} vs {fmt_b}"
            )
        for hf in hf_to_mm:
            if hf_to_mm[hf] not in vision_rows:
                raise ValueError(f"vision map target {hf_to_mm[hf]} missing from table")
    else:
        hf_to_mm = {}
        fmt_a = fmt_b = None

    # The runtime requires certain logical parameters to share ONE contiguous
    # native object (module docstring: SwiGLU mlp, 4-way attention/GDN
    # projections, vision qkv+bias). Clear all grouping now; after the table
    # assignment below, rebuild explicit groups exactly per the assigned
    # formats so incompatible members (MTP attention, GDN a/b) stay separate.
    recipe.explicit_groups.clear()
    model.packing_groups.clear()

    methods = {
        "cast_direct": cast_direct,
        "grouped_absmax": grouped_absmax,
        "custom_q2_encoder(bypass guard)": custom_q2,
    }
    used_text: set[str] = set()
    used_vision: set[str] = set()
    counts: Counter = Counter()
    elems_by_key: Counter = Counter()
    assigned = 0
    for name, param in model.parameters.items():
        if name in recipe.aliases:
            continue
        sels = recipe.selections.get(name, [])
        if len(sels) != 1:
            raise ValueError(
                f"{name}: expected one recipe selection, got {len(sels)} "
                "(a parameter the dense wiring left without a source)"
            )
        source = sels[0].source
        key = _source_label_name(source.label, name)
        if name.startswith("vision/"):
            if key == COMPOSITE_HF:
                row = (fmt_a, vision_rows[COMPOSITE_PARTS[0]][1],
                       vision_rows[COMPOSITE_PARTS[0]][2]
                       + vision_rows[COMPOSITE_PARTS[1]][2])
                used_vision.update(COMPOSITE_PARTS)
            else:
                mm = hf_to_mm.get(key)
                if mm is None:
                    raise ValueError(f"{name}: HF name {key!r} not in vision name map")
                row = vision_rows[mm]
                used_vision.add(mm)
        else:
            tkey = table_key(name, key)
            row = text_rows.get(tkey)
            if row is None:
                raise ValueError(f"{name}: source tensor {tkey!r} has no table row")
            used_text.add(tkey)
        new_format, method_name, _table_elems = row
        elems_by_key[(name.startswith("vision/"), key if name.startswith("vision/") else tkey)] += prod(param.shape)
        # The table governs every parameter, including the ones the official
        # recipe wires as direct: the table's direct rows record the exact
        # runtime-required formats (bf16 default for
        # b.direct parameters, bindings.h; fp32 for gdn/a_log + gdn/dt_bias;
        # bf16 for the GDN A/B controls per ops/weight_input.cpp), so plain
        # table reassignment is runtime-compliant. Matrix parameters such as
        # text/output_head follow the reference conversion even when the
        # official wiring is cast_direct bf16 (the reference stores
        # output_weight at 5.5bpw; the runtime
        # accepts quantized b.parameter matrices — official_recipes.py
        # quantizes the head to Q6/Q8 in its own artifacts).
        recipe.assign(name, format=new_format, method=methods[method_name],
                      source=source)
        counts[(new_format, method_name)] += 1
        assigned += 1

    missing_text = set(text_rows) - used_text
    missing_vision = set(vision_rows) - used_vision
    if missing_text:
        raise ValueError(f"text table rows never assigned: {sorted(missing_text)[:8]}")
    if missing_vision:
        raise ValueError(f"vision table rows never assigned: {sorted(missing_vision)[:8]}")
    # Every table row's element count must equal the sum over its logical params.
    for (is_vision, key), total in elems_by_key.items():
        if is_vision:
            if key == COMPOSITE_HF:
                expected = (
                    vision_rows[COMPOSITE_PARTS[0]][2]
                    + vision_rows[COMPOSITE_PARTS[1]][2]
                )
            else:
                expected = vision_rows[hf_to_mm[key]][2]
        else:
            expected = text_rows[key][2]
        if total != expected:
            raise ValueError(
                f"{key}: logical elements {total} != table elems {expected}"
            )

    # Rebuild the runtime-required fusion objects from the assigned formats.
    # A group is emitted only when all members share one format (Recipe.emit
    # rejects heterogeneous explicit groups); the table guarantees that for
    # the attention/GDN/mlp/vision pairs below.
    def fmt_of(parameter: str) -> str:
        return recipe.selections[parameter][0].format

    def group_if_same(parameters: list[str]) -> None:
        formats = {fmt_of(p) for p in parameters}
        if len(formats) == 1:
            recipe.group(parameters)
        else:
            raise ValueError(
                f"group {parameters}: formats disagree {sorted(formats)} "
                "(table must assign one format per fusion group)"
            )

    mlp_groups = 0
    text_layers = sorted({
        int(n.split("/")[2])
        for n in model.parameters
        if n.startswith("text/layers/")
    })
    for layer in text_layers:
        base = f"text/layers/{layer}/"
        if base + "attention/query" in model.parameters:
            group_if_same([base + "attention/query", base + "attention/key"])
            group_if_same([base + "attention/gate", base + "attention/value"])
        elif base + "gdn/query" in model.parameters:
            group_if_same([base + "gdn/query", base + "gdn/key"])
            group_if_same([base + "gdn/value", base + "gdn/z"])
        if base + "mlp/gate" in model.parameters:
            group_if_same([base + "mlp/gate", base + "mlp/up"])
            mlp_groups += 1
    if "mtp/layers/0/mlp/gate" in model.parameters:
        # MTP mlp: same dense SwiGLU requirement as text mlp. MTP attention
        # q/k/gate/v stay separate objects (projection.rows path, parameters.cpp:347-377).
        group_if_same(["mtp/layers/0/mlp/gate", "mtp/layers/0/mlp/up"])
        mlp_groups += 1
    # GDN a/b: stay separate Paired BF16 objects (weight_input.cpp:268-284).
    vision_groups = 0
    layer = 0
    while f"vision/layers/{layer}/attention/query" in model.parameters:
        base = f"vision/layers/{layer}/attention/"
        group_if_same([base + "query", base + "key", base + "value"])
        group_if_same([base + "query_bias", base + "key_bias", base + "value_bias"])
        vision_groups += 2
        layer += 1

    print(
        f"[direct_convert] fusion groups rebuilt: {mlp_groups} mlp, "
        f"{vision_groups} vision (qkv+bias pairs per layer); "
        "MTP attention + GDN a/b kept separate per runtime paths",
        flush=True,
    )
    print(f"[direct_convert] reassigned {assigned} parameters "
          f"(text rows {len(used_text)}/{len(text_rows)}, "
          f"vision rows {len(used_vision)}/{len(vision_rows)})", flush=True)
    for (fmt, method), count in sorted(counts.items()):
        print(f"[direct_convert]   {fmt:16s} {method:15s} {count}", flush=True)
