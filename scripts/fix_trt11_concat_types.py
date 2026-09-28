#!/usr/bin/env python3
"""Patch YOLO ONNX models for TensorRT 11 strong typing.

TensorRT 11 always runs in strongly-typed mode: every input of a Concat layer
must have the same data type. Some Ultralytics exports (e.g. the YOLOv26
end-to-end head) emit a Cast node producing a float32 tensor that is then
concatenated with float16 tensors, which TensorRT 10 tolerated but TensorRT 11
rejects with:

    IConcatenationLayer `inputs` must all be of the same type.

This script rewrites the offending Cast nodes so all Concat inputs share the
type of the first input. Run it once per ONNX file:

    python3 scripts/fix_trt11_concat_types.py dep/yolo26m_cs2_20260612.onnx
"""

import sys

import onnx
from onnx import TensorProto, shape_inference


def patch(path: str) -> None:
    model = onnx.load(path)
    # Populate value_info so intermediate tensor types are known.
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    types = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        types[vi.name] = vi.type.tensor_type.elem_type

    producer = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node
        # Ground truth for Cast outputs is the `to` attribute itself —
        # shape inference does not always reflect it.
        if node.op_type == "Cast":
            to = next((a.i for a in node.attribute if a.name == "to"), None)
            if to is not None:
                for out in node.output:
                    types[out] = to

    changed = 0
    for node in graph.node:
        if node.op_type != "Concat":
            continue
        input_types = [types.get(inp) for inp in node.input]
        known = [t for t in input_types if t]
        if not known or len(set(known)) <= 1:
            continue
        target = input_types[0] if input_types[0] else known[0]
        for inp, t in zip(node.input, input_types):
            if t is None or t == target:
                continue
            prod = producer.get(inp)
            if prod is not None and prod.op_type == "Cast":
                for attr in prod.attribute:
                    if attr.name == "to":
                        print(
                            f"{path}: {prod.name}: cast "
                            f"{TensorProto.DataType.Name(attr.i)} -> "
                            f"{TensorProto.DataType.Name(target)}"
                        )
                        attr.i = target
                        changed += 1
            else:
                print(
                    f"{path}: WARNING: mixed-type Concat '{node.name}', input "
                    f"'{inp}' is not Cast-produced; manual fix needed"
                )

    if changed:
        onnx.save(model, path)
    print(f"{path}: {changed} cast(s) patched")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} model.onnx [model2.onnx ...]")
    for p in sys.argv[1:]:
        patch(p)
