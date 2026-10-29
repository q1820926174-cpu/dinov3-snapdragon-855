#!/usr/bin/env python3
"""Run an exported FP32 ONNX model on the exact held-out RAW inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="features_fp32", help="Output filename prefix")
    args = parser.parse_args()
    records = [json.loads(line) for line in
               (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if r["split"] == "test"]
    if not records:
        raise ValueError("manifest contains no test samples")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise ValueError("expected one ONNX input")
    input_meta = session.get_inputs()[0]
    if input_meta.type != "tensor(float)":
        raise ValueError(f"expected a float32 input, got {input_meta.type}")
    outputs = {o.name for o in session.get_outputs()}
    wanted = ["dense_features", "global_embedding"]
    if not set(wanted).issubset(outputs):
        raise ValueError(f"missing expected output names: {outputs}")
    dense, global_features = [], []
    for i, record in enumerate(records):
        if not record.get("raw_file") or not record.get("shape"):
            raise ValueError("manifest lacks RAW inputs; generate the dataset without --dry-run")
        raw = np.fromfile(args.dataset / record["raw_file"], dtype=np.float32)
        shape = record["shape"]
        if raw.size != int(np.prod(shape)):
            raise ValueError(f"unexpected RAW size for {record['raw_file']}")
        raw = raw.reshape(shape)
        if record["layout"] == "NHWC":
            raw = np.transpose(raw, (0, 3, 1, 2))
        elif record["layout"] != "NCHW":
            raise ValueError(f"unknown input layout: {record['layout']}")
        expected = input_meta.shape
        if len(expected) != raw.ndim or any(isinstance(d, int) and d != s for d, s in zip(expected, raw.shape)):
            raise ValueError(f"ONNX expects {expected}; RAW input is {raw.shape}")
        d, g = session.run(wanted, {input_meta.name: np.ascontiguousarray(raw)})
        if not np.isfinite(d).all() or not np.isfinite(g).all():
            raise ValueError(f"non-finite model output for {record['raw_file']}")
        dense.append(d)
        global_features.append(g)
        print(f"{i + 1}/{len(records)} {record['game']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / (args.prefix + "_dense.npy"), np.concatenate(dense, axis=0))
    np.save(args.out / (args.prefix + ".npy"), np.concatenate(global_features, axis=0))
    (args.out / "feature_manifest.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
