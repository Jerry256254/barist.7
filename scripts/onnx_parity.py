#!/usr/bin/env python3
"""ONNX/quant parity: fp32 vs fp16 vs int8 vs ONNX max score deviation + verdict agreement.
Writes reports/onnx_parity.json. Usage: --bundle /tmp/bundle_b1"""
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/tmp/bundle_b1")
    ap.add_argument("--onnx", default=None, help="onnx file name inside the bundle (default: barist1.onnx or barist2.onnx if present)")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    b = Path(args.bundle)

    import torch
    sys.path.insert(0, str(b))
    import censer as C

    # fixed sample: heldout2 + cs_benign + test
    texts = []
    for f in ["data/processed/mustcatch_heldout2.jsonl", "data/processed/cs_benign_eval.jsonl", "data/processed/test.jsonl"]:
        for i, l in enumerate(open(Path(__file__).parent.parent / f, encoding="utf-8")):
            if i >= args.n // 3:
                break
            texts.append(json.loads(l)["text"])
    texts = texts[: args.n]

    pipe = C.CenserPipeline(b, device="cpu")
    fp32 = [r["score"] for r in pipe.predict_batch(texts)]

    # fp16
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp())
    for f in b.iterdir():
        if f.is_file():
            shutil.copy2(f, tmp / f.name)
    from safetensors.torch import load_file, save_file
    st = load_file(tmp / "model.safetensors")
    save_file({k: v.half() for k, v in st.items()}, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    p16 = C.CenserPipeline(tmp, device="cpu")
    fp16 = [r["score"] for r in p16.predict_batch(texts)]

    # int8
    pq = C.CenserPipeline(b, device="cpu", prefer_quant=True)
    int8 = [r["score"] for r in pq.predict_batch(texts)]

    # onnx
    onnx = None
    try:
        import onnxruntime as ort
        import numpy as np
        onnx_name = args.onnx or next((n for n in ("barist7.onnx","barist4.onnx","barist3.onnx","barist2.onnx","barist1.onnx") if (b / n).exists()), "barist7.onnx")
        sess = ort.InferenceSession(str(b / onnx_name), providers=["CPUExecutionProvider"])
        tok = C.Tokenizer.from_file(str(b / "tokenizer.json"))
        bos, user, asst = tok.token_to_id("[BOS]"), tok.token_to_id("[USER]"), tok.token_to_id("[ASSISTANT]")
        onnx = []
        for t in texts:
            ids = [bos, user, *tok.encode(t[:4000], add_special_tokens=False).ids, asst][:4096]
            arr = np.array([ids], dtype=np.int64)
            logits = sess.run(None, {"input_ids": arr})[0][0]
            import torch as _t
            p = float((_t.tensor(logits) / pipe.temperature).sigmoid()[1])
            onnx.append(round(p, 4))
    except ImportError:
        print("onnxruntime not available, skipping onnx")

    def cmp(a, b_):
        diffs = [abs(x - y) for x, y in zip(a, b_)]
        agree = sum(1 for x, y in zip(a, b_) if (x >= 0.5) == (y >= 0.5))
        return {"max_abs_diff": round(max(diffs), 6), "verdict_agreement": round(agree / len(a), 4)}

    rep = {"n": len(texts), "fp32_vs_fp16": cmp(fp32, fp16), "fp32_vs_int8": cmp(fp32, int8)}
    if onnx:
        rep["fp32_vs_onnx"] = cmp(fp32, onnx)
    rep["gate"] = "fp32_vs_onnx max_abs_diff <= 1e-3 (target 1e-4); fp16/int8 reported informationally"
    rep["pass"] = rep.get("fp32_vs_onnx", {}).get("max_abs_diff", 1) <= 1e-3 and rep.get("fp32_vs_onnx", {}).get("verdict_agreement", 0) >= 0.99
    out = Path(__file__).parent.parent / "reports/onnx_parity.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
