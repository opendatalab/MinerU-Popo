#!/usr/bin/env python3
"""Quantization output-equivalence regression check for MinerU-Popo on vLLM.

Starts from the official ``output_cases`` example documents, builds
deterministic single-chunk "Title Level Analysis" prompts, sends them with
greedy decoding (``temperature=0``) to two OpenAI-compatible vLLM endpoints
serving the same checkpoint, and compares the completions token-by-token.

What this checks
----------------
Whether switching one serving engine (e.g. adding vLLM's online FP8 weight
quantization via ``--quantization fp8``) changes the model's outputs vs the
bfloat16 baseline. It is an engine-vs-engine equivalence check, **not** an
absolute accuracy benchmark: these prompts are simplified single-chunk
versions of the per-page prompts the full pipeline assembles, so predicted
levels may deviate from the reference trees in ``output_cases``. A small
number of flips on borderline blocks is expected under quantization noise
and does not by itself indicate quality loss.

Recipe
------
.. code-block:: bash

    # Terminal 1 - bfloat16 baseline (default serving):
    vllm serve /path/to/MinerU-Popo --served-model-name Popo --port 8010 \
        --max-model-len 16384 --max-num-seqs 4 --trust-remote-code

    # Terminal 2 - candidate (online FP8):
    vllm serve /path/to/MinerU-Popo --served-model-name Popo --port 8011 \
        --max-model-len 16384 --max-num-seqs 4 --trust-remote-code --quantization fp8

    # Compare:
    python eval/quant_equivalence_check.py --url-b http://127.0.0.1:8011/v1

Speed numbers are only comparable between the two instances when both see
the same machine load: run on an otherwise-idle GPU, or keep any background
traffic constant throughout the run.

Exit code: 0 if every sample is byte-identical, 1 otherwise.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_prompt(tree_path, max_blocks):
    """Deterministically render a tree JSON into a Title-Level prompt."""
    tree = json.load(open(tree_path, encoding="utf-8"))
    blocks = []

    def walk(node):
        if node.get("type") != "root":
            txt = ((node.get("title") or "") +
                   (" " + node["content"] if node.get("content") else "")).strip()
            if txt:
                locs = node.get("location") or []
                bbox = locs[0].get("bbox") if locs and locs[0].get("bbox") else [0, 0, 0, 0]
                page = locs[0].get("page", 0) if locs else 0
                blocks.append((len(blocks) + 1, page, bbox, txt[:200]))
        for child in node.get("children") or []:
            walk(child)

    walk(tree)
    lines = ["<|id|>%d<|page|>%d<|box|>%s<|content|>%s" %
             (i, p, " ".join(str(x) for x in b), c)
             for i, p, b, c in blocks[:max_blocks]]
    return "<image>\nTitle Level Analysis: " + "\n".join(lines)


def chat(url, model, key, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    dt = time.time() - t0
    used = data.get("usage", {}).get("completion_tokens")
    text = data["choices"][0]["message"]["content"]
    return text, (used or len(text)), dt


def diff_levels(a, b):
    """Return per-block level disagreements found in the two outputs."""
    pat = re.compile(r"<\|id\|>(\d+)<\|level\|>(-?\d+)")
    la, lb = dict(pat.findall(a)), dict(pat.findall(b))
    return [(k, la.get(k), lb.get(k)) for k in sorted(set(la) | set(lb), key=int)
            if la.get(k) != lb.get(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases-dir", default=os.path.join(REPO_ROOT, "output_cases"))
    ap.add_argument("--samples", default="1,2,3",
                    help="comma-separated tree ids under output_cases/trees")
    ap.add_argument("--blocks", type=int, default=60)
    ap.add_argument("--url-a", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--url-b", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--key-a", default=os.environ.get("KEY_A", "EMPTY"))
    ap.add_argument("--key-b", default=os.environ.get("KEY_B", "EMPTY"))
    ap.add_argument("--model", default="Popo")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--repeat", type=int, default=2,
                    help="runs per engine per sample, alternating order, to expose "
                         "load-induced variance in the speed comparison")
    args = ap.parse_args()

    all_identical = True
    for sid in [s.strip() for s in args.samples.split(",") if s.strip()]:
        path = os.path.join(args.cases_dir, "trees", "%s.json" % sid)
        if not os.path.exists(path):
            sys.exit("missing sample: %s" % path)
        prompt = build_prompt(path, args.blocks)

        outs = []
        for i in range(args.repeat):  # alternate order to de-bias speed readings
            outs.append(("a", chat(args.url_a, args.model, args.key_a, prompt, args.max_tokens)))
            outs.append(("b", chat(args.url_b, args.model, args.key_b, prompt, args.max_tokens)))
        text_a = next(t for eng, (t, _, _) in outs if eng == "a")
        text_b = next(t for eng, (t, _, _) in outs if eng == "b")

        rates = {"a": [], "b": []}
        for eng, (_, toks, dt) in outs:
            rates[eng].append(toks / dt)
        identical = text_a == text_b
        all_identical &= identical
        print("sample %s | identical=%s | tokens=%d"
              " | a %.1f tok/s | b %.1f tok/s" %
              (sid, identical, len(text_a),
               max(rates["a"]), max(rates["b"])))
        if not identical:
            for bid, va, vb in diff_levels(text_a, text_b):
                print("  block %s: a=%s b=%s" % (bid, va, vb))

    sys.exit(0 if all_identical else 1)


if __name__ == "__main__":
    main()
