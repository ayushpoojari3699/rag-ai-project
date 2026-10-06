"""Measure DocuMind's retrieval accuracy and response time on your own questions.

1. Index your PDFs (upload them in the app, or run `python ingest.py`).
2. Copy eval_questions.example.json to eval_questions.json and write 15-20
   questions, each with the file and page where the answer really is.
3. Make sure Ollama is running, then:  python benchmark.py

Reports:
  hit@k  - share of questions where the correct page is among the k retrieved passages
  timings - median and 90th-percentile retrieval / generation / total time
"""

import json
import statistics
import sys
from pathlib import Path

from rag_core import DEFAULT_TOP_K, RAGEngine


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def main(path="eval_questions.json", k=DEFAULT_TOP_K, skip_llm=False):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    engine = RAGEngine()
    hits, timings = 0, {"retrieval": [], "generation": [], "total": []}

    for item in items:
        if skip_llm:
            import time
            t0 = time.perf_counter()
            retrieved = engine.retrieve(item["question"], k)
            ms = round((time.perf_counter() - t0) * 1000)
            sources = [{"file": d.metadata.get("source"), "page": d.metadata.get("page", 0) + 1}
                       for d, _ in retrieved]
            t = {"retrieval": ms, "generation": 0, "total": ms}
        else:
            res = engine.answer(item["question"], k)
            sources, t = res["sources"], res["timings_ms"]
        ok = any(s["file"] == item["file"] and s["page"] == item["page"] for s in sources)
        hits += ok
        for key in timings:
            timings[key].append(t[key])
        print(f"[{'HIT ' if ok else 'MISS'}] {t['total']:>6} ms  {item['question']}")

    n = len(items)
    print(f"\nQuestions: {n}   hit@{k}: {hits}/{n} = {hits / n:.0%}")
    for key, vals in timings.items():
        if any(vals):
            print(f"{key:>10}: median {statistics.median(vals) / 1000:.2f} s, "
                  f"p90 {pct(vals, 0.9) / 1000:.2f} s")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "eval_questions.json", skip_llm="--retrieval-only" in sys.argv)
