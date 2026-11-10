"""CLI: python -m src.jobs.eval_nightly --split dev [--limit N]

Smoke test with --limit 5. Run the full dev eval only after a real change (each question costs
3 LLM calls: answer, accuracy judge, faithfulness judge; all cached on disk).
"""
import argparse
import json
import sys

from src.config import load_config
from src.eval.rag_metrics import run_eval
from src.ingestion.loader import setup_logging
from src.llm.llm_client import BudgetNearlyExhausted


def _fmt(x, pct=True):
    if x is None:
        return "n/a"
    return f"{x:.3f}" if pct else f"{x:.0f}"


def print_summary(s: dict, k: int) -> None:
    print("\n=== eval summary ===")
    print(f"split={s['split']} limit={s['limit']} n={s['n']} errors={s['errors']} idk={s['idk']}")
    print(f"recall@{k}={_fmt(s[f'recall_at_{k}'])}  MRR={_fmt(s['mrr'])}  "
          f"accuracy={_fmt(s['accuracy'])}  faithfulness={_fmt(s['faithfulness'])} (n={s['faithfulness_n']})")
    print(f"latency p50={_fmt(s['latency_p50_ms'], False)} ms  p95={_fmt(s['latency_p95_ms'], False)} ms "
          f"(over {s['latency_n']} uncached questions)")
    print(f"stage p50 (ms): {json.dumps({k_: round(v) if v is not None else None for k_, v in s['stage_p50_ms'].items()})}")
    print(f"LLM calls served by: {s['served_by']}  | generation source: {s['generation_source']}")
    print(f"config: {s['config']}")
    print(f"run file: {s['run_file']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke test)")
    ap.add_argument("--final", action="store_true", help="allow the test split (Task 11 only)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg.data.log_file)
    try:
        summary = run_eval(args.split, limit=args.limit, cfg=cfg, allow_test=args.final)
    except BudgetNearlyExhausted as e:
        print(f"\nSTOPPED: {e}\nA partial run was saved. Tell the user before continuing.", file=sys.stderr)
        return 2
    print_summary(summary, cfg.eval.recall_k)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
