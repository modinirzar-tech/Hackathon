#!/usr/bin/env python3
"""Run the locally-measurable KPI gates and print the scorecard. Exit code 1 if any measured KPI FAILs."""
import asyncio, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("DB_PATH", ":memory:")
os.environ.setdefault("MANIFEST_LOCK", tempfile.mktemp())

from assistant.eval.kpi import evaluate, summarize  # noqa: E402
from assistant.runtime import build_runtime  # noqa: E402

async def main() -> int:
    kpis = await evaluate(build_runtime())
    for k in kpis:
        print(f"[{k.status:12}] KPI-{k.id:02d} {k.name:<58} thr {k.threshold:<12} value {k.value}  {k.note}")
    s = summarize(kpis)
    print(f"\n{s['pass']} PASS / {s['fail']} FAIL / {s['not_measured']} NOT_MEASURED of {s['total']}")
    if "--json" in sys.argv:
        Path("kpi_report.json").write_text(json.dumps([k.dict() for k in kpis], indent=2))
    return 1 if s["fail"] else 0

sys.exit(asyncio.run(main()))
