"""
Audit runner — sends probes to an AI API and records raw completions.

CLI usage:
    python -m audit.runner --category hiring --n-probes 30
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

from audit.probes import Probe, render_probes, POSITIVE_KEYWORDS


class AuditRunner:
    def __init__(
        self,
        model: str,
        api_key: str,
        results_dir: Path,
        max_tokens: int = 256,
    ):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens

    def run(
        self,
        probes: list[Probe],
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> list[dict]:
        results = []
        for probe in probes:
            result = self._call_with_retry(probe, max_retries, delay)
            results.append(result)
        return results

    def _call_with_retry(self, probe: Probe, max_retries: int, delay: float) -> dict:
        last_error: Optional[str] = None
        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": probe.text}],
                )
                latency_ms = int((time.monotonic() - t0) * 1000)
                completion = response.content[0].text if response.content else ""
                return {
                    "probe_id": probe.probe_id,
                    "category": probe.category,
                    "group": probe.group,
                    "subgroup": probe.subgroup,
                    "prompt": probe.text,
                    "completion": completion,
                    "latency_ms": latency_ms,
                    "error": None,
                }
            except anthropic.RateLimitError:
                wait = delay * (2 ** attempt)
                time.sleep(wait)
                last_error = "rate_limit"
            except anthropic.APIError as e:
                wait = delay * (2 ** attempt)
                time.sleep(wait)
                last_error = str(e)

        return {
            "probe_id": probe.probe_id,
            "category": probe.category,
            "group": probe.group,
            "subgroup": probe.subgroup,
            "prompt": probe.text,
            "completion": "",
            "latency_ms": -1,
            "error": last_error or "unknown",
        }

    def save(self, results: list[dict], tag: Optional[str] = None) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"{ts}{'_' + tag if tag else ''}.json"
        out_path = self.results_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        # Always overwrite the "latest" symlink-equivalent (plain copy pointer)
        latest = self.results_dir / "latest.json"
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a bias audit against an AI API.")
    p.add_argument("--category", default="hiring", choices=["hiring", "lending", "medical"])
    p.add_argument("--n-probes", type=int, default=0,
                   help="Max probes to run (0 = all generated probes)")
    p.add_argument("--model", default=None)
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = args.model or os.environ.get("AUDIT_MODEL", "claude-haiku-4-5-20251001")
    results_dir = Path(args.results_dir or os.environ.get("RESULTS_DIR", "data/results"))

    probes = render_probes(args.category)
    if args.n_probes and args.n_probes < len(probes):
        probes = probes[: args.n_probes]

    print(f"Running {len(probes)} probes against {model} ...")
    runner = AuditRunner(model=model, api_key=api_key, results_dir=results_dir)
    results = runner.run(probes)
    out = runner.save(results, tag=args.category)

    errors = sum(1 for r in results if r["error"])
    print(f"Done. {len(results) - errors}/{len(results)} succeeded. Results: {out}")


if __name__ == "__main__":
    main()
