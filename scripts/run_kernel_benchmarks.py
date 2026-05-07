from __future__ import annotations

import argparse
from pathlib import Path

from recsys_lab.benchmarks.kernel_harness import run_kernel_benchmark
from recsys_lab.benchmarks.synthetic_kernel_cases import (
    build_synthetic_kernel_cases,
    get_synthetic_kernel_case,
)
from recsys_lab.benchmarks.writers import (
    write_kernel_benchmark_json,
    write_kernel_benchmark_summary_csv,
)

DEFAULT_OUTPUT_DIR = Path("artifacts") / "benchmarks" / "kernel"
SYNTHETIC_CASE_NAMES = tuple(case.model for case in build_synthetic_kernel_cases())


def main() -> int:
    args = _parse_args()
    if args.warmup_repeats < 1:
        raise SystemExit("--warmup-repeats must be at least 1 for warm-run benchmarks")
    cases = _select_cases(args.case)
    payloads = []
    for case in cases:
        payload = run_kernel_benchmark(
            case,
            warmup_repeats=args.warmup_repeats,
            timed_repeats=args.timed_repeats,
            epochs_per_repeat=args.epochs_per_repeat,
        )
        write_kernel_benchmark_json(payload, args.output_dir)
        payloads.append(payload)

    summary_path = write_kernel_benchmark_summary_csv(
        payloads,
        args.output_dir / "kernel_benchmark_summary.csv",
    )
    print(f"Wrote {len(payloads)} kernel benchmark payload(s) to {args.output_dir}")
    print(f"Wrote summary CSV to {summary_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic tiny kernel benchmarks.")
    parser.add_argument(
        "--case",
        choices=("all", *SYNTHETIC_CASE_NAMES),
        default="all",
        help="Synthetic kernel case to run.",
    )
    parser.add_argument(
        "--timed-repeats",
        type=int,
        default=5,
        help="Number of timed repeats per case.",
    )
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=1,
        help="Number of warmup repeats excluded from timed results.",
    )
    parser.add_argument(
        "--epochs-per-repeat",
        type=int,
        default=1,
        help="Kernel epochs executed inside each repeat timer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Benchmark output directory.",
    )
    return parser.parse_args()


def _select_cases(case_name: str):
    if case_name == "all":
        return build_synthetic_kernel_cases()
    return (get_synthetic_kernel_case(case_name),)


if __name__ == "__main__":
    raise SystemExit(main())
