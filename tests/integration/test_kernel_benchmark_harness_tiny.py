import json

from recsys_lab.benchmarks.kernel_harness import run_kernel_benchmark
from recsys_lab.benchmarks.synthetic_kernel_cases import build_synthetic_kernel_cases
from recsys_lab.benchmarks.writers import (
    write_kernel_benchmark_json,
    write_kernel_benchmark_summary_csv,
)


def test_kernel_benchmark_harness_tiny_runs_all_six_cases(tmp_path) -> None:
    output_dir = tmp_path / "artifacts" / "benchmarks" / "kernel"
    payloads = []

    for case in build_synthetic_kernel_cases():
        payload = run_kernel_benchmark(case, warmup_repeats=1, timed_repeats=2)
        payloads.append(payload)
        output_path = write_kernel_benchmark_json(payload, output_dir)

        assert output_path.exists()
        written = json.loads(output_path.read_text(encoding="utf-8"))
        assert written["benchmark_version"] == "kernel_benchmark_harness_v1"
        assert len(written["repeat_wall_seconds"]) == 2
        assert all(seconds > 0.0 for seconds in written["repeat_wall_seconds"])
        assert written["mean_wall_seconds"] > 0.0
        assert written["ratings_per_second_mean"] > 0.0
        assert written["state_checks"]["finite_parameters_after"] is True
        assert written["compile_excluded"] is True
        assert written["state_copy_excluded"] is True

    summary_path = write_kernel_benchmark_summary_csv(payloads, output_dir / "kernel_benchmark_summary.csv")

    assert len(payloads) == 6
    assert summary_path.exists()
