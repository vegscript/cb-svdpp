from __future__ import annotations

from recsys_lab.experiments.common import build_run_id, build_runtime_metadata


class _FakeProcess:
    def cpu_affinity(self) -> list[int]:
        return [2, 4, 6]


def test_build_runtime_metadata_includes_thread_env_and_cpu_fingerprint(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)

    monkeypatch.setattr("recsys_lab.experiments.common.platform.python_version", lambda: "3.11.9")
    monkeypatch.setattr("recsys_lab.experiments.common.platform.platform", lambda: "test-platform")
    monkeypatch.setattr("recsys_lab.experiments.common.platform.node", lambda: "test-node")
    monkeypatch.setattr("recsys_lab.experiments.common.platform.processor", lambda: "test-cpu")
    monkeypatch.setattr(
        "recsys_lab.experiments.common.psutil.cpu_count",
        lambda logical=True: 16 if logical else 8,
    )
    monkeypatch.setattr("recsys_lab.experiments.common.psutil.Process", lambda: _FakeProcess())
    monkeypatch.setattr(
        "recsys_lab.experiments.common.threadpool_info",
        lambda: [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "threading_layer": "pthreads",
                "num_threads": 2,
                "version": "0.3.28",
            }
        ],
    )

    runtime = build_runtime_metadata(
        device_profile_name="hpc_cpu",
        runtime_dtype="float32",
        device_config_payload={
            "threading": {
                "omp_num_threads": 4,
                "blas_threads": 2,
            }
        },
    )

    assert runtime["device_profile"] == "hpc_cpu"
    assert runtime["dtype"] == "float32"
    assert runtime["python_version"] == "3.11.9"
    assert runtime["platform"] == "test-platform"
    assert runtime["hostname"] == "test-node"
    assert runtime["processor"] == "test-cpu"
    assert runtime["cpu_logical_count"] == 16
    assert runtime["cpu_physical_count"] == 8
    assert runtime["cpu_affinity"] == [2, 4, 6]
    assert runtime["cpu_affinity_count"] == 3
    assert runtime["threading"]["omp_num_threads"] == 4
    assert runtime["threading"]["blas_threads"] == 2
    assert runtime["threading"]["env_omp_num_threads"] == "4"
    assert runtime["threading"]["env_mkl_num_threads"] == "2"
    assert runtime["threading"]["env_openblas_num_threads"] == "2"
    assert runtime["threading"]["env_numexpr_num_threads"] is None
    assert runtime["threadpools"] == [
        {
            "user_api": "blas",
            "internal_api": "openblas",
            "prefix": "libopenblas",
            "threading_layer": "pthreads",
            "num_threads": 2,
            "version": "0.3.28",
        }
    ]

def test_build_run_id_includes_split_id_when_present() -> None:
    run_id = build_run_id(
        timestamp="2026-04-16T080000Z",
        dataset_short_name="ml1m",
        model_name="biased_mf",
        device_profile_name="local_i5_2500k_24gb",
        model_seed=1,
        split_id_value="benchmark_random_v1_tr080_va010_s003",
    )

    assert run_id == (
        "2026-04-16T080000Z_ml1m_biased_mf_local_i5_2500k_24gb_"
        "benchmark_random_v1_tr080_va010_s003_s001"
    )

