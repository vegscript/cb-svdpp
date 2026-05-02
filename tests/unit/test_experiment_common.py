from __future__ import annotations

from pathlib import Path

from recsys_lab.experiments.common import (
    build_run_id,
    build_runtime_metadata,
    offset_utc_timestamp,
    reserve_timestamped_artifact_dir,
)


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
            "metadata": {
                "status": "validated_hpc",
            },
            "device_profile": {
                "name": "hpc_cpu",
                "compute_class": "hpc_cpu",
                "cpu_model": "test_hpc_cpu",
                "logical_threads": 16,
                "physical_cores": 8,
                "ram_gb": 128,
                "gpu_enabled": False,
            },
            "storage": {
                "cache_preference": "local_scratch",
                "archive_preference": "shared_storage",
            },
            "threading": {
                "omp_num_threads": 4,
                "blas_threads": 2,
            },
            "resource_limits": {
                "ram_guardrail_fraction": 0.8,
            },
            "precision": {
                "default_dtype": "float32",
                "reference_dtype": "float64",
            }
        },
    )

    assert runtime["device_profile"] == "hpc_cpu"
    assert runtime["dtype"] == "float32"
    assert runtime["python_version"] == "3.11.9"
    assert runtime["platform"] == "test-platform"
    assert runtime["hostname"] == "test-node"
    assert runtime["processor"] == "test-cpu"
    assert runtime["device_profile_contract"] == {
        "contract_version": "device_profile_contract_v1",
        "profile_name": "hpc_cpu",
        "compute_class": "hpc_cpu",
        "metadata_status": "validated_hpc",
        "claim_eligible": True,
        "blocking_reasons": [],
        "ram_guardrail_fraction": 0.8,
    }
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


def test_offset_utc_timestamp_advances_without_changing_format() -> None:
    assert offset_utc_timestamp("2026-04-21T123456Z", seconds=1) == "2026-04-21T123457Z"


def test_reserve_timestamped_artifact_dir_advances_timestamp_on_collision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "recsys_lab.experiments.common.utc_timestamp",
        lambda: "2026-04-21T123456Z",
    )

    first_timestamp, first_id, first_dir = reserve_timestamped_artifact_dir(
        artifacts_root=tmp_path / "artifacts" / "benchmarks",
        id_from_timestamp=lambda timestamp: f"{timestamp}_ml100k_test_local",
    )
    second_timestamp, second_id, second_dir = reserve_timestamped_artifact_dir(
        artifacts_root=tmp_path / "artifacts" / "benchmarks",
        id_from_timestamp=lambda timestamp: f"{timestamp}_ml100k_test_local",
    )

    assert first_timestamp == "2026-04-21T123456Z"
    assert second_timestamp == "2026-04-21T123457Z"
    assert first_id == "2026-04-21T123456Z_ml100k_test_local"
    assert second_id == "2026-04-21T123457Z_ml100k_test_local"
    assert first_dir.is_dir()
    assert second_dir.is_dir()


def test_build_run_id_includes_split_id_when_present() -> None:
    run_id = build_run_id(
        timestamp="2026-04-16T080000Z",
        dataset_short_name="ml1m",
        model_name="biased_mf",
        device_profile_name="local_i5_2500k_24gb",
        model_seed=1,
        split_id_value="benchmark_random_v1_tr080_va010_s003",
    )

    assert run_id == ("2026-04-16T080000Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001")
