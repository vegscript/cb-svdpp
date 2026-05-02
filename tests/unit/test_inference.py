import numpy as np

from recsys_lab.models.inference import build_unique_user_context_batch, build_user_context_cache


def test_build_unique_user_context_batch_materializes_only_unique_users() -> None:
    calls: list[int] = []

    def _context_for_user(user_id: int) -> np.ndarray:
        calls.append(user_id)
        return np.asarray([user_id, user_id + 0.5], dtype=np.float64)

    users, inverse, contexts = build_unique_user_context_batch(
        user_ids=np.asarray([3, 1, 3, 2, 1], dtype=np.int32),
        context_dim=2,
        build_user_context=_context_for_user,
    )

    assert users.tolist() == [3, 1, 3, 2, 1]
    assert inverse.tolist() == [2, 0, 2, 1, 0]
    assert contexts.shape == (3, 2)
    assert calls == [1, 2, 3]
    np.testing.assert_allclose(
        contexts,
        np.asarray([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=np.float64),
    )


def test_build_user_context_cache_materializes_each_user_once() -> None:
    calls: list[int] = []

    def _context_for_user(user_id: int) -> np.ndarray:
        calls.append(user_id)
        return np.asarray([user_id, user_id + 0.5], dtype=np.float64)

    cache = build_user_context_cache(
        n_users=4,
        context_dim=2,
        build_user_context=_context_for_user,
    )

    assert calls == [0, 1, 2, 3]
    np.testing.assert_allclose(
        cache,
        np.asarray(
            [
                [0.0, 0.5],
                [1.0, 1.5],
                [2.0, 2.5],
                [3.0, 3.5],
            ],
            dtype=np.float64,
        ),
    )
