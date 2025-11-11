from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fastloader.gpu_feature_store import GPUFeatureStore


@pytest.fixture()
def feature_dir(tmp_path: Path) -> Path:
    return tmp_path


def _make_tensor(value: int) -> torch.Tensor:
    return torch.full((4,), value, dtype=torch.float32)


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    torch.save(tensor, path)


def test_prefetch_and_get_returns_tensor(feature_dir: Path) -> None:
    tensor_path = feature_dir / "feature.pt"
    expected = _make_tensor(1)
    _save_tensor(tensor_path, expected)

    store = GPUFeatureStore(device="cpu", asynchronous=False)
    try:
        store.prefetch([tensor_path])
        loaded = store.get(tensor_path)
    finally:
        store.close()

    assert torch.equal(loaded, expected)


def test_cache_eviction_reloads_when_reaccessed(feature_dir: Path) -> None:
    first_path = feature_dir / "first.pt"
    second_path = feature_dir / "second.pt"
    _save_tensor(first_path, _make_tensor(1))
    _save_tensor(second_path, _make_tensor(2))

    load_counts: Counter[str] = Counter()

    def load_fn(path: Path) -> torch.Tensor:
        load_counts[str(path)] += 1
        return torch.load(path)

    tensor_bytes = _make_tensor(0).element_size() * _make_tensor(0).nelement()
    store = GPUFeatureStore(
        device="cpu",
        asynchronous=False,
        max_cache_bytes=tensor_bytes,
        load_fn=load_fn,
    )

    try:
        store.get(first_path)
        store.get(second_path)
        assert load_counts[str(first_path)] == 1
        assert load_counts[str(second_path)] == 1

        store.get(first_path)
        assert load_counts[str(first_path)] == 2
    finally:
        store.close()


def test_get_batch_with_collate_fn(feature_dir: Path) -> None:
    paths = []
    tensors = []
    for i in range(3):
        tensor = _make_tensor(i)
        path = feature_dir / f"tensor_{i}.pt"
        _save_tensor(path, tensor)
        paths.append(path)
        tensors.append(tensor)

    store = GPUFeatureStore(device="cpu", asynchronous=False)
    try:
        stacked = store.get_batch(paths, collate_fn=torch.stack)
    finally:
        store.close()

    assert isinstance(stacked, torch.Tensor)
    assert stacked.shape == (3, 4)
    for idx, expected in enumerate(tensors):
        assert torch.equal(stacked[idx], expected)
