from pathlib import Path
from typing import Iterable, List

import pytest

torch = pytest.importorskip("torch")

from fastloader.gpu_dataloader import GPUCachedPathDataLoader


def _write_tensor(path: Path, value: int) -> torch.Tensor:
    tensor = torch.full((2,), value, dtype=torch.float32)
    torch.save(tensor, path)
    return tensor


def test_dataloader_prefetches_and_returns_gpu_batches(tmp_path: Path) -> None:
    paths: List[Path] = []
    expected_tensors: List[torch.Tensor] = []
    for value in range(3):
        path = tmp_path / f"tensor_{value}.pt"
        expected_tensors.append(_write_tensor(path, value))
        paths.append(path)

    # Underlying loader emits path batches of size two then one.
    base_loader: Iterable[list[Path]] = [[paths[0], paths[1]], [paths[2]]]

    loader = GPUCachedPathDataLoader(
        base_loader,
        prefetch_batches=2,
        collate_fn=torch.stack,
        build_batch_fn=lambda gpu_batch, original, _: {
            "gpu": gpu_batch,
            "original": list(original),
        },
        device="cpu",
        asynchronous=False,
    )

    try:
        batches = list(loader)
    finally:
        loader.close()

    assert len(batches) == 2

    first = batches[0]
    assert torch.equal(first["gpu"], torch.stack(expected_tensors[:2]))
    assert first["original"] == [paths[0], paths[1]]

    second = batches[1]
    assert torch.equal(second["gpu"], torch.stack([expected_tensors[2]]))
    assert second["original"] == [paths[2]]


def test_invalid_prefetch_batches_raises(tmp_path: Path) -> None:
    base_loader: Iterable[list[Path]] = [[tmp_path / "dummy.pt"]]

    try:
        GPUCachedPathDataLoader(
            base_loader,
            prefetch_batches=0,
            device="cpu",
            asynchronous=False,
        )
    except ValueError as exc:
        assert "prefetch_batches" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for invalid prefetch_batches")
