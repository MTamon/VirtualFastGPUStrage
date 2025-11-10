"""DataLoader wrapper that transparently serves GPU-cached features."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Deque, Iterable, Iterator, Sequence, Tuple, TypeVar

from .gpu_feature_store import GPUFeatureStore

BatchT = TypeVar("BatchT")

Paths = Sequence[str | Path]
CollateFn = Callable[[Sequence[object]], object]
PathExtractor = Callable[[BatchT], Paths]
BatchBuilder = Callable[[object, BatchT, Paths], object]


class GPUCachedPathDataLoader:
    """Wrap a path-emitting loader and return GPU-resident batches."""

    def __init__(
        self,
        loader: Iterable[BatchT],
        *,
        prefetch_batches: int = 2,
        collate_fn: CollateFn | None = None,
        path_extractor: PathExtractor | None = None,
        build_batch_fn: BatchBuilder | None = None,
        store: GPUFeatureStore | None = None,
        **store_kwargs,
    ) -> None:
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be >= 1")
        self._loader = loader
        self._prefetch_batches = prefetch_batches
        self._collate_fn = collate_fn
        self._path_extractor = path_extractor
        self._build_batch_fn = build_batch_fn
        self._store = store or GPUFeatureStore(**store_kwargs)
        self._owns_store = store is None

    def __iter__(self) -> Iterator[object]:
        iterator = iter(self._loader)
        buffer: Deque[Tuple[BatchT, list[str | Path]]] = deque()

        def enqueue(batch: BatchT) -> None:
            paths = self._extract_paths(batch)
            canonical = list(paths)
            if canonical:
                self._store.prefetch(canonical)
            buffer.append((batch, canonical))

        try:
            while len(buffer) < self._prefetch_batches:
                batch = next(iterator)
                enqueue(batch)
        except StopIteration:
            iterator = None  # type: ignore[assignment]

        while buffer:
            batch, paths = buffer.popleft()
            gpu_items = self._store.get_batch(paths, collate_fn=self._collate_fn)
            if self._build_batch_fn is not None:
                yield self._build_batch_fn(gpu_items, batch, paths)
            else:
                yield gpu_items

            if iterator is None:
                continue
            try:
                next_batch = next(iterator)
            except StopIteration:
                iterator = None
            else:
                enqueue(next_batch)

    def __len__(self) -> int:
        if hasattr(self._loader, "__len__"):
            return len(self._loader)  # type: ignore[arg-type]
        raise TypeError("Underlying loader does not implement __len__")

    def __getattr__(self, name: str) -> object:
        try:
            return getattr(self._loader, name)
        except AttributeError as exc:  # pragma: no cover - passthrough
            raise AttributeError(f"{type(self).__name__!s} has no attribute '{name}'") from exc

    @property
    def store(self) -> GPUFeatureStore:
        return self._store

    def close(self) -> None:
        if self._owns_store:
            self._store.close()

    def __enter__(self) -> "GPUCachedPathDataLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def _extract_paths(self, batch: BatchT) -> Paths:
        if self._path_extractor is not None:
            paths = self._path_extractor(batch)
        else:
            paths = batch  # type: ignore[assignment]
        if isinstance(paths, (str, bytes)):
            raise TypeError("Batch must be a sequence of paths; got str/bytes")
        if not isinstance(paths, Sequence):
            raise TypeError("path_extractor must return a Sequence of paths")
        return paths


__all__ = ["GPUCachedPathDataLoader"]
