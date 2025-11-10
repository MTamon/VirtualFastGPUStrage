"""GPU-backed feature storage that serves tensors by file path."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, cast

import torch

TensorLike = Any
LoadFn = Callable[[Path], TensorLike]


_STOP = object()


@dataclass
class _CacheEntry:
    """Internal cache record for a single feature file."""

    path: str
    tensor: TensorLike | None = None
    size_bytes: int = 0
    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    loading: bool = False
    ready_event: torch.cuda.Event | None = None

    def reset_event(self) -> None:
        self.event = threading.Event()
        self.error = None
        self.ready_event = None
        self.loading = True
        self.tensor = None
        self.size_bytes = 0


class GPUFeatureStore:
    """Mirror feature tensors from storage onto GPU memory.

    The store accepts file paths for serialized ``.pt`` tensors. It keeps a
    bounded cache on the target device and streams new items from disk in a
    background worker thread. Existing PyTorch ``DataLoader`` instances can emit
    batches of paths, and the store will return the corresponding GPU-resident
    tensors via :meth:`get_batch`.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        max_cache_bytes: int | None = None,
        max_cache_items: int | None = None,
        load_fn: LoadFn | None = None,
        asynchronous: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if max_cache_bytes is not None and max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        if max_cache_items is not None and max_cache_items <= 0:
            raise ValueError("max_cache_items must be positive")
        self.max_cache_bytes = max_cache_bytes
        self.max_cache_items = max_cache_items
        self._load_fn: LoadFn = load_fn or (lambda p: torch.load(p, map_location="cpu"))
        self.asynchronous = asynchronous and self.device.type == "cuda"

        self._lock = threading.RLock()
        self._queue: "Queue[object]" = Queue()
        self._entries: MutableMapping[str, _CacheEntry] = {}
        self._lru: "OrderedDict[str, None]" = OrderedDict()
        self._current_bytes = 0
        self._closed = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._stream: torch.cuda.Stream | None = None
        if self.asynchronous:
            self._stream = torch.cuda.Stream(device=self.device)

    # ------------------------------------------------------------------
    # public API
    def prefetch(self, paths: Iterable[str | Path]) -> None:
        """Schedule the given paths for loading into GPU memory."""

        normalized = [self._canonical_path(p) for p in paths]
        with self._lock:
            if self._closed:
                raise RuntimeError("GPUFeatureStore has been closed")
            for path in normalized:
                entry = self._entries.get(path)
                if entry is not None and entry.tensor is not None:
                    # Already cached; refresh LRU order
                    self._lru[path] = None
                    self._lru.move_to_end(path)
                    continue

                if entry is None:
                    entry = _CacheEntry(path=path)
                    self._entries[path] = entry
                if entry.loading:
                    continue
                entry.reset_event()
                self._queue.put(path)

    def get(self, path: str | Path) -> TensorLike:
        """Return the GPU-resident tensor for ``path`` (blocking if necessary)."""

        normalized = self._canonical_path(path)
        self.prefetch([normalized])
        entry = self._entries[normalized]
        entry.event.wait()
        if entry.error is not None:
            raise RuntimeError(f"Failed to load feature '{normalized}'") from entry.error
        if entry.tensor is None:
            raise RuntimeError(f"Feature '{normalized}' is unavailable after loading")
        if entry.ready_event is not None:
            torch.cuda.current_stream(self.device).wait_event(entry.ready_event)
        with self._lock:
            self._lru[normalized] = None
            self._lru.move_to_end(normalized)
        return entry.tensor

    def get_batch(
        self,
        paths: Sequence[str | Path],
        *,
        collate_fn: Callable[[Sequence[TensorLike]], TensorLike] | None = None,
    ) -> TensorLike:
        """Fetch a batch of tensors for the provided ``paths``.

        Parameters
        ----------
        paths:
            Iterable of file paths referencing serialized tensors.
        collate_fn:
            Optional callable that receives the list of loaded items (on the
            target device) and returns a collated structure for consumption by a
            model or data collator.
        """

        items = [self.get(path) for path in paths]
        if collate_fn is not None:
            return collate_fn(items)
        return items

    def release(self, paths: Iterable[str | Path]) -> None:
        """Manually evict specific paths from the GPU cache."""

        normalized = [self._canonical_path(p) for p in paths]
        with self._lock:
            for path in normalized:
                entry = self._entries.pop(path, None)
                if entry is None or entry.tensor is None:
                    continue
                self._current_bytes -= entry.size_bytes
                self._lru.pop(path, None)
                entry.tensor = None
                entry.size_bytes = 0
                entry.ready_event = None
                entry.loading = False

    def close(self) -> None:
        """Shut down the background loader and release cached tensors."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=1.0)
        with self._lock:
            self._entries.clear()
            self._lru.clear()
            self._current_bytes = 0
        if self._stream is not None:
            self._stream = None

    def __enter__(self) -> "GPUFeatureStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    # ------------------------------------------------------------------
    # internal helpers
    def _canonical_path(self, path: str | Path) -> str:
        return str(Path(path).expanduser().resolve())

    def _worker(self) -> None:
        while True:
            request = self._queue.get()
            if request is _STOP:
                self._queue.task_done()
                break
            path = cast(str, request)
            with self._lock:
                entry = self._entries.get(path)
            if entry is None:
                self._queue.task_done()
                continue
            try:
                payload = self._load(Path(path))
                size_bytes = self._estimate_tensor_bytes(payload)
                if self.max_cache_bytes is not None and size_bytes > self.max_cache_bytes:
                    raise MemoryError(
                        f"Feature '{path}' ({size_bytes} bytes) exceeds max_cache_bytes "
                        f"of {self.max_cache_bytes}"
                    )
                stored = False
                with self._lock:
                    current_entry = self._entries.get(path)
                    if current_entry is entry:
                        self._evict_to_fit(required_bytes=size_bytes, skip=path)
                        entry.tensor = payload
                        entry.size_bytes = size_bytes
                        entry.loading = False
                        if self.asynchronous and self._stream is not None:
                            ready_event = torch.cuda.Event()
                            ready_event.record(self._stream)
                            entry.ready_event = ready_event
                        else:
                            entry.ready_event = None
                        self._lru[path] = None
                        self._lru.move_to_end(path)
                        self._current_bytes += size_bytes
                        stored = True
                if not stored:
                    entry.loading = False
                    entry.tensor = None
                    entry.size_bytes = 0
                    entry.ready_event = None
                    entry.error = RuntimeError(
                        f"Feature '{path}' was released before loading completed"
                    )
                entry.event.set()
            except Exception as exc:  # noqa: BLE001
                entry.error = exc
                entry.loading = False
                entry.event.set()
            finally:
                self._queue.task_done()

    def _load(self, path: Path) -> TensorLike:
        data = self._load_fn(path)
        return self._move_to_device(data)

    def _move_to_device(self, data: TensorLike) -> TensorLike:
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=self.asynchronous)
        if isinstance(data, Mapping):
            return type(data)({k: self._move_to_device(v) for k, v in data.items()})
        if isinstance(data, tuple):
            return type(data)(self._move_to_device(v) for v in data)
        if isinstance(data, list):
            return [self._move_to_device(v) for v in data]
        return data

    def _estimate_tensor_bytes(self, data: TensorLike) -> int:
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement()
        if isinstance(data, Mapping):
            return sum(self._estimate_tensor_bytes(v) for v in data.values())
        if isinstance(data, tuple):
            return sum(self._estimate_tensor_bytes(v) for v in data)
        if isinstance(data, list):
            return sum(self._estimate_tensor_bytes(v) for v in data)
        return 0

    def _evict_to_fit(self, *, required_bytes: int, skip: str) -> None:
        while self._needs_eviction(required_bytes, skip):
            if not self._lru:
                break
            evict_path, _ = self._lru.popitem(last=False)
            if evict_path == skip:
                # Put it back and stop evicting to avoid spinning.
                self._lru[evict_path] = None
                self._lru.move_to_end(evict_path)
                break
            entry = self._entries.pop(evict_path, None)
            if entry is None or entry.tensor is None:
                continue
            self._current_bytes -= entry.size_bytes
            entry.tensor = None
            entry.size_bytes = 0
            entry.ready_event = None
            entry.loading = False

    def _needs_eviction(self, required_bytes: int, skip: str) -> bool:
        over_bytes = (
            self.max_cache_bytes is not None
            and self._current_bytes + required_bytes > self.max_cache_bytes
        )
        over_items = (
            self.max_cache_items is not None
            and (len(self._lru) + (0 if skip in self._lru else 1)) > self.max_cache_items
        )
        return over_bytes or over_items

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()
