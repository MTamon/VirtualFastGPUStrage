"""Dataset utilities for streaming pre-extracted feature tensors from storage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import torch

TensorOrDict = Union[torch.Tensor, Sequence[torch.Tensor], Tuple[torch.Tensor, ...], dict]


@dataclass
class FeatureRecord:
    """A simple container representing a single feature file on disk."""

    path: Path

    def load(self) -> TensorOrDict:
        """Load the serialized tensor payload from disk.

        The default implementation expects ``.pt`` files saved with ``torch.save``.
        Override or provide a custom loader for other formats (e.g. ``.npy``).
        """

        return torch.load(self.path, map_location="cpu")


class FeatureFileDataset(torch.utils.data.Dataset):
    """A dataset that lazily loads tensors from a directory of feature files."""

    def __init__(
        self,
        files: Union[str, Path, Sequence[Union[str, Path]], Iterable[Union[str, Path]]],
        loader: Callable[[Path], TensorOrDict] | None = None,
    ) -> None:
        if isinstance(files, (str, Path)):
            files = sorted(Path(files).expanduser().glob("*.pt"))
        else:
            files = [Path(f).expanduser() for f in files]
        self.records: List[FeatureRecord] = [FeatureRecord(path=f) for f in files]
        self.loader = loader or (lambda p: torch.load(p, map_location="cpu"))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.records)

    def __getitem__(self, index: int) -> TensorOrDict:  # type: ignore[override]
        record = self.records[index]
        return self.loader(record.path)
