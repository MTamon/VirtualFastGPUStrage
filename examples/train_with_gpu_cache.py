"""Example training loop using the GPU-cached DataLoader wrapper."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from fastloader import GPUCachedPathDataLoader


class DummyModel(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class FeaturePathDataset(Dataset[str]):
    """Dataset that returns paths to serialized feature files."""

    def __init__(self, data_dir: Path) -> None:
        self.paths = sorted(Path(data_dir).expanduser().glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found under {data_dir}")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, index: int) -> str:  # type: ignore[override]
        return str(self.paths[index])


def _create_synthetic_features(out_dir: Path, num_samples: int, feature_dim: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(num_samples):
        feature = torch.randn(feature_dim)
        label = torch.randint(0, 10, (1,), dtype=torch.long)
        torch.save({"feature": feature, "label": label}, out_dir / f"sample_{idx:06d}.pt")


def collate_on_gpu(records: Sequence[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack([item["feature"] for item in records])
    labels = torch.cat([item["label"].reshape(1) for item in records]).to(features.device)
    return features, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing .pt feature files")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-cache-bytes",
        type=int,
        default=None,
        help="Upper bound on GPU cache size in bytes",
    )
    parser.add_argument(
        "--max-cache-items",
        type=int,
        default=256,
        help="Maximum number of feature files to keep on the GPU",
    )
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic features for demo")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU DataLoader workers")
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="Number of path batches to keep prefetched on the GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.synthetic:
        _create_synthetic_features(args.data_dir, num_samples=10_000, feature_dim=args.feature_dim)

    dataset = FeaturePathDataset(args.data_dir)
    path_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list,
    )

    model = DummyModel(args.feature_dim, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    with GPUCachedPathDataLoader(
        device=device,
        max_cache_bytes=args.max_cache_bytes,
        max_cache_items=args.max_cache_items,
        loader=path_loader,
        collate_fn=collate_on_gpu,
        prefetch_batches=args.prefetch_batches,
    ) as gpu_loader:
        for epoch in range(args.epochs):
            for features, labels in gpu_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1} finished: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
