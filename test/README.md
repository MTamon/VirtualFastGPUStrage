# Test Suite Overview

This directory contains automated regression tests for the GPU-backed feature
loading pipeline.

## Files
- `test_gpu_feature_store.py` – unit coverage for the `GPUFeatureStore`
  caching, eviction, and batching helpers.
- `test_gpu_cached_dataloader.py` – integration-style checks for the
  `GPUCachedPathDataLoader` wrapper around iterable loaders.

## Running the tests

Execute all tests from the project root with:

```bash
python -m pytest test
```

PyTorch and `pytest` are the only runtime requirements. A CUDA-capable GPU is
**not** required; the tests pin the cache to CPU devices so they can execute in
CPU-only environments.
