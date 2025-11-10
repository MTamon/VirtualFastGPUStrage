# VirtualFastGPUStrage

保存済みの PyTorch 特徴量ファイル（`.pt`）を GPU メモリ上にミラーリングし、ファイルパスから直接 GPU テンソルを取得できるストレージラッパーです。既存の `DataLoader` / `LightningDataModule` が収集したファイルパスをそのままバッチとして扱い、学習コード側では `torch.load()` の代わりに GPU 転送済みのテンソルを即座に参照できます。さらに、ラッパー型の `GPUCachedPathDataLoader` を使うと通常の DataLoader API で GPU 常駐済みのバッチを受け取れます。

## 特徴
- `.pt` ファイルのパスを渡すだけで、対応するテンソル／辞書構造を GPU 上に展開
- GPU メモリの容量上限（バイト数またはアイテム数）を指定してキャッシュを制御
- バックグラウンドスレッドがストレージから非同期で読み込み、学習ステップと重ね合わせ
- 既存の DataLoader/LightningDataModule とは疎結合で、CPU 側の前処理やマルチプロセスもそのまま活用可能

## 主要モジュール
- `fastloader/gpu_feature_store.py` — GPU 上に特徴量をキャッシュする `GPUFeatureStore`
- `fastloader/gpu_dataloader.py` — 既存の DataLoader を GPU キャッシュ付きでラップする `GPUCachedPathDataLoader`
- `fastloader/dataset.py` — （任意）`.pt` ファイルを遅延ロードする補助的なデータセット
- `examples/train_with_gpu_cache.py` — GPU キャッシュ対応 DataLoader を使った学習ループ例

## 使い方
1. 事前抽出済み特徴量を `torch.save` などで `.pt` ファイルとして保存します。辞書やタプル内のテンソルもサポートされます。
2. PyTorch の `DataLoader` に `.pt` ファイルのパスを返すデータセット（または `LightningDataModule`）を組み込みます。
3. 学習ループでは `GPUFeatureStore` を直接利用するか、`GPUCachedPathDataLoader` で DataLoader をラップして GPU 上のデータを受け取ります。

```python
from fastloader import GPUCachedPathDataLoader

gpu_loader = GPUCachedPathDataLoader(
    path_loader,
    device="cuda:0",
    max_cache_bytes=8 * 1024 ** 3,  # 8GB までキャッシュ
    max_cache_items=256,
    collate_fn=collate_on_gpu,
)

for features, labels in gpu_loader:
    ...  # 特徴量はすでに GPU 上
```

### ストリーミングの一例
`GPUCachedPathDataLoader` は内部で次バッチを先読みします。より細かく制御したい場合は、`GPUFeatureStore` を直接利用して `prefetch` と `get_batch` を呼び出すこともできます。

```python
def stream_gpu_batches(path_loader, store):
    iterator = iter(path_loader)
    try:
        next_paths = next(iterator)
    except StopIteration:
        return
    store.prefetch(next_paths)
    for paths in iterator:
        yield store.get_batch(next_paths)
        store.prefetch(paths)
        next_paths = paths
    yield store.get_batch(next_paths)
```

### GPUCachedPathDataLoader の主な引数
- `prefetch_batches`: 何バッチ先まで GPU へ先読みするかを指定します。
- `collate_fn`: GPU 上に展開されたアイテムのリストから最終的なバッチを組み立てます。
- `path_extractor`: DataLoader が返すバッチ構造から特徴量ファイルのパス列を取り出すカスタム関数を指定できます。
- `build_batch_fn`: GPU データと元のバッチ（ラベルなどのメタデータ）を任意の形で組み合わせたい場合に使用します。

### キャッシュの調整ポイント
- `max_cache_bytes`: GPU メモリの総使用量上限（バイト）。未指定の場合はバイト数では制限しません。
- `max_cache_items`: キャッシュに保持するアイテム数の上限。ディスク上のファイルサイズが均一な場合に便利です。
- `release(paths)`: 使い終えたパスを手動でキャッシュから削除したいときに利用します。

## 注意事項
- `asynchronous=True` かつ CUDA デバイスを指定した場合は専用 CUDA ストリームで GPU 転送を行います。PyTorch のカレントストリームとは `wait_event` で同期されます。
- `.pt` 内部がテンソル以外の Python オブジェクトのみの場合、GPU 転送すべきデータが存在しないためキャッシュサイズは 0 として扱われます。
- 1 つのアイテムが `max_cache_bytes` を超える場合は `MemoryError` が送出されます。

## サンプルスクリプト
```
python examples/train_with_gpu_cache.py /path/to/features --synthetic --device cuda --epochs 3
```
`--synthetic` を指定するとダミー特徴量を生成して挙動を確認できます。実運用では不要です。

## ライセンス
このプロジェクトは [MIT License](LICENSE) の下で公開されています。
