# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

音声波形のみから話者特徴量(voice print)と発話内容特徴量(content)を分離抽出することを目指した実験的リポジトリ。JVSコーパス(日本語話者100名)で学習し、話者変換(あるコンテンツに別話者のvoice printを合成)を試みていた。**開発は途中で中断されており、コードは未完成・一部壊れた状態**(下記「既知の問題」参照)。

## 環境・コマンド

- Python 3.10 固定(`requires-python = ">=3.10,<3.11"`)、パッケージ管理は uv のみ(Dockerは廃止済み)。CUDA/GPU 前提のコード。
- テスト・リンタは未設定。`test.py` は pytest ではなく、データセットの動作確認用スクリプト。

```bash
# 依存関係のインストール
uv sync

# 初回セットアップ: .env を作成しリソースの場所を指定
cp .env.example .env

# 学習 — いずれもDDP前提。単一GPUでも torchrun が必要
# (無条件に init_process_group を呼ぶため)。GPUは6×RTX 3090
uv run torchrun --nproc_per_node=6 src/voice_print_matrix/train.py     # VPM(話者分離)モデル
uv run torchrun --nproc_per_node=6 src/voice_print_matrix/train_ae.py  # 単純オートエンコーダ

# 評価(話者変換 / 再構成)
uv run python src/voice_print_matrix/eval.py      # metan の content + zundamon の voice print で変換
uv run python src/voice_print_matrix/eval_ae.py   # zundamon.wav の再構成
uv run python src/voice_print_matrix/specgram.py  # スペクトログラムのPNG出力
```

## 必要なデータ(gitに含まれない)

リソースの置き場所は `.env` の `RESOURCES_DIR` 環境変数で指定する(`src/voice_print_matrix/config.py` が `python-dotenv` で読み込み、未設定時はリポジトリ直下の `resources/` にフォールバック)。現在の実体は `/mnt/raid0/VoicePrintMatrix/resources/`。実行には以下が必要:

- `$RESOURCES_DIR/jvs_ver1/jvs001/ ... jvs100/` — JVSコーパス(22050Hzで読み込み)
- `$RESOURCES_DIR/zundamon.wav`, `$RESOURCES_DIR/metan.wav` — 評価用音声
- `$RESOURCES_DIR/weight/` — 学習済み重みの保存先(`vpm_ae.pt`, `ae.pt`)

## アーキテクチャ

音声は22050Hzで読み込み、長さ2048サンプルのセグメント列として扱う。全モデルの入出力は `(batch, length, segment_length=2048)`。

- **`qgru.py`** — 全モデルの基盤となる系列モデル。並列prefix-scan(`scan()`)で計算するGRU風の再帰層(QGRU)+ SwiGLU FFN のブロックを積んだ `QGRUModel`。セグメント系列方向の時間依存を担う。
- **`vpm_ae.py`** — モデル定義がすべてここに集約されている:
  - `Encoder`: MelSpectrogram → QGRU で各セグメントを潜在ベクトルへ
  - `Decoder`: DDSP風の合成デコーダ(正弦波オシレータバンク + 学習フィルタをFFT畳み込みで適用)。コメントアウトが多く試行錯誤の跡が残っている
  - `HiFiGANDecoder`: HiFi-GAN風のアップサンプリングデコーダ(MRFブロック)。`AutoEncoder` はこちらを使用
  - `VPMAutoEncoder`: content_encoder / print_encoder の2系統エンコーダ + Decoder。本命のモデル
  - `AutoEncoder`: Encoder + デコーダの単純AE(train_ae.py で使用)。`decoder_type` 引数で `'ddsp'`(Decoder、デフォルト)と `'hifigan'`(HiFiGANDecoder)を切り替え。**現在の方針は HiFiGAN 路線**: DDSP は f0 を sin→cumsum 経由の勾配で学習する設計のため multi-resolution STFT 損失下では入力を無視した平均スペクトル解に崩壊することが確認済み(改善するには本家DDSP同様に外部ピッチトラッカーによる f0 条件付けが必要)
- **`train.py`** — VPMAutoEncoder のDDP学習。損失は4項:
  1. `loss_ae`: 波形再構成MSE
  2. `loss_vp`: バッチ内の voice print 同士のcosine類似度行列("voice print matrix")を、同一話者なら+1・異話者なら−1に近づける対照的損失
  3. `loss_udc` / `loss_udp`: voice print をバッチ内でシャッフルしてデコードした波形を再エンコードし、content / voice print が保存されるよう課すサイクル一貫性損失(`upside_down`)
- **`jvs_batch_dataset.py`** — JVS全話者のwavを連結し `(segments_per_batch=256, 2048)` のブロックに切り出す `TensorDataset` を返す。話者ラベルはセグメント単位。`size_ratio` で使用話者数を絞れる(デバッグ用に `size_ratio=0.01` など)。毎回全wavをメモリにロードする(キャッシュなし)。
- **`utils.py`** — `MultiResolutionSTFTLoss`: Hann窓・75%オーバーラップ・複数FFTサイズの multi-resolution STFT 損失(spectral convergence + log振幅L1)。セグメント連結後の波形全体に適用する(窓が境界をまたぐため不連続も罰せられる)。`multiscale_spectrum` は旧損失用の変換(矩形窓・オーバーラップなし、長さは2の冪必須)。train_ae.py 冒頭の `loss_type`(`'mrstft'` / `'multiscale'`)でどちらを使うか選択できる。

## 既知の問題(中断時点の状態)

- `train.py` と `eval.py` が `from voice_print_matrix.ae import AutoEncoder` を import しているが、`ae.py` は存在しない(モデルは `vpm_ae.py` にある)。この import を修正しないと実行不可。
- `VPMAutoEncoder.__init__` が `Encoder(..., dim_out=...)` を呼ぶが、`Encoder.__init__` に `dim_out` 引数はない(TypeError になる)。QGRUModel の `dim_out` に渡す意図だったと思われる。
- `train.py` の `voice_print_matrix` 変数がパッケージ名と同名でシャドーイングしている点に注意。
