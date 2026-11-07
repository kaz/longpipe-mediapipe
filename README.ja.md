# Longpipe model for @mediapipe/tasks-vision

[English](README.md)

[Longpipe](https://github.com/sb2702/longpipe) の人物セグメンテーション（マッティング）モデルを TFLite に変換したものです。
[`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision) の `ImageSegmenter` で、MediaPipe 公式の
`selfie_segmenter` の代わりにそのまま使えます。Longpipe の実行環境は不要です。

- Longpipe の全 5 サイズ（xs / small / medium / large / xl）を、それぞれ**横長（landscape）**と**縦長（portrait）**の 2 種類で用意
- 入出力とメタデータは `selfie_segmenter` と同じなので、`modelAssetPath` を差し替えるだけで動きます

## デモ

https://kaz.github.io/longpipe-mediapipe/

## モデル

| ファイル | 入力（幅×高さ） | サイズ |
|---|---|---|
| `models/longpipe_landscape_xs_float16.tflite` | 384×240 | 1.8 MB |
| `models/longpipe_landscape_small_float16.tflite` | 384×224 | 1.8 MB |
| `models/longpipe_landscape_medium_float16.tflite` | 512×320 | 6.5 MB |
| `models/longpipe_landscape_large_float16.tflite` | 640×400 | 6.5 MB |
| `models/longpipe_landscape_xl_float16.tflite` | 1280×768 | 14.6 MB |
| `models/longpipe_portrait_{xs,small,medium,large,xl}_float16.tflite` | 縦横を入れ替えたもの（例：224×384） | 同じ |

- 入力：`[1, H, W, 3]` の float32 RGB。正規化のメタデータは mean 0 / std 255（画素値 0〜1）
- 出力：`[1, H, W, 1]` の float32 で、人物である確率（sigmoid はモデル内で適用済み）
- メタデータは `selfie_segmenter` と同じ：ラベル `selfie`、`SEGMENTER_METADATA` の活性化は `NONE`、min parser version 1.5.0
- 重みは float16 で格納しています（配布されている Longpipe の重みがもともと float16 なので劣化はありません）。計算は float32 です
- 使っている演算：`CONV_2D`、`DEPTHWISE_CONV_2D`、`PAD`、`RESIZE_BILINEAR`、`CONCATENATION`、`ADD`、`LOGISTIC`、`DEQUANTIZE`（すべて GPU デリゲート対応）

### landscape と portrait の使い分け

`ImageSegmenter` は、縦横比を無視してフレームをモデルの入力サイズに引き伸ばします。MediaPipe 公式モデルはこれに耐えますが、
Longpipe のモデルは耐えられません。スマートフォンの縦長フレームを横長モデルに入れると、人物の大部分が欠けます。
**縦長フレームには portrait モデルを使ってください**（重みは同じで、入力の縦横を入れ替えたものです）。

```js
const portrait = video.videoHeight > video.videoWidth;
const modelAssetPath = `models/longpipe_${portrait ? 'portrait' : 'landscape'}_small_float16.tflite`;
// 向きが変わったら ImageSegmenter を作り直す
```

モデルを 1 つで済ませたい場合は、フレームに黒い余白を付けてモデルの縦横比に合わせ、返ってきたマスクを切り出す方法（レターボックス）でも動きます。
ただし精度は少し下がります（下の数値を参照）。

## 使い方

```js
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm');
const segmenter = await ImageSegmenter.createFromOptions(vision, {
  baseOptions: { modelAssetPath: 'models/longpipe_landscape_small_float16.tflite', delegate: 'GPU' },
  runningMode: 'VIDEO',
  outputConfidenceMasks: true,
});

segmenter.segmentForVideo(video, performance.now(), (result) => {
  const alpha = result.confidenceMasks[0].getAsFloat32Array();   // 人物の確率（映像と同じサイズ）
});
```

結果の扱いは `selfie_segmenter` とまったく同じです。confidence mask は 1 枚（人物の確率、入力画像のサイズで返る）で、
category mask は `0` が人物、`255` が背景です。

## 精度

正解アルファ付きの合成画像 10 枚（背景透過の人物写真 2 枚 × 背景 5 種類）で、マスクの IoU を測りました。
入力は MediaPipe と同じく、モデルの入力サイズに引き伸ばしています。動作確認の目安で、ベンチマークではありません。

| サイズ | landscape モデル・16:9 | portrait モデル・9:16 | landscape モデル・9:16（引き伸ばし） |
|---|---|---|---|
| xs | 0.972 | 0.969 | 0.806 |
| small | 0.982 | 0.975 | 0.856 |
| medium | 0.961 | 0.970 | 0.737 |
| large | 0.960 | 0.970 | 0.589 |
| xl | 0.969 | 0.982 | 0.504 |

medium と large は、背景画像に別の人物が写っている 1 枚で、その人物も切り抜いてしまいます（正解には手前の人物しか含まれていません）。
その画像を除くと、0.975 / 0.974 です。9:16 のフレームをレターボックスで landscape モデルに入れた場合は 0.94〜0.97 でした。

small モデルは、独立に書いた NumPy の参照実装とも照合しました（最大差 約 1e-4）。`@mediapipe/tasks-vision` 上でも、
CPU デリゲートでは参照実装と 1e-4 以内で一致（マスク IoU 1.000）、GPU デリゲートでは IoU 0.9997 でした。GPU 側の差は計算精度が低いことによるものです。

## 速度（目安）

`@mediapipe/tasks-vision` 1.1.1、GPU デリゲート、Apple M2、Chromium 152、640×360 のフレームで 9 モデルを順番に実行したときの、
マスクの読み出しを含む 1 回あたりの時間です。1 台・ウォームアップ後の値で、端末や実行ごとに大きく変わります。

| モデル | ms |
|---|---|
| selfie_segmenter / selfie_segmenter_landscape | 約 14 / 約 16 |
| longpipe xs / small / medium / large / xl | 約 10 / 11 / 14 / 15 / 29 |

## モデルの再生成

```bash
python3.12 -m venv .venv-tf && .venv-tf/bin/pip install -r tools/requirements-convert.txt
python3.12 -m venv .venv-mp && .venv-mp/bin/pip install -r tools/requirements-metadata.txt
TF_PY=.venv-tf/bin/python MP_PY=.venv-mp/bin/python tools/build_all.sh          # 全サイズ（`... build_all.sh small` のように個別指定も可）
```

`build_all.sh` は、Longpipe v0.0.5 の重み（`https://cdn.longpipe.dev/models/v/0.0.5/model_<size>.f16.bin`）をダウンロードして
SHA-256 を照合し、各サイズを 2 つの向きで変換（`tools/convert_to_tflite.py`）してから、MediaPipe 用のメタデータを書き込みます（`tools/add_metadata.py`）。
再生成したモデルは、同梱しているラベルファイル内のタイムスタンプを除いて、リポジトリのものと同一になります。

## 変換の仕組み

- **重みファイル**：先頭 4 バイトがヘッダー長、続いて各テンソルの `{offset, length}` を記した JSON ヘッダー、その後に float16 の値が並びます。
- **重みの並び**：畳み込みの重みは WebGPU 向けに `[タップ][出力 4ch 群][入力 4ch 群]` の 4×4 ブロック（列が入力チャネル）で、
  depthwise は `[タップ][チャネル]` で格納されています。ラッパー部の細いテンソルは 4ch にパディングされています。
- **ネットワーク**：EfficientNet-Lite のエンコーダ（xs〜large は Lite0、xl は Lite3）と U-Net デコーダに、高解像度化のラッパーが付いた構成です。
  ラッパーはフレームを縮小してベースに入力し、ベースの特徴量を元の解像度で合成し直します（xs / small / medium は 1 段、large / xl は 2 段）。
  stride 2 の層は TF 式の `SAME` パディングです。
- グラフは素の `tf.nn` の演算で書き、TFLite コンバーターで変換しています。portrait 版は、同じグラフを縦横を入れ替えた入力サイズで組み直したものです（完全畳み込みのネットワークなので重みはそのまま使えます）。

変換したのは背景切り抜きの部分だけです。同じ `.bin` に入っている、Longpipe のオプティカルフローによる時間方向の安定化や顔関連のヘッドは含めていないので、
マスクはフレームごとに独立して計算されます。

## ライセンス

[LICENSE](LICENSE) を参照してください。`models/` 内のモデルには [models/LICENSE](models/LICENSE) が適用されます。

## クレジット

- モデルは [Longpipe](https://github.com/sb2702/longpipe) の重みから変換したもので、ネットワークの配線は Longpipe SDK のソースコードに従っています。
- MediaPipe 公式モデルと `@mediapipe/tasks-vision` は同梱していません。デモの実行時に読み込みます。
- デモ用の写真（[Unsplash License](https://unsplash.com/license)。リポジトリには含めず、実行時に Unsplash から読み込みます）：
  人物は [Vitaly Gariev](https://unsplash.com/@silverkblack) 氏（[横長デモ用](https://unsplash.com/photos/MLoyNixjAPo)）と
  [Vincenzo De Simone](https://unsplash.com/@foffoxx) 氏（[縦長デモ用](https://unsplash.com/photos/BjnXKceifJA)）、
  背景は [Toa Heftiba](https://unsplash.com/@heftiba) 氏（[写真](https://unsplash.com/photos/cSpfzCB2jFk)）の作品です。

このプロジェクトは Longpipe、Google、MediaPipe チームとは関係がなく、承認を受けたものでもありません。
