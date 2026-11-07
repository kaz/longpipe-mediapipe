# Longpipe model for @mediapipe/tasks-vision

[日本語](README.ja.md)

[Longpipe](https://github.com/sb2702/longpipe)'s person segmentation (matting) models, converted to TFLite so they run in
[`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision)'s `ImageSegmenter` as a drop-in replacement
for MediaPipe's `selfie_segmenter` — no Longpipe runtime required.

- All five Longpipe sizes (xs / small / medium / large / xl), each in a **landscape** and a **portrait** variant
- Same input/output contract and metadata as `selfie_segmenter`: swap `modelAssetPath`, keep your code
- `index.html`: a webcam demo that runs any number of Longpipe and MediaPipe models side by side on the same frame

## Demo

```bash
python3 -m http.server 8000
# open http://localhost:8000/
```

Pick **Webcam** or **Demo image**, add panels with **＋ add model**, and switch model / delegate (GPU, CPU) per panel.
The default panels depend on the screen shape when the page opens (landscape: `selfie_segmenter_landscape` + `longpipe_landscape_small`,
portrait: `selfie_segmenter` + `longpipe_portrait_small`). `?panels=lp_landscape_xl,mp_selfie&source=demo` overrides them.

Camera access needs a secure context: `localhost` works on the same machine; phones need HTTPS (e.g. GitHub Pages or a tunnel).
`@mediapipe/tasks-vision`, the MediaPipe models and the demo / background photos are loaded from jsDelivr, Google's model
storage and Unsplash at runtime (the background falls back to a solid colour if its photo cannot be loaded).

## Models

| File | Input (W×H) | Size |
|---|---|---|
| `models/longpipe_landscape_xs_float16.tflite` | 384×240 | 1.8 MB |
| `models/longpipe_landscape_small_float16.tflite` | 384×224 | 1.8 MB |
| `models/longpipe_landscape_medium_float16.tflite` | 512×320 | 6.5 MB |
| `models/longpipe_landscape_large_float16.tflite` | 640×400 | 6.5 MB |
| `models/longpipe_landscape_xl_float16.tflite` | 1280×768 | 14.6 MB |
| `models/longpipe_portrait_{xs,small,medium,large,xl}_float16.tflite` | transposed (e.g. 224×384) | same |

- Input: `[1, H, W, 3]` float32 RGB, normalization metadata mean 0 / std 255 (i.e. pixels in [0, 1])
- Output: `[1, H, W, 1]` float32 person probability (sigmoid applied in the graph)
- Metadata identical to `selfie_segmenter`: label `selfie`, `SEGMENTER_METADATA` activation `NONE`, min parser version 1.5.0
- Weights are stored as float16 (the published Longpipe weights are float16, so this is lossless); compute is float32
- Ops: `CONV_2D`, `DEPTHWISE_CONV_2D`, `PAD`, `RESIZE_BILINEAR`, `CONCATENATION`, `ADD`, `LOGISTIC`, `DEQUANTIZE` — all GPU-delegate compatible

### Landscape or portrait?

`ImageSegmenter` stretches every frame to the model's input size, ignoring aspect ratio. MediaPipe's own models tolerate that;
the Longpipe models do not — a portrait phone frame squeezed into a landscape model loses large parts of the person.
**Use the portrait models for portrait frames** (same weights, transposed input canvas):

```js
const portrait = video.videoHeight > video.videoWidth;
const modelAssetPath = `models/longpipe_${portrait ? 'portrait' : 'landscape'}_small_float16.tflite`;
// re-create the ImageSegmenter when the orientation changes
```

If you must use a single model, letterboxing the frame (black bars to the model's aspect, then cropping the mask) also works
but costs some accuracy — see the numbers below.

## Usage

```js
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm');
const segmenter = await ImageSegmenter.createFromOptions(vision, {
  baseOptions: { modelAssetPath: 'models/longpipe_landscape_small_float16.tflite', delegate: 'GPU' },
  runningMode: 'VIDEO',
  outputConfidenceMasks: true,
});

segmenter.segmentForVideo(video, performance.now(), (result) => {
  const alpha = result.confidenceMasks[0].getAsFloat32Array();   // person probability, video-sized
});
```

Results behave exactly like `selfie_segmenter`: one confidence mask (person probability, resized to the input image size),
and a category mask where `0` = person and `255` = background.

## Accuracy

Mask IoU against ground-truth alpha on 10 synthetic composites (2 transparent-background portraits × 5 backgrounds),
fed MediaPipe-style (stretched to the model input). A small sanity check, not a benchmark.

| Size | Landscape model, 16:9 frames | Portrait model, 9:16 frames | Landscape model, 9:16 frames (stretched) |
|---|---|---|---|
| xs | 0.972 | 0.969 | 0.806 |
| small | 0.982 | 0.975 | 0.856 |
| medium | 0.961 | 0.970 | 0.737 |
| large | 0.960 | 0.970 | 0.589 |
| xl | 0.969 | 0.982 | 0.504 |

medium / large also segment a second person that appears in one composite's background image (the ground truth contains only
the foreground person); without that image they score 0.975 / 0.974. Letterboxing 9:16 frames into the landscape models
scored 0.94–0.97.

The small model was also checked against an independent NumPy reference implementation (max difference ~1e-4). Inside
`@mediapipe/tasks-vision` its output matches that reference within 1e-4 on the CPU delegate (mask IoU 1.000) and at
IoU 0.9997 on the GPU delegate, the difference coming from reduced GPU precision.

## Performance (rough)

`@mediapipe/tasks-vision` 1.1.1, GPU delegate, Apple M2, Chromium 152, 640×360 frames, nine models running back to back,
per call including mask readback. Single machine, after warm-up; expect large variation across devices and runs.

| Model | ms |
|---|---|
| selfie_segmenter / selfie_segmenter_landscape | ~14 / ~16 |
| longpipe xs / small / medium / large / xl | ~10 / ~11 / ~14 / ~15 / ~29 |

## Rebuilding the models

```bash
python3.12 -m venv .venv-tf && .venv-tf/bin/pip install -r tools/requirements-convert.txt
python3.12 -m venv .venv-mp && .venv-mp/bin/pip install -r tools/requirements-metadata.txt
TF_PY=.venv-tf/bin/python MP_PY=.venv-mp/bin/python tools/build_all.sh          # all sizes, or e.g. `... build_all.sh small`
```

`build_all.sh` downloads the Longpipe v0.0.5 weight packs (`https://cdn.longpipe.dev/models/v/0.0.5/model_<size>.f16.bin`),
checks their SHA-256, converts each size in both orientations (`tools/convert_to_tflite.py`) and writes the MediaPipe metadata
(`tools/add_metadata.py`). Rebuilt models are identical to the committed ones except for the timestamp inside the packed label file.

## How the conversion works

- **Weight pack**: `u32` header length, a JSON header mapping each tensor to `{offset, length}`, then packed float16 values.
- **Layouts**: convolution weights are stored for WebGPU as `[tap][out group][in group]` 4×4 blocks (column = input channel);
  depthwise weights as `[tap][channel]`. Narrow wrapper tensors are padded to 4 channels.
- **Network**: EfficientNet-Lite encoder (Lite0 for xs–large, Lite3 for xl) with a U-Net decoder, plus a high-resolution wrapper
  that downsamples the frame into the base network and fuses the base features back at full resolution
  (one-stage for xs / small / medium, two-stage for large / xl). Stride-2 layers use TF-style `SAME` padding.
- The graph is written with plain `tf.nn` ops and converted with the TFLite converter; portrait variants rebuild the same
  graph at the transposed canvas (the network is fully convolutional).

Only the matting path is converted. Longpipe's optical-flow temporal stabilisation and face heads, which share the `.bin`
files, are not included, so masks are computed per frame.

## License

See [LICENSE](LICENSE).

## Credits

- The models are converted from the [Longpipe](https://github.com/sb2702/longpipe) weights, and the network wiring follows the Longpipe SDK source.
- MediaPipe models and `@mediapipe/tasks-vision` are not included; the demo loads them at runtime.
- Demo photos ([Unsplash License](https://unsplash.com/license), not included — loaded from Unsplash at runtime):
  people by [Vitaly Gariev](https://unsplash.com/@silverkblack) ([landscape demo](https://unsplash.com/photos/MLoyNixjAPo)) and
  [Vincenzo De Simone](https://unsplash.com/@foffoxx) ([portrait demo](https://unsplash.com/photos/BjnXKceifJA)),
  background by [Toa Heftiba](https://unsplash.com/@heftiba) ([photo](https://unsplash.com/photos/cSpfzCB2jFk)).

This project is not affiliated with or endorsed by Longpipe, Google, or the MediaPipe team.
