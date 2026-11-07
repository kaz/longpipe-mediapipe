"""Convert Longpipe tier weight packs (model_<tier>[.f16].bin) to TFLite, shaped like MediaPipe's
selfie_segmenter so each can be used as a drop-in ImageSegmenter model.

  input : [1, H, W, 3] float32  RGB in [0, 1]   (metadata: mean 0 / std 255)
  output: [1, H, W, 1] float32  person alpha in [0, 1] (sigmoid applied in-graph)

H x W is the tier's native canvas (see TIERS). The graph is written with raw tf.nn ops (no
Keras) so padding / resize semantics are explicit. No Longpipe code is used; the wiring
mirrors sdk/src/model/{tier_config,tier_model,blocks/*,networks/*}.ts.

usage: python tools/convert_to_tflite.py model_<tier>.f16.bin out.tflite --tier <xs|small|medium|large|xl>
       [--float32]   (default: float16 weights, float32 compute — lossless for .f16.bin packs)
"""
import argparse
import json
import struct

import numpy as np
import tensorflow as tf

# tier -> wrapper variant, canvas (H, W), base input (H, W)   [sdk/src/model/tier_config.ts]
TIERS = {
    "xs":     dict(variant="B", canvas=(240, 384),  base=(80, 128)),
    "small":  dict(variant="A", canvas=(224, 384),  base=(112, 192)),
    "medium": dict(variant="A", canvas=(320, 512),  base=(160, 256)),
    "large":  dict(variant="D", canvas=(400, 640),  base=(160, 256)),
    "xl":     dict(variant="E", canvas=(768, 1280), base=(192, 320)),
}
# EfficientNet-Lite stage -> (depthwise kernel, stride of its first block)
STAGE_SPEC = {"s1": (3, 2), "s2": (5, 2), "s3": (3, 2), "s4": (5, 1), "s5": (5, 2), "s6": (3, 1)}
# decoder skip taps, deepest first, by number of decoder blocks
DECODER_SKIPS = {2: ["s2", "s1"], 3: ["s4", "s2", "s1"]}


# ---------------------------------------------------------------- weight pack
def load_pack(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<I", raw[:4])[0]
    header = json.loads(raw[4:4 + n])
    dtype = "<f2" if header.get("__dtype__") == "f16" else "<f4"
    data = np.frombuffer(raw[4 + n:], dtype=dtype).astype(np.float32)

    def walk(o):
        if isinstance(o, dict):
            if set(o) == {"offset", "length"}:
                return data[o["offset"]:o["offset"] + o["length"]]
            return {k: walk(v) for k, v in o.items() if k != "__dtype__"}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o

    return walk(header)


def hwio(p, k):
    """Packed [tap][out_group][in_group] mat4x4 (col=in, row=out) -> TF kernel [k, k, in, out]."""
    w, cout = p["weights"], p["bias"].size
    cin = w.size // (k * k * cout)
    t = w.reshape(k * k, cout // 4, cin // 4, 4, 4).transpose(0, 2, 3, 1, 4)    # tap, ig, i4, og, o4
    return np.ascontiguousarray(t.reshape(k, k, cin, cout))


def dw_kernel(p, k):
    """Packed depthwise [tap][channel] -> TF depthwise kernel [k, k, C, 1]."""
    return np.ascontiguousarray(p["weights"].reshape(k, k, -1, 1))


# ---------------------------------------------------------------- ops
def conv(x, k, b, stride=1, padding="SAME", act=None):
    y = tf.nn.bias_add(tf.nn.conv2d(x, tf.constant(k), strides=stride, padding=padding), tf.constant(b))
    return tf.nn.relu6(y) if act == "relu6" else tf.nn.relu(y) if act == "relu" else y


def dwconv(x, k, b, stride=1):
    # 'SAME' == TF/timm same padding (extra pad bottom/right on stride 2), as the model was trained.
    y = tf.nn.depthwise_conv2d(x, tf.constant(k), strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.relu6(tf.nn.bias_add(y, tf.constant(b)))


def resize(x, h, w):
    return tf.image.resize(x, [h, w], method="bilinear")   # half_pixel_centers=True, no antialias


SYM1 = [[0, 0], [1, 1], [1, 1], [0, 0]]   # explicit symmetric pad 1 (wrapper convs)


# ---------------------------------------------------------------- base network
def base_net(x, b):
    """EfficientNet-Lite encoder + U-Net decoder. Returns featLowRes (finalUp output, base/2)."""
    e = b["encoder"]
    h = conv(x, hwio(e["stem"], 3)[:, :, :3, :], e["stem"]["bias"], 2, "SAME", "relu6")   # 4th lane = 0
    h = dwconv(h, dw_kernel(e["s0"]["dw"], 3), e["s0"]["dw"]["bias"])
    h = conv(h, hwio(e["s0"]["pw"], 1), e["s0"]["pw"]["bias"])
    taps = {}
    for stage in [s for s in STAGE_SPEC if s in e]:
        k, first_stride = STAGE_SPEC[stage]
        for i, blk in enumerate(e[stage]):
            s = first_stride if i == 0 else 1
            t = conv(h, hwio(blk["expand"], 1), blk["expand"]["bias"], act="relu6") if "expand" in blk else h
            t = dwconv(t, dw_kernel(blk["dw"], k), blk["dw"]["bias"], s)
            t = conv(t, hwio(blk["proj"], 1), blk["proj"]["bias"])
            h = t + h if (s == 1 and t.shape[-1] == h.shape[-1]) else t
        taps[stage] = h
    d = b["decoder"]
    h = conv(h, hwio(b["bottleneck"], 1), b["bottleneck"]["bias"], act="relu6")
    for blk, skip in zip(d["blocks"], DECODER_SKIPS[len(d["blocks"])]):
        sk = taps[skip]
        h = tf.concat([resize(h, sk.shape[1], sk.shape[2]), sk], axis=-1)        # [deep, skip]
        h = conv(h, hwio(blk["conv1"], 3), blk["conv1"]["bias"], act="relu6")
        h = conv(h, hwio(blk["conv2"], 3), blk["conv2"]["bias"], act="relu6")
    h = resize(h, h.shape[1] * 2, h.shape[2] * 2)
    return conv(h, hwio(d["finalUpsample"], 1), d["finalUpsample"]["bias"], act="relu6")


# ---------------------------------------------------------------- full tier (wrapper + base)
def build(W, tier, portrait=False):
    cfg = TIERS[tier]
    (H, Wd), (bh, bw), v = cfg["canvas"], cfg["base"], cfg["variant"]
    if portrait:   # fully convolutional: same weights at the transposed canvas (e.g. 224x384 -> 384x224)
        (H, Wd), (bh, bw) = (Wd, H), (bw, bh)
    w = W["wrapper"]
    # The pack stores the wrapper's narrow tensors padded to 4 lanes; the 4th in/out lane of
    # down1/down2/adapter is zero, so only 3 channels are carried here.
    down1 = (hwio(w["down1"], 3)[:, :, :3, :3], w["down1"]["bias"][:3])
    adapter = (hwio(w["adapter"], 1)[:, :, :3, :3], w["adapter"]["bias"][:3])
    expand = (np.ascontiguousarray(w["expandFeat"]["weights"].reshape(3, 3, -1, 2)), w["expandFeat"]["bias"])
    if v in "DE":
        down2 = (hwio(w["down2"], 3)[:, :, :3, :3], w["down2"]["bias"][:3])
        # up1Combine: 9 x 2 mat3x2 (col-major) over concat(u.xy, d.xyzw) -> [3,3,6,2]; drop zero d.w
        up1 = (np.ascontiguousarray(w["up1Combine"]["weights"].reshape(3, 3, 6, 2)[:, :, :5, :]), w["up1Combine"]["bias"])
    uc = w["upCombine"]["weights"]
    if v == "D":   # 27 vec4 = [tap][u (w0,w1,0,0) | d_full (w2..w5) | rgb (w6,w7,w8,0)]
        g = uc.reshape(3, 3, 3, 4)
        head = np.concatenate([g[:, :, 0, :2], g[:, :, 1, :3], g[:, :, 2, :3]], axis=2)[..., None]   # u, dFull, rgb
    else:          # 18 vec4 = [u taps (w0,w1,0,0) x9 | rgb taps (w2,w3,w4,0) x9]
        g = uc.reshape(2, 3, 3, 4)
        head = np.concatenate([g[0][..., :2], g[1][..., :3]], axis=2)[..., None]                    # u, rgb
    head = (np.ascontiguousarray(head), w["upCombine"]["bias"])

    @tf.function(input_signature=[tf.TensorSpec([1, H, Wd, 3], tf.float32, name="image")])
    def segment(image):
        # ---- wrapper down path -> adapted base input
        if v in "AB":
            d = conv(image, *down1, 3 if v == "B" else 2, SYM1, "relu")
            adapted = conv(d, *adapter)
        elif v == "D":
            d_full = conv(image, *down1, 1, SYM1, "relu")                           # full canvas res
            d1 = resize(d_full, round(H / 1.25), round(Wd / 1.25))
            adapted = conv(conv(d1, *down2, 2, SYM1, "relu"), *adapter)
        else:  # E
            d1 = conv(image, *down1, 2, SYM1, "relu")
            adapted = conv(conv(d1, *down2, 2, SYM1, "relu"), *adapter)
        assert tuple(adapted.shape[1:3]) == (bh, bw), adapted.shape

        # ---- base -> pre-head feature, interpolated to base-input res (forward_features)
        feat = resize(base_net(adapted, W["base"]), bh, bw)
        carrier = conv(feat, *expand, act="relu")                                   # c_up = 2

        # ---- wrapper up path -> alpha at canvas res
        if v in "DE":
            u1 = conv(tf.concat([resize(carrier, d1.shape[1], d1.shape[2]), d1], -1), *up1, act="relu")
            u = resize(u1, H, Wd)
        else:
            u = resize(carrier, H, Wd)
        x = tf.concat([u, d_full, image] if v == "D" else [u, image], axis=-1)
        return tf.sigmoid(conv(x, *head), name="segmentation_masks")

    return segment


def to_tflite(fn, fp16=True):
    c = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], fn)
    if fp16:   # float16 weights + DEQUANTIZE, float32 compute (MediaPipe's "float16" scheme)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    return c.convert()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--portrait", action="store_true", help="transposed canvas for portrait (e.g. phone) frames")
    a = ap.parse_args()
    buf = to_tflite(build(load_pack(a.weights), a.tier, a.portrait), fp16=not a.float32)
    open(a.out, "wb").write(buf)
    H, W = TIERS[a.tier]["canvas"]
    if a.portrait:
        H, W = W, H
    print(f"saved {a.out} ({len(buf) / 1e6:.2f} MB, input {W}x{H}, variant {TIERS[a.tier]['variant']})")
