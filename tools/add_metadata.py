"""Attach MediaPipe ImageSegmenter metadata identical in shape to the official selfie_segmenter:
  input  "image": RGB, NormalizationOptions mean [0] / std [255]  (uint8 pixels -> [0, 1])
  output "segmentation_masks": 1 channel, labels.txt = "selfie", SEGMENTER_METADATA activation NONE
         (the graph already applies sigmoid)

usage (needs `pip install mediapipe`): python tools/add_metadata.py in.tflite out.tflite
"""
import sys
import types


def _shim_metadata_version(min_version="1.5.0"):
    # mediapipe 0.10.3x / 1.x wheels ship metadata.py without its native
    # _pywrap_metadata_version helper (only used to stamp min_parser_version).
    # Provide a stand-in returning the value the official selfie_segmenter carries.
    try:
        from mediapipe.tasks.cc.metadata.python import _pywrap_metadata_version  # noqa: F401
        return
    except ImportError:
        pass
    for n in ["mediapipe.tasks.cc", "mediapipe.tasks.cc.metadata", "mediapipe.tasks.cc.metadata.python"]:
        sys.modules.setdefault(n, types.ModuleType(n))
    m = types.ModuleType("mediapipe.tasks.cc.metadata.python._pywrap_metadata_version")
    m.GetMinimumMetadataParserVersion = lambda buf: min_version
    sys.modules[m.__name__] = m
    sys.modules["mediapipe.tasks.cc.metadata.python"]._pywrap_metadata_version = m


_shim_metadata_version()
from mediapipe.tasks.python.metadata.metadata_writers import image_segmenter, metadata_writer  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
writer = image_segmenter.MetadataWriter.create(
    bytearray(open(src, "rb").read()),
    input_norm_mean=[0.0],
    input_norm_std=[255.0],
    labels=metadata_writer.Labels().add(["selfie"]),
    activation=image_segmenter.Activation.NONE,
)
model, _ = writer.populate()

# The writer only sets name/description; add version/author/license (the weights are MIT and
# the notice must travel with copies of them).
import flatbuffers  # noqa: E402
from mediapipe.tasks.metadata import metadata_schema_py_generated as fb  # noqa: E402
from mediapipe.tasks.python.metadata import metadata as md  # noqa: E402

meta = fb.ModelMetadataT.InitFromObj(fb.ModelMetadata.GetRootAsModelMetadata(md.get_metadata_buffer(model), 0))
meta.version = "1"
meta.author = "Longpipe weights by Sam Bhattacharyya (github.com/sb2702/longpipe); converted to TFLite"
meta.license = ("Weights: MIT License, Copyright (c) 2026 Sam Bhattacharyya. "
                "https://github.com/sb2702/longpipe/blob/main/WEIGHTS_LICENSE")
b = flatbuffers.Builder(0)
b.Finish(meta.Pack(b), md.MetadataPopulator.METADATA_FILE_IDENTIFIER)
pop = md.MetadataPopulator.with_model_buffer(model)
pop.load_metadata_buffer(b.Output())   # labels.txt stays packed from the first pass
pop.populate()
model = pop.get_model_buffer()

open(dst, "wb").write(model)
print(md.convert_to_json(md.get_metadata_buffer(model)))
print(f"saved {dst}")
