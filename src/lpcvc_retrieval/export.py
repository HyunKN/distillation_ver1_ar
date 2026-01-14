from __future__ import annotations
import os
import torch
from .model import OnnxWrapper, ClipLite

def export_onnx(model: ClipLite, onnx_path: str, opset: int = 17):
    model.eval()
    wrapper = OnnxWrapper(model)
    dummy_img = torch.zeros(1,3,224,224, dtype=torch.float32)
    dummy_txt = torch.zeros(1,77, dtype=torch.int32)
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_txt),
        onnx_path,
        input_names=["image","text_input"],
        output_names=["image_embedding","text_embedding"],
        opset_version=int(opset),
        dynamic_axes={
            "image": {0:"batch"},
            "text_input": {0:"batch"},
            "image_embedding": {0:"batch"},
            "text_embedding": {0:"batch"},
        },
    )


def export_onnx_split(model: ClipLite, out_dir: str = "exported_onnx", opset: int = 18,
                      img_name: str = "image_encoder.onnx", txt_name: str = "text_encoder.onnx"):
    """Export student model as TWO ONNX files (image/text encoders) in the LPCVC sample style.

    - image_encoder.onnx:  input 'image' float32 (1,3,224,224) -> output 'embedding' float32 (1,D)
    - text_encoder.onnx:   input 'text'  int32   (1,77)        -> output 'text_embedding' float32 (1,D)

    Notes:
    - Competition evaluation feeds image as float32 in [0,1] (after /255) and text as int32.
    - We keep any extra normalization (CLIP mean/std) INSIDE the model (VisionTower).
    - TextTower already casts int32 -> int64 internally for embedding lookup.
    """
    import torch
    import torch.nn as nn

    class _ImageEnc(nn.Module):
        def __init__(self, m: ClipLite):
            super().__init__()
            self.m = m
        def forward(self, image: torch.Tensor):
            return self.m.encode_image(image)

    class _TextEnc(nn.Module):
        def __init__(self, m: ClipLite):
            super().__init__()
            self.m = m
        def forward(self, text: torch.Tensor):
            return self.m.encode_text(text)

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    dummy_img = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    dummy_txt = torch.zeros(1, 77, dtype=torch.int32)

    image_onnx_path = os.path.join(out_dir, img_name)
    text_onnx_path  = os.path.join(out_dir, txt_name)

    torch.onnx.export(
        _ImageEnc(model),
        dummy_img,
        image_onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=int(opset),
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    torch.onnx.export(
        _TextEnc(model),
        dummy_txt,
        text_onnx_path,
        input_names=["text"],
        output_names=["text_embedding"],
        opset_version=int(opset),
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    return image_onnx_path, text_onnx_path
