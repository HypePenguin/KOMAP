"""KOMAP EfficientNetV2-S U-Net: standalone inference and probability ensemble API.

CLI: python efficient.py --input images --output predictions --save-probs
API: predictor = EfficientPredictor(); probs = predictor.predict_proba(rgb_uint8)
probs is float32 [4, H, W], in CLASSES order at the original image resolution.
Only this Python file and model/best.pt are needed after installing dependencies.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "model" / "best.pt"
MODEL_SHA256 = "e464af75e27dc7e8204287b2dc6f11fca8184c05aad09db5050dd0a320b0c626"
ENCODER = "tu-tf_efficientnetv2_s.in21k_ft_in1k"
CLASSES = ("Al", "Primary Si", "Al3Ni", "Eutectic Si")
PALETTE = np.array([[153, 127, 76], [76, 178, 76], [25, 76, 153], [204, 204, 204]], dtype=np.uint8)
TILE = 768
STRIDE = 384
EXPECTED_PARAMETERS = 22_093_908
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: str | Path) -> np.ndarray:
    """Read uint8 RGB without EXIF rotation, matching the original inference."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def tile_positions(length: int, tile: int = TILE, stride: int = STRIDE) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@contextmanager
def keep_system_awake():
    """Keep Windows awake during CLI inference; display settings are unchanged."""
    if os.name != "nt":
        yield
        return
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint32
    continuous, system_required = 0x80000000, 0x00000001
    previous = kernel.SetThreadExecutionState(continuous | system_required)
    if not previous:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        kernel.SetThreadExecutionState(previous)


class EfficientPredictor:
    """Load once, then predict any number of RGB images for an ensemble.

    No local module, configuration JSON, training data or network download is used.
    CUDA uses the original FP16 autocast inference; CPU uses FP32 and may differ
    slightly. Stored weights are always loaded unchanged. Importing this module
    does not load a model, launch inference or write files.
    """

    classes = CLASSES
    palette = PALETTE

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, *, device: str = "auto", batch_size: int = 8):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices: cpu, cuda, cuda:N")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install CUDA-enabled PyTorch or select --device cpu")
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"Model checkpoint missing: {self.weights}")
        self.model_sha256 = file_sha256(self.weights)
        if self.model_sha256 != MODEL_SHA256:
            raise ValueError("Checkpoint SHA256 differs from the shared train89 EMA model")
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=True)
        if checkpoint.get("epoch") != 25 or checkpoint.get("model_kind") != "EMA":
            raise ValueError("Expected the selected epoch-25 EMA checkpoint")
        self.model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=3,
                              classes=4, decoder_channels=(256, 128, 64, 32, 16)).float()
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        if self.parameter_count != EXPECTED_PARAMETERS:
            raise ValueError("Model architecture differs from the recorded architecture")
        self.selected_epoch = checkpoint["epoch"]
        self.batch_size = batch_size
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model = self.model.to(self.device).to(memory_format=torch.channels_last)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, image: np.ndarray | str | Path) -> np.ndarray:
        """Return original-resolution float32 probabilities shaped [4, H, W].

        Arrays must be uint8 RGB [H,W,3], never OpenCV BGR or normalized tensors.
        D4 probability averaging, uniform tile blending and native resize follow
        the original implementation exactly. Do not apply another softmax.
        """
        if isinstance(image, (str, Path)):
            image = load_rgb(image)
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected an RGB numpy array with shape [H,W,3], or an image path")
        if image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError("Expected a non-empty uint8 RGB image in the range 0..255")
        h, w = image.shape[:2]
        up = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        ih, iw = up.shape[:2]
        ph, pw = max(ih, TILE), max(iw, TILE)
        up = cv2.copyMakeBorder(up, 0, ph - ih, 0, pw - iw, cv2.BORDER_REFLECT_101)
        result = np.zeros((4, ph, pw), np.float32)
        weight = np.zeros((ph, pw), np.float32)
        coords = [(y, x) for y in tile_positions(ph) for x in tile_positions(pw)]
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=torch.float32)[None, :, None, None]
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=torch.float32)[None, :, None, None]
        for start in range(0, len(coords), self.batch_size):
            batch_coords = coords[start:start + self.batch_size]
            patches = np.stack([up[y:y + TILE, x:x + TILE].transpose(2, 0, 1)
                                for y, x in batch_coords]).astype(np.float32) / 255
            tensor = torch.from_numpy(patches).to(self.device)
            tensor = (tensor - mean) / std
            total = None
            for flip in (False, True):
                for k in range(4):
                    transformed = torch.rot90(tensor, k, (-2, -1))
                    if flip:
                        transformed = torch.flip(transformed, [-1])
                    amp = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else torch.autocast("cpu", enabled=False)
                    with amp:
                        logits = self.model(transformed)
                    prob = logits.float().softmax(1)
                    if flip:
                        prob = torch.flip(prob, [-1])
                    prob = torch.rot90(prob, -k, (-2, -1))
                    total = prob if total is None else total + prob
            probs = (total / 8).cpu().numpy()
            for (y, x), prob in zip(batch_coords, probs):
                result[:, y:y + TILE, x:x + TILE] += prob
                weight[y:y + TILE, x:x + TILE] += 1
        result = result[:, :ih, :iw] / weight[None, :ih, :iw]
        native = cv2.resize(result.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
        if not np.isfinite(native).all():
            raise FloatingPointError("Non-finite prediction probabilities. Check RGB uint8 input; retry with device='cpu' for FP32 inference.")
        return np.ascontiguousarray(native.transpose(2, 0, 1), dtype=np.float32)

    def predict(self, image: np.ndarray | str | Path) -> np.ndarray:
        """Return uint8 class IDs [H,W], values 0..3 in CLASSES order."""
        return self.predict_proba(image).argmax(0).astype(np.uint8)

    def __call__(self, image: np.ndarray | str | Path) -> np.ndarray:
        return self.predict_proba(image)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Image file or folder (non-recursive)")
    parser.add_argument("--output", type=Path, help="New output folder; existing directories are rejected")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Tiles per batch; original setting: 8")
    parser.add_argument("--save-probs", action="store_true", help="Save float32 [4,H,W] .npy for ensembling")
    parser.add_argument("--check-model", action="store_true", help="Check checkpoint SHA and load model, then exit")
    args = parser.parse_args(argv)
    files = []
    if not args.check_model:
        if args.input is None or not args.input.exists():
            parser.error("Provide --input pointing to an image file or folder")
        files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS) if args.input.is_dir() else [args.input]
        if not files or any(p.suffix.lower() not in EXTENSIONS for p in files):
            parser.error("No supported input images; use PNG/JPG/BMP/TIF")
        names = [p.stem.removesuffix("_image") for p in files]
        if len(set(name.casefold() for name in names)) != len(names):
            parser.error("Input image names would produce duplicate output files")
        output = args.output or ROOT / "predictions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if output.exists():
            parser.error(f"Output already exists; use a new directory: {output}")
    torch.set_num_threads(4)
    device = "cpu" if args.check_model and args.device == "auto" else args.device
    predictor = EfficientPredictor(args.weights, device=device, batch_size=args.batch_size)
    info = {"model_sha256": predictor.model_sha256, "selected_epoch": predictor.selected_epoch,
            "parameters": predictor.parameter_count, "device": str(predictor.device),
            "model_kind": "EMA", "weight_conversion": "none", "classes": list(CLASSES),
            "palette": PALETTE.tolist(), "tile": TILE, "stride": STRIDE, "upscale": 2,
            "tta": "D4", "blend": "uniform probability mean", "batch_size": args.batch_size,
            "probability_layout": "C,H,W", "probability_dtype": "float32",
            "resolution": "original image", "mIoU": None}
    if args.check_model:
        print(json.dumps(info, ensure_ascii=False))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "labels").mkdir()
    if args.save_probs:
        (output / "probabilities").mkdir()
    with keep_system_awake():
        for index, (path, name) in enumerate(zip(files, names), 1):
            probs = predictor.predict_proba(path)
            mask = probs.argmax(0).astype(np.uint8)
            Image.fromarray(PALETTE[mask]).save(output / (name + "_mask.png"))
            Image.fromarray(mask).save(output / "labels" / (name + "_label.png"))
            if args.save_probs:
                np.save(output / "probabilities" / (name + "_prob.npy"), probs, allow_pickle=False)
            print(f"[{index}/{len(files)}] {path.name}", flush=True)
    info.update(count=len(files), inputs=[p.name for p in files], save_probs=args.save_probs,
                note="No ground-truth metrics computed by this inference command.")
    (output / "inference_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OUTPUT: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
