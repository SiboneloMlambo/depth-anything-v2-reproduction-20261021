"""
Loads the authors' released V1-S / V2-S checkpoints, each with its own
release's model definition. V1 and V2 are deliberately not loaded through
the same class: a permissive cross-version load can run with missing
weights and silently invalidate the comparison.

This file is the one place where the authors' code is used. Everything
downstream (data loading, alignment, metrics, comparisons, plots) is
student-written.

The two official repositories are cloned at pinned commits into external/
by `scripts/setup_external.sh` (called from reproduce.sh).

torch is imported lazily so the offline unit tests do not need it.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPO_V1 = REPO_ROOT / "external" / "Depth-Anything"
EXTERNAL_REPO_V2 = REPO_ROOT / "external" / "Depth-Anything-V2"

# ViT-S config, copied verbatim from the official run.py model_configs['vits'].
VITS_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


@contextmanager
def working_directory(path: Path):
    """Temporarily change directory (contextlib.chdir needs Python 3.11+)."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_depth_anything_vits(checkpoint_path: str | Path, model_version: str, device: str | None = None):
    """Load V1-S or V2-S with the matching official implementation.

    Loading is strict: any missing or unexpected key stops the run instead
    of producing plausible-looking output from partially initialised weights.
    """
    import torch

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if model_version == "v1":
        if not EXTERNAL_REPO_V1.is_dir():
            raise ImportError(f"Official V1 repo missing at {EXTERNAL_REPO_V1}; run scripts/setup_external.sh")
        if str(EXTERNAL_REPO_V1) not in sys.path:
            sys.path.insert(0, str(EXTERNAL_REPO_V1))
        from depth_anything.dpt import DepthAnything

        # V1 builds its DINOv2 backbone with
        #   torch.hub.load('torchhub/facebookresearch_dinov2_main', ..., source='local')
        # and that path is relative to the *current directory*. Run from this
        # repo's root it does not exist, so build the model from inside the
        # V1 repository. (Bug found in code review before any real run.)
        with working_directory(EXTERNAL_REPO_V1):
            model = DepthAnything(VITS_CONFIG)
    elif model_version == "v2":
        if not EXTERNAL_REPO_V2.is_dir():
            raise ImportError(f"Official V2 repo missing at {EXTERNAL_REPO_V2}; run scripts/setup_external.sh")
        if str(EXTERNAL_REPO_V2) not in sys.path:
            sys.path.insert(0, str(EXTERNAL_REPO_V2))
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(**VITS_CONFIG)
    else:
        raise ValueError("model_version must be 'v1' or 'v2'")

    device = device or get_device()
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device).eval()
    model._reproduction_version = model_version
    model._reproduction_device = device
    return model


def infer_disparity(model, raw_image_bgr, input_size: int = 518):
    """Raw disparity map (H, W), float32, at the ORIGINAL image resolution.
    Larger value = closer point.

    Each version uses its own official preprocessing and upsampling:
      V2: the official `infer_image` (bilinear upsampling, align_corners=True)
      V1: a copy of the official run.py pipeline (align_corners=False)
    The align_corners difference is the authors', and is recorded in run_log.md.
    """
    import cv2
    import torch
    import torch.nn.functional as F

    version = getattr(model, "_reproduction_version", None)
    if version == "v2":
        return model.infer_image(raw_image_bgr, input_size)
    if version != "v1":
        raise ValueError("Model was not loaded by load_depth_anything_vits().")

    from torchvision.transforms import Compose
    from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
    #Change image from bgr to RGB,then use the transform to change to 518,
    #Then use PrepareForNet to change to Transposes the array from HWC → CHW (height, width, channels → channels, height, width)
    image = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB) / 255.0
    height, width = image.shape[:2]
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    image = transform({"image": image})["image"]
    tensor = torch.from_numpy(image).unsqueeze(0).to(model._reproduction_device)
    with torch.no_grad():
        depth = model(tensor)
    depth = F.interpolate(depth[None], (height, width), mode="bilinear", align_corners=False)[0, 0]
    return depth.detach().cpu().numpy()
