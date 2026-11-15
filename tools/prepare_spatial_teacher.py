"""Stage 0 cache preparation from exact, preprocessed teacher image tensors.

Manifest JSON: [{"sample_id": "episode/frame", "camera_ids": ["front", "wrist"],
                 "images_path": "frame.pt"}]. Each tensor is [V,3,H,W], RGB [0,1],
after the same geometry, normalization inversion and resizing used by the trainer.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def main():
    # [Spatial-DsRA v3]: Precompute frozen teacher features with explicit sample/camera/preprocessing index.
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["openvla", "openpi"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / ("openvla-SF" if args.branch == "openvla" else "openpi-SF/src")
    sys.path.insert(0, str(source))
    module_path = source / ("prismatic/models/spatial_dsra.py" if args.branch == "openvla"
                            else "openpi/models_pytorch/spatial_dsra.py")
    spec = importlib.util.spec_from_file_location("spatial_dsra", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from vggt.models.vggt import VGGT
    teacher = VGGT(enable_camera=False, enable_point=False, enable_depth=False,
                   enable_track=False, feature_only=True).to(args.device)
    teacher.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=False)
    teacher.requires_grad_(False).eval()
    cache = module.FrozenTeacherCache(args.output, args.checkpoint)
    index = []
    for record in json.loads(args.manifest.read_text(encoding="utf-8")):
        path = args.manifest.parent / record["images_path"]
        images = torch.load(path, map_location=args.device, weights_only=True)
        if images.ndim != 4 or images.shape[1] != 3 or len(record["camera_ids"]) != images.shape[0]:
            raise ValueError("Expected [V,3,H,W] tensor and V camera IDs")
        with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                            enabled=str(args.device).startswith("cuda")):
            _, patch_hw = cache.extract(teacher, images.unsqueeze(0), args.layer)
        index.append(dict(sample_id=record["sample_id"], camera_ids=record["camera_ids"],
                          cache_file=cache.key(images, args.layer) + ".pt", patch_hw=patch_hw,
                          images_path=str(path), teacher_id=cache.teacher_id))
    (args.output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(index)} indexed teacher samples in {args.output}")


if __name__ == "__main__":
    main()
