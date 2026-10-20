# Data

Nothing in this repository redistributes DA-2K or DIODE. `python -m src.download`
(called by `reproduce.sh`) downloads both and checks them against
`manifests/SHA256SUMS`.

## DA-2K

- Source: https://huggingface.co/datasets/depth-anything/DA-2K (`DA-2K.zip`)
- Format: see `DA-2K.md` in the official V2 repo. `annotations.json` maps each
  image path to a list of `{point1: [h, w], point2: [h, w], closer_point}` entries.
- The scene type is read from the image path. The loader stops if any image's
  path contains none of the eight scenario names. If that happens, fix
  `infer_scene_type` and log the change in `run_log.md`.
- Licence and provenance: see `run_log.md`. **Confirm the licence by reading
  the dataset card yourself before the full run.** If the images turn out not
  to be lawfully usable, stop and say so in `run_log.md`; do not substitute
  another dataset.

Layout after download:
```
data/da2k/
├── annotations.json
└── <image folders, as shipped in DA-2K.zip>
```

## DIODE (validation split only)

- Source: https://diode-dataset.org/ (archive `https://diode-dataset.s3.amazonaws.com/val.tar.gz`)
- Size check: the full run stops unless the loader finds exactly 771 frames
  (the published size of the validation split).

Layout after download:
```
data/diode/val/
├── indoors/scene_*/scan_*/<id>.png, <id>_depth.npy, <id>_depth_mask.npy
└── outdoor/scene_*/scan_*/<id>.png, <id>_depth.npy, <id>_depth_mask.npy
```
