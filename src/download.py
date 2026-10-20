"""
Downloads both checkpoints, DA-2K and the DIODE validation split, and
verifies every file against manifests/SHA256SUMS.

  - If a file's hash is already in the manifest, it must match or the run stops.
  - If it is not (the very first run), the hash is computed and appended to
    the manifest. Commit manifests/SHA256SUMS after that first run; from then
    on every later run verifies against it.

Stages already present on disk are skipped (their hashes are still checked
for checkpoints). Hugging Face revisions are recorded in manifests/sources.json.

Usage:
    python -m src.download [--root .]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

CHECKPOINTS = {
    "checkpoints/depth_anything_v1_vits.pth": {
        "repo_id": "LiheYoung/Depth-Anything", "repo_type": "space",
        "filename": "checkpoints/depth_anything_vits14.pth",
    },
    "checkpoints/depth_anything_v2_vits.pth": {
        "repo_id": "depth-anything/Depth-Anything-V2-Small", "repo_type": "model",
        "filename": "depth_anything_v2_vits.pth",
    },
}
DA2K = {"repo_id": "depth-anything/DA-2K", "repo_type": "dataset", "filename": "DA-2K.zip"}
DIODE_VAL_URL = "https://diode-dataset.s3.amazonaws.com/val.tar.gz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, str] = {}
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    digest, name = line.split(maxsplit=1)
                    self.entries[name.strip()] = digest

    def verify_or_record(self, name: str, file_path: Path) -> str:
        digest = sha256(file_path)
        if name in self.entries:
            if self.entries[name] != digest:
                raise SystemExit(
                    f"SHA256 mismatch for {name}:\n  expected {self.entries[name]}\n  got      {digest}\n"
                    "The upstream file changed or the download is corrupt. Do not continue; log it in run_log.md.")
            print(f"[download] verified {name}")
        else:
            self.entries[name] = digest
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                if self.path.stat().st_size == 0:
                    f.write("# sha256  name   (first recorded by src/download.py; commit this file)\n")
                f.write(f"{digest}  {name}\n")
            print(f"[download] recorded new hash for {name}: {digest}  -> commit {self.path}")
        return digest


def _hf_revision(spec) -> str | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = {"model": api.model_info, "space": api.space_info, "dataset": api.dataset_info}[spec["repo_type"]]
        return info(spec["repo_id"]).sha
    except Exception as e:  # recording the revision is best-effort
        print(f"[download] could not read revision for {spec['repo_id']}: {e}")
        return None


def _hf_download(spec) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"], repo_type=spec["repo_type"]))


def download_checkpoints(root: Path, manifest: Manifest, sources: dict):
    for rel, spec in CHECKPOINTS.items():
        target = root / rel
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_hf_download(spec), target)
            sources[rel] = {**spec, "revision": _hf_revision(spec), "downloaded": date.today().isoformat()}
        manifest.verify_or_record(rel, target)


def zip_member_name(info: zipfile.ZipInfo) -> str:
    """The real file name of a zip entry.

    Entries without the zip "UTF-8" flag (bit 11) are decoded by Python as
    cp437, which garbles non-ASCII names. DA-2K.zip has such an entry
    (a file name containing Chinese characters, written by a tool that stored
    UTF-8 or GBK bytes without setting the flag), so its image landed on disk
    under a garbled name and the annotation key could not find it. Recover
    the original bytes and decode them as UTF-8, then GBK.
    """
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        return info.filename
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return info.filename


def extract_zip(archive: Path, dest: Path) -> None:
    """extractall, but with non-ASCII entry names decoded correctly."""
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            info.filename = zip_member_name(info)
            z.extract(info, dest)


def _da2k_images_missing(target: Path) -> list[str]:
    raw = json.loads((target / "annotations.json").read_text(encoding="utf-8"))
    return sorted(k for k in raw if not (target / k).is_file())


def download_da2k(root: Path, manifest: Manifest, sources: dict):
    target = root / "data" / "da2k"
    if (target / "annotations.json").is_file():
        missing = _da2k_images_missing(target)
        if not missing:
            print("[download] DA-2K already present; skipping")
            return
        print(f"[download] DA-2K present but {len(missing)} annotated images missing "
              f"(first: {missing[:3]}); re-extracting")
        shutil.rmtree(target)
    archive = _hf_download(DA2K)
    manifest.verify_or_record("DA-2K.zip", archive)
    sources["DA-2K.zip"] = {**DA2K, "revision": _hf_revision(DA2K), "downloaded": date.today().isoformat()}

    staging = root / "data" / "_da2k_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    extract_zip(archive, staging)
    found = list(staging.rglob("annotations.json"))
    if len(found) != 1:
        raise SystemExit(f"Expected one annotations.json in DA-2K.zip, found {len(found)}")
    target.mkdir(parents=True, exist_ok=True)
    for item in found[0].parent.iterdir():
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    shutil.rmtree(staging)
    missing = _da2k_images_missing(target)
    if missing:
        raise SystemExit(f"{len(missing)} annotated DA-2K images missing after extraction (first: {missing[:3]})")
    print(f"[download] DA-2K ready in {target}")


def download_diode_val(root: Path, manifest: Manifest, sources: dict):
    diode_root = root / "data" / "diode"
    if (diode_root / "val").is_dir():
        print("[download] DIODE val already present; skipping")
        return
    diode_root.mkdir(parents=True, exist_ok=True)
    archive = diode_root / "val.tar.gz"
    print(f"[download] fetching {DIODE_VAL_URL} (~2.6 GB)")
    urllib.request.urlretrieve(DIODE_VAL_URL, archive)
    manifest.verify_or_record("diode/val.tar.gz", archive)
    sources["diode/val.tar.gz"] = {"url": DIODE_VAL_URL, "downloaded": date.today().isoformat()}
    with tarfile.open(archive, "r:gz") as tar:
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(diode_root, **kwargs)
    archive.unlink()
    if not (diode_root / "val").is_dir():
        raise SystemExit("data/diode/val not found after extracting DIODE")
    print(f"[download] DIODE val ready in {diode_root / 'val'}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    manifest = Manifest(root / "manifests" / "SHA256SUMS")
    sources_path = root / "manifests" / "sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.is_file() else {}

    download_checkpoints(root, manifest, sources)
    download_da2k(root, manifest, sources)
    download_diode_val(root, manifest, sources)

    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
