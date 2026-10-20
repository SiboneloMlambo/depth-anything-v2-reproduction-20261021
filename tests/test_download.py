"""Hash-manifest behaviour of src/download.py (no network)."""
import pytest

from src.download import Manifest, sha256


def test_manifest_records_then_verifies(tmp_path):
    f = tmp_path / "ckpt.pth"
    f.write_bytes(b"weights")
    m = Manifest(tmp_path / "manifests" / "SHA256SUMS")
    digest = m.verify_or_record("checkpoints/ckpt.pth", f)
    assert digest == sha256(f)

    # A fresh Manifest reads the committed file and verifies against it.
    m2 = Manifest(tmp_path / "manifests" / "SHA256SUMS")
    assert m2.entries == {"checkpoints/ckpt.pth": digest}
    m2.verify_or_record("checkpoints/ckpt.pth", f)

    f.write_bytes(b"different weights")
    with pytest.raises(SystemExit):
        Manifest(tmp_path / "manifests" / "SHA256SUMS").verify_or_record("checkpoints/ckpt.pth", f)
