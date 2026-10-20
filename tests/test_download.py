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


def _zip_without_utf8_flag(path, name_bytes: bytes, data: bytes):
    """A one-entry zip whose name is raw bytes with the UTF-8 flag unset,
    as written by many Windows zip tools (DA-2K.zip has such an entry)."""
    import struct
    import zlib
    crc = zlib.crc32(data)
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0, crc, len(data), len(data),
                        len(name_bytes), 0) + name_bytes + data
    central = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 0, 0, 0, crc, len(data), len(data),
                          len(name_bytes), 0, 0, 0, 0, 0, 0) + name_bytes
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local), 0)
    path.write_bytes(local + central + end)


@pytest.mark.parametrize("encoding", ["utf-8", "gbk"])
def test_extract_zip_recovers_non_ascii_names(tmp_path, encoding):
    from src.download import extract_zip
    name = "images/transparent_reflective/51685585792_a553d5930b_h的副本.jpg"
    archive = tmp_path / "a.zip"
    _zip_without_utf8_flag(archive, name.encode(encoding), b"jpeg bytes")
    extract_zip(archive, tmp_path / "out")
    assert (tmp_path / "out" / name).read_bytes() == b"jpeg bytes"


def test_plain_extractall_garbles_the_same_name(tmp_path):
    """Documents the original bug: zipfile decodes flagless names as cp437."""
    import zipfile
    name = "images/x/a_的副本.jpg"
    archive = tmp_path / "a.zip"
    _zip_without_utf8_flag(archive, name.encode("utf-8"), b"x")
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp_path / "out")
    assert not (tmp_path / "out" / name).exists()
