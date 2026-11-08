"""Bounded bytes and path safety in a temporary store, without internet inputs."""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from gidr.infrastructure.artifacts.store import ArtifactError, ArtifactStore
from gidr.infrastructure.localfs import LocalLock


def test_concurrent_publication_is_no_clobber_and_bounded(tmp_path):
    store = ArtifactStore(tmp_path / "store", max_bytes=16)
    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(store.publish, [b"synthetic"] * 20))
    assert all(r == refs[0] for r in refs) and len(store.published()) == 1
    path = store.path(refs[0].sha256)
    inode = path.stat().st_ino
    store.publish(b"synthetic")
    assert path.stat().st_ino == inode
    with pytest.raises(ArtifactError, match="BYTE_LIMIT"):
        store.publish_chunks([b"a" * 10, b"b" * 10])
    assert not list(store.staging.iterdir())
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_stream_failure_and_split_secret_do_not_publish(tmp_path):
    store = ArtifactStore(tmp_path / "store", forbidden_values=(b"SYNTHETIC-CANARY",))
    with pytest.raises(ArtifactError, match="SECRET"):
        store.publish_chunks([b"SYNTHETIC-", b"CANARY"])

    def broken():
        yield b"partial"
        raise OSError("synthetic input failure")

    with pytest.raises(OSError):
        store.publish_chunks(broken())
    assert not store.published() and not list(store.staging.iterdir())


def test_forged_size_does_not_quarantine_valid_content(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    ref = store.publish(b"synthetic")
    bad = type(ref).model_validate({**ref.model_dump(), "size_bytes": 1})
    with pytest.raises(ArtifactError, match="REFERENCE_SIZE"):
        store.read(bad)
    assert store.read(ref) == b"synthetic" and not list(store.quarantine.iterdir())


def test_symlink_root_digest_and_lock_timeout(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="SYMLINK"):
        ArtifactStore(link)
    store = ArtifactStore(tmp_path / "store")
    ref = store.publish(b"synthetic")
    path = store.path(ref.sha256)
    path.unlink()
    path.symlink_to(outside / "no-file")
    with pytest.raises(ValueError, match="SYMLINK"):
        store.read(ref)
    first = LocalLock(tmp_path / "lock")
    second = LocalLock(tmp_path / "lock", timeout=0.05)
    with first.hold(), pytest.raises(TimeoutError):
        with second.hold():
            pass


@pytest.mark.parametrize("operation", ["fsync", "link"])
def test_filesystem_failure_never_returns_a_published_reference(tmp_path, monkeypatch, operation):
    store = ArtifactStore(tmp_path / "store")

    def fail(*args, **kwargs):
        raise OSError("Synthetic filesystem failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, operation, fail)
        with pytest.raises(OSError):
            store.publish(b"Synthetic failed publication")
    assert not store.published()
    assert not list(store.staging.iterdir())
