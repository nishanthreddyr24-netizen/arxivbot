"""No test may touch the real cache.

This is not a tidiness rule. The suite destroyed a user's data: five specs in
the live store were overwritten by ``tests/test_store.py``, among them the
twelve-component Transformer spec, replaced by a fixture holding one component
called ``encoder``. One entry was left containing the literal text
``{not json``.

``test_store.py`` did set ``ARXIVBOT_CACHE`` to a temp directory. It did not
help, because :func:`arxivbot.storage.store` memoises the store in a module
global on first use. Any earlier test that so much as read from the store
built a ``DiskStore`` pointed at the real cache directory, and from then on the
environment variable was decoration - a later test's ``put`` went to the user's
disk.

So isolation has to reset the global as well as the variable, it has to happen
for every test rather than the one module that remembered, and it has to be
checked rather than assumed. The last fixture here does the checking: it reads
back where the store actually points and fails the run if that is anywhere but
the temp directory. A guard that cannot observe its own failure is how this
happened in the first place.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Every test gets an empty store in its own temp directory."""
    from arxivbot import storage

    # tmp_path itself, not a subdirectory: a test that sets ARXIVBOT_CACHE to
    # its own tmp_path is isolated too, and the check below should accept it.
    root = tmp_path
    monkeypatch.setenv("ARXIVBOT_CACHE", str(root))
    # The variable alone is not enough: the store is memoised, so a store
    # built by an earlier test would survive and keep pointing at the real
    # cache. Clear it before, and again after, so nothing leaks either way.
    monkeypatch.delenv("ARXIVBOT_STORE", raising=False)
    monkeypatch.delenv("ARXIVBOT_STORE_PATH", raising=False)
    storage.use(None)
    try:
        yield root
    finally:
        storage.use(None)


@pytest.fixture(autouse=True)
def _store_stays_inside_the_sandbox(isolated_store):
    """Fail the test if anything made the store point at the real cache.

    Runs after the test body, so it catches a store rebuilt mid-test as well
    as one that was never isolated.
    """
    yield
    from arxivbot import storage

    live = storage._store
    root = getattr(live, "root", None)
    if root is None:
        return  # a SqlStore the test installed deliberately; nothing on disk
    assert root == isolated_store or isolated_store in root.parents, (
        f"the store escaped the sandbox and pointed at {root} - a test was "
        f"about to write to the real cache"
    )
