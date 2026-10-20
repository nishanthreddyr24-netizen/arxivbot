"""The HTTP edge: what the browser is told when the request is wrong.

A 500 says the server broke. For `not-a-real-id` nothing broke - the string
was never an arXiv id - and four endpoints were answering 500 anyway, so the
page showed "request failed (500)" and the reader had no way to know the typo
was theirs to fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    import index

    return TestClient(index.app)


# Every endpoint that takes a paper identifier.
ID_ROUTES = [
    ("/api/paper", {"id": "not-a-real-id"}),
    ("/api/spec", {"id": "not-a-real-id"}),
    ("/api/components", {"id": "not-a-real-id"}),
    ("/api/ask", {"id": "not-a-real-id", "q": "anything"}),
    ("/api/generate", {"id": "not-a-real-id", "q": "anything"}),
    (
        "/api/merge",
        {"host": "not-a-real-id", "donor": "1512.03385", "component": "x"},
    ),
]


@pytest.mark.parametrize("route,params", ID_ROUTES)
def test_an_unparseable_id_is_the_callers_mistake(client, route, params):
    """4xx, never 5xx: the id is wrong, not the server."""
    response = client.get(route, params=params)
    assert 400 <= response.status_code < 500, (
        f"{route} answered {response.status_code} for an id that is not an "
        f"arXiv id, which tells the reader the server is broken"
    )


def test_the_reason_is_in_the_body(client):
    """"request failed (500)" taught nobody anything; the id should be named."""
    response = client.get("/api/components", params={"id": "not-a-real-id"})
    assert response.status_code == 400
    assert "not-a-real-id" in str(response.json()["detail"])


def test_starting_an_extraction_on_a_bad_id_is_also_a_400(client):
    response = client.post("/api/extract", json={"paper": "not-a-real-id"})
    assert response.status_code == 400


def test_a_well_formed_id_with_nothing_stored_is_a_404(client):
    """A 400 must not swallow the different case of 'valid, but unknown'."""
    response = client.get("/api/components", params={"id": "9999.99999"})
    assert response.status_code == 404
