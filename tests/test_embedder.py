"""The default embedding path.

This file exists because of a real escape: `model2vec` was the config default but
shipped as an optional extra and had no test touching it, so `pip install
stillsane` produced a tool that failed on its first command. Every test in the
suite used the hashing fallback, so nothing noticed.

The offline tests below cover the wiring. The `network` ones cover the model
itself and are deselected by default (see `addopts` in pyproject) so a plain
`pytest` still runs with no download.
"""

from __future__ import annotations

import numpy as np
import pytest

from stillsane.config import Config
from stillsane.signals import HashingEmbedder, Model2VecEmbedder, default_embedder
from stillsane.signals.semantic import DEFAULT_MODEL

MODEL2VEC_CONFIG = {
    "targets": [{"name": "prod", "base_url": "https://x/v1", "model": "m"}],
    "probes": [{"id": "p", "prompt": "hello"}],
}


# --- Wiring (offline) -----------------------------------------------------


def test_the_config_default_is_model2vec():
    """If this changes, the network tests below stop covering the default path."""
    assert Config.model_validate(MODEL2VEC_CONFIG).embedder == "model2vec"


def test_default_embedder_honours_the_config_value():
    assert isinstance(default_embedder("model2vec"), Model2VecEmbedder)
    assert isinstance(default_embedder("hashing"), HashingEmbedder)


def test_model2vec_is_importable():
    """It is a hard dependency now, so a plain install must provide it.

    The failure this guards against is not subtle -- it is `stillsane baseline`
    exiting 3 on a brand-new install -- but it is invisible without this line.
    """
    import model2vec  # noqa: F401


def test_constructing_the_embedder_does_not_load_the_model():
    """Construction must stay cheap: `build_signals` makes one per probe, and a
    config error should surface before anything spends 32MB of bandwidth."""
    embedder = Model2VecEmbedder()
    assert embedder._model is None


def test_a_broken_model_name_explains_itself():
    """An air-gapped machine should get advice, not a traceback from the hub client."""
    embedder = Model2VecEmbedder("definitely-not-a-real-org/definitely-not-a-real-model")
    with pytest.raises(RuntimeError, match="embedder: hashing"):
        embedder.encode(["anything"])


# --- The model itself (network) -------------------------------------------


@pytest.fixture(scope="module")
def model2vec():
    return default_embedder("model2vec")


@pytest.mark.network
def test_model2vec_encodes(model2vec):
    vectors = model2vec.encode(["hello world", "something else entirely"])
    assert vectors.shape[0] == 2
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


@pytest.mark.network
def test_model2vec_ranks_distances_sensibly(model2vec):
    """The property the whole tool rests on: semantically closer text is nearer.

    Asserted as an ordering rather than against fixed numbers, so it survives a
    model update without becoming a flaky test about specific magnitudes.
    """
    texts = [
        '{"total": 1240.50, "due_date": "2026-07-01"}',
        '{"due_date": "2026-07-01", "total": 1240.5}',
        "Here you go! The total is 1240.50, due on the 1st of July.",
        "I cannot help with that request.",
    ]
    v = model2vec.encode(texts)

    def distance(i: int, j: int) -> float:
        return float(1 - np.dot(v[i], v[j]))

    reordered = distance(0, 1)
    rewritten = distance(0, 2)
    unrelated = distance(0, 3)
    assert reordered < rewritten < unrelated


@pytest.mark.network
def test_key_reordering_stays_under_the_semantic_floor(model2vec):
    """Same JSON with keys swapped must land inside `SemanticDistance.floor`.

    Otherwise every temperature-0 JSON probe alerts on key ordering, which is not
    drift, and the tool is unusable for the single most common probe shape.
    """
    from stillsane.signals.semantic import SemanticDistance

    v = model2vec.encode(
        [
            '{"total": 1240.50, "due_date": "2026-07-01"}',
            '{"due_date": "2026-07-01", "total": 1240.50}',
        ]
    )
    assert float(1 - np.dot(v[0], v[1])) < SemanticDistance.floor


@pytest.mark.network
def test_the_model_name_we_ship_still_exists(model2vec):
    assert model2vec.model_name == DEFAULT_MODEL
    assert model2vec.encode(["x"]).shape[0] == 1
