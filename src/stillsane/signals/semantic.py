"""Semantic distance via local embeddings.

The embedder is injected rather than constructed inline so the comparison engine
can be tested with a deterministic stub -- no model download, no network, no
spend. `default_embedder()` is what production actually gets.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ..models import Sample
from .base import PairwiseSignal

#: Static model: 32MB on disk, numpy-only inference, no torch in the dependency
#: tree. Chosen so `pip install stillsane` stays light enough that the five-minute
#: promise survives first contact.
DEFAULT_MODEL = "minishlab/potion-base-8M"

#: The commit of `DEFAULT_MODEL` every baseline is measured against. Without it the
#: hub resolves `main` on each load, so the day the upstream repo pushes a new
#: revision every install silently starts embedding on a different scale, and bands
#: learned under the old vectors are compared against distances from the new ones.
#: Bump it deliberately -- `config.config_hash` folds any change into the baseline
#: hash, so a bump forces a recapture instead of silently mismatching.
DEFAULT_MODEL_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"

#: The revision that was current when stillsane first shipped, and so the one every
#: existing baseline was captured under. Hashed as plain "model2vec", so pinning did
#: not invalidate baselines that were already valid: same weights, same scale.
ORIGINAL_MODEL_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalised row vector per input text."""


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Model2VecEmbedder:
    """Wraps model2vec. Imported lazily so the package loads without it."""

    def __init__(self, model_name: str = DEFAULT_MODEL, revision: str | None = None) -> None:
        self.model_name = model_name
        # The pin belongs to the default model only; another repo has other commits.
        self.revision = revision or (DEFAULT_MODEL_REVISION if model_name == DEFAULT_MODEL else None)
        self._model = None

    def _resolve_folder(self) -> str:
        """Local snapshot of the pinned revision, downloading only if it is absent.

        Cache first: `from_pretrained` on a repo id re-resolves against the hub on
        every call (model2vec defaults `force_download=True`), which costs a
        connect timeout per run on an egress-restricted runner and follows `main`.
        `HF_HUB_OFFLINE=1` is honoured by `huggingface_hub` for the download step.
        """
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return snapshot_download(self.model_name, revision=self.revision, local_files_only=True)
        except LocalEntryNotFoundError:
            return snapshot_download(self.model_name, revision=self.revision)

    def _load(self):
        if self._model is None:
            # Hugging Face's progress bars go to stderr on the first run. This is a
            # monitoring tool whose output gets piped and parsed, so keep it quiet
            # unless the user has said otherwise.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                from model2vec import StaticModel
            except ImportError as exc:  # pragma: no cover - a broken environment
                raise RuntimeError(
                    "Could not import model2vec, which stillsane depends on for "
                    "semantic drift detection.\n"
                    "  pip install --upgrade stillsane\n"
                    "Or set `embedder: hashing` in your config to run fully offline "
                    "with a weaker signal."
                ) from exc
            try:
                self._model = StaticModel.from_pretrained(
                    self._resolve_folder(), force_download=False
                )
            except Exception as exc:
                # First use downloads ~32MB. An air-gapped box or a blocked egress
                # should get told what happened and how to proceed, not a traceback
                # from somewhere inside the hub client.
                raise RuntimeError(
                    f"Could not load the embedding model {self.model_name!r}: {exc}\n"
                    "The first run downloads it (~32MB) and caches it. If this "
                    "machine has no internet access, set `embedder: hashing` in your "
                    "config to run fully offline with a weaker signal."
                ) from exc
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = np.asarray(self._load().encode(list(texts)), dtype=np.float32)
        return _l2_normalise(vectors)


_TOKEN = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic offline fallback. No download, no deps beyond numpy.

    Hashes word unigrams and character trigrams into a fixed-width space. Good
    enough to notice that output was rewritten wholesale; genuinely worse than a
    real model at spotting a paraphrase that preserves meaning. Documented as
    degraded rather than hidden, because a drift tool that quietly downgrades its
    own sensitivity is worse than one that says so.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            for word in _TOKEN.findall(lowered):
                out[row, self._bucket(word)] += 1.0
            for i in range(len(lowered) - 2):
                out[row, self._bucket(lowered[i : i + 3])] += 0.5
        # Sublinear scaling: stops one repeated token dominating the vector.
        np.log1p(out, out=out)
        return _l2_normalise(out)


def default_embedder(kind: str = "model2vec", model_name: str = DEFAULT_MODEL) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    return Model2VecEmbedder(model_name)


def embedder_identity(kind: str) -> str:
    """What `config_hash` records for this embedder: its name, plus its revision
    when that is not the original one, so a bump invalidates baselines."""
    if kind != "model2vec" or DEFAULT_MODEL_REVISION == ORIGINAL_MODEL_REVISION:
        return kind
    return f"{kind}@{DEFAULT_MODEL_REVISION}"


class SemanticDistance(PairwiseSignal):
    """Cosine distance between response embeddings."""

    name = "semantic_distance"
    #: Below this, two outputs are the same text with trivial differences. Prevents
    #: a byte-identical baseline producing a zero-width band.
    floor = 0.02

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._cache: dict[str, np.ndarray] = {}

    def prepare(self, samples: Sequence[Sample]) -> None:
        """Encode every distinct text in one batch.

        Pairwise scoring touches each text O(n) times; without this the embedder
        would be called once per pair.
        """
        pending = []
        seen = set()
        for s in samples:
            if not s.ok:
                continue
            key = self._key(s.text)
            if key not in self._cache and key not in seen:
                seen.add(key)
                pending.append((key, s.text))
        if not pending:
            return
        vectors = self.embedder.encode([text for _, text in pending])
        for (key, _), vec in zip(pending, vectors, strict=True):
            self._cache[key] = vec

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def _vector(self, sample: Sample) -> np.ndarray:
        key = self._key(sample.text)
        if key not in self._cache:
            self._cache[key] = self.embedder.encode([sample.text])[0]
        return self._cache[key]

    def distance(self, a: Sample, b: Sample) -> float | None:
        if not (a.ok and b.ok):
            return None
        cos = float(np.dot(self._vector(a), self._vector(b)))
        return float(min(1.0, max(0.0, 1.0 - cos)))
