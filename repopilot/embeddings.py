"""Text → a normalized vector. One small CPU model, reused between requests."""

from functools import lru_cache
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@lru_cache(maxsize=1)
def embedding_model():
    # Lazy import/load: V0 can still start without downloading the V1 model.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu", cache_folder=os.getenv("EMBEDDING_CACHE", ".model-cache"))


def embed(texts: list[str]) -> list[list[float]]:
    # Normalization makes vector length 1; cosine measures direction similarity.
    return embedding_model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
