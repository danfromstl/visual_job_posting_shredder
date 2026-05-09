from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class EmbeddingRecommender:
    def __init__(self, cache_dir: Path, model_name: str = MODEL_NAME) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.tag_cache_path = cache_dir / "not_relevant_vectors.jsonl"
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tagger-embeddings")
        self._refresh_future: Future | None = None
        self._model: Any | None = None
        self._model_error: str | None = None
        self._tag_vectors: dict[str, dict[str, Any]] = {}
        self._snippet_vectors: dict[str, list[float]] = {}
        self._average_vector: list[float] | None = None
        self._seed_count = 0
        self._threshold: float | None = None
        self._signature = ""
        self._load_tag_cache()

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._refresh_future is not None and not self._refresh_future.done()
            return {
                "model_name": self.model_name,
                "available": self._model is not None and self._model_error is None,
                "load_error": self._model_error,
                "seed_count": self._seed_count,
                "threshold": self._threshold,
                "refresh_running": running,
                "cached_not_relevant_vectors": len(self._tag_vectors),
                "cached_snippet_vectors": len(self._snippet_vectors),
            }

    def queue_average_refresh(self, annotations: dict[str, dict]) -> None:
        snapshot = dict(annotations)
        with self._lock:
            if self._refresh_future is not None and not self._refresh_future.done():
                return
            self._refresh_future = self._executor.submit(self.refresh_average, snapshot)

    def score_rows(self, annotations: dict[str, dict], rows: list[dict]) -> dict[str, Any]:
        try:
            self.refresh_average(annotations)
        except Exception as exc:
            return self._unavailable_payload(str(exc))

        with self._lock:
            average_vector = self._average_vector
            threshold = self._threshold
            seed_count = self._seed_count

        if average_vector is None or threshold is None or not seed_count:
            return {
                "available": False,
                "reason": "no_seed_tags",
                "message": "No not-relevant seed tags are available yet.",
                "model_name": self.model_name,
                "recommendations": [],
                "seed_count": seed_count,
                "threshold": threshold,
            }

        texts = [str(row.get("text", "")) for row in rows]
        vectors = self._vectors_for_rows(rows, texts)
        recommendations = []
        for row, vector in zip(rows, vectors):
            score = cosine(vector, average_vector)
            label = "not_relevant" if score >= threshold else "relevant"
            recommendations.append(
                {
                    "id": row.get("id"),
                    "score": round(score, 4),
                    "threshold": round(threshold, 4),
                    "suggested_label": label,
                    "model_name": self.model_name,
                }
            )

        return {
            "available": True,
            "model_name": self.model_name,
            "seed_count": seed_count,
            "threshold": round(threshold, 4),
            "recommendations": recommendations,
        }

    def refresh_average(self, annotations: dict[str, dict]) -> dict[str, Any]:
        records = []
        for key, record in sorted(annotations.items()):
            if record.get("not_relevant") is not True:
                continue
            snippet_text = str(record.get("text", "")).strip()
            if snippet_text:
                records.append((key, snippet_text))

        signature = self._signature_for_records(records)
        with self._lock:
            if signature == self._signature and self._average_vector is not None:
                return self.status()

        if not records:
            with self._lock:
                self._average_vector = None
                self._seed_count = 0
                self._threshold = None
                self._signature = signature
            return self.status()

        missing = []
        active_keys = set()
        with self._lock:
            for key, snippet_text in records:
                active_keys.add(key)
                digest = text_hash(snippet_text)
                cached = self._tag_vectors.get(key)
                if cached is None or cached.get("text_hash") != digest:
                    missing.append((key, digest, snippet_text))

        if missing:
            vectors = self._encode([item[2] for item in missing])
            with self._lock:
                for (key, digest, _), vector in zip(missing, vectors):
                    self._tag_vectors[key] = {
                        "key": key,
                        "text_hash": digest,
                        "embedding": vector,
                        "model_name": self.model_name,
                    }
                for key in list(self._tag_vectors):
                    if key not in active_keys:
                        self._tag_vectors.pop(key, None)
                self._write_tag_cache()

        with self._lock:
            vectors = [
                self._tag_vectors[key]["embedding"]
                for key, _ in records
                if key in self._tag_vectors
            ]

        if not vectors:
            with self._lock:
                self._average_vector = None
                self._seed_count = 0
                self._threshold = None
                self._signature = signature
            return self.status()

        dimensions = len(vectors[0])
        sums = [0.0] * dimensions
        for vector in vectors:
            for index, value in enumerate(vector):
                sums[index] += float(value)
        average = normalize_vector([value / len(vectors) for value in sums])
        seed_scores = [cosine(vector, average) for vector in vectors]
        threshold = self._threshold_from_seed_scores(seed_scores)

        with self._lock:
            self._average_vector = average
            self._seed_count = len(vectors)
            self._threshold = threshold
            self._signature = signature

        return self.status()

    def _unavailable_payload(self, reason: str) -> dict[str, Any]:
        with self._lock:
            load_error = self._model_error or reason
        return {
            "available": False,
            "reason": "embedding_model_unavailable",
            "message": load_error,
            "model_name": self.model_name,
            "recommendations": [],
            "seed_count": self._seed_count,
            "threshold": self._threshold,
        }

    def _vectors_for_rows(self, rows: list[dict], texts: list[str]) -> list[list[float]]:
        missing_indexes = []
        vectors: list[list[float] | None] = []
        with self._lock:
            for row, snippet_text in zip(rows, texts):
                key = self._row_cache_key(row, snippet_text)
                cached = self._snippet_vectors.get(key)
                vectors.append(cached)
                if cached is None:
                    missing_indexes.append(len(vectors) - 1)

        if missing_indexes:
            missing_texts = [texts[index] for index in missing_indexes]
            encoded = self._encode(missing_texts)
            with self._lock:
                for index, vector in zip(missing_indexes, encoded):
                    row = rows[index]
                    key = self._row_cache_key(row, texts[index])
                    self._snippet_vectors[key] = vector
                    vectors[index] = vector

        return [vector if vector is not None else [] for vector in vectors]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        encoded = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in encoded]

    def _load_model(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if self._model_error is not None:
                raise RuntimeError(self._model_error)

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
        except Exception as exc:
            with self._lock:
                self._model_error = (
                    "Could not load sentence-transformers/all-mpnet-base-v2. "
                    "Install dependencies and make sure the model can be downloaded. "
                    f"Original error: {exc}"
                )
            raise

        with self._lock:
            self._model = model
        return model

    def _load_tag_cache(self) -> None:
        if not self.tag_cache_path.exists():
            return
        with self.tag_cache_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("model_name") != self.model_name:
                    continue
                key = str(record.get("key", ""))
                embedding = record.get("embedding")
                if key and isinstance(embedding, list):
                    self._tag_vectors[key] = record

    def _write_tag_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.tag_cache_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(self._tag_vectors.values(), key=lambda item: item["key"]):
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(tmp_path, self.tag_cache_path)

    def _signature_for_records(self, records: list[tuple[str, str]]) -> str:
        digest = hashlib.sha256()
        for key, snippet_text in records:
            digest.update(key.encode("utf-8", errors="replace"))
            digest.update(b"\0")
            digest.update(text_hash(snippet_text).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _threshold_from_seed_scores(self, seed_scores: list[float]) -> float:
        if len(seed_scores) < 4:
            return 0.42
        mean = statistics.fmean(seed_scores)
        stdev = statistics.pstdev(seed_scores)
        return max(0.34, min(0.56, mean - (1.25 * stdev)))

    def _row_cache_key(self, row: dict, snippet_text: str) -> str:
        source_file = str(row.get("_source_file", ""))
        snippet_id = str(row.get("id", ""))
        return f"{source_file}::{snippet_id}::{text_hash(snippet_text)}"
