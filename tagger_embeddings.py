from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
NOT_RELEVANT_LABEL = "not_relevant"
REVIEWED_RELEVANT_LABEL = "reviewed_relevant"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class EmbeddingRecommender:
    def __init__(self, cache_dir: Path, model_name: str = MODEL_NAME) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.seed_cache_path = cache_dir / "seed_vectors.jsonl"
        self.legacy_not_relevant_cache_path = cache_dir / "not_relevant_vectors.jsonl"
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tagger-embeddings")
        self._refresh_future: Future | None = None
        self._pending_seed_sets: tuple[dict[str, dict], dict[str, dict]] | None = None
        self._model: Any | None = None
        self._model_error: str | None = None
        self._seed_vectors: dict[str, dict[str, Any]] = {}
        self._snippet_vectors: dict[str, list[float]] = {}
        self._clusters: list[dict[str, Any]] = []
        self._seed_counts = {NOT_RELEVANT_LABEL: 0, REVIEWED_RELEVANT_LABEL: 0}
        self._signature = ""
        self._load_seed_cache()

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._refresh_future is not None and not self._refresh_future.done()
            return {
                "model_name": self.model_name,
                "available": self._model is not None and self._model_error is None,
                "load_error": self._model_error,
                "not_relevant_seed_count": self._seed_counts[NOT_RELEVANT_LABEL],
                "reviewed_relevant_seed_count": self._seed_counts[REVIEWED_RELEVANT_LABEL],
                "seed_count": sum(self._seed_counts.values()),
                "irrelevance_cluster_count": len(self._clusters),
                "refresh_running": running,
                "cached_seed_vectors": len(self._seed_vectors),
                "cached_snippet_vectors": len(self._snippet_vectors),
            }

    def queue_refresh(
        self,
        not_relevant_annotations: dict[str, dict],
        reviewed_relevant_annotations: dict[str, dict],
    ) -> None:
        not_relevant_snapshot = dict(not_relevant_annotations)
        reviewed_relevant_snapshot = dict(reviewed_relevant_annotations)
        with self._lock:
            self._pending_seed_sets = (not_relevant_snapshot, reviewed_relevant_snapshot)
            if self._refresh_future is not None and not self._refresh_future.done():
                return
            self._refresh_future = self._executor.submit(self._refresh_pending)

    def _refresh_pending(self) -> dict[str, Any]:
        while True:
            with self._lock:
                snapshot = self._pending_seed_sets
                self._pending_seed_sets = None
            if snapshot is None:
                return self.status()
            self.refresh_seed_sets(*snapshot)
            with self._lock:
                if self._pending_seed_sets is None:
                    return self.status()

    def score_rows(
        self,
        not_relevant_annotations: dict[str, dict],
        reviewed_relevant_annotations: dict[str, dict],
        rows: list[dict],
    ) -> dict[str, Any]:
        try:
            self.refresh_seed_sets(not_relevant_annotations, reviewed_relevant_annotations)
        except Exception as exc:
            return self._unavailable_payload(str(exc))

        with self._lock:
            seed_counts = dict(self._seed_counts)
            clusters = list(self._clusters)
            not_relevant_vectors = self._vectors_for_label(NOT_RELEVANT_LABEL)
            reviewed_relevant_vectors = self._vectors_for_label(REVIEWED_RELEVANT_LABEL)

        if not not_relevant_vectors and not reviewed_relevant_vectors:
            return {
                "available": False,
                "reason": "no_seed_tags",
                "message": "No reviewed seed tags are available yet.",
                "model_name": self.model_name,
                "recommendations": [],
                **self._count_payload(seed_counts, clusters),
            }

        texts = [str(row.get("text", "")) for row in rows]
        vectors = self._vectors_for_rows(rows, texts)
        recommendations = []
        for row, vector in zip(rows, vectors):
            not_relevant_score, _ = self._nearest(vector, not_relevant_vectors)
            reviewed_relevant_score, _ = self._nearest(vector, reviewed_relevant_vectors)
            label, score, confidence = self._label_from_scores(
                not_relevant_score,
                reviewed_relevant_score,
                bool(not_relevant_vectors),
                bool(reviewed_relevant_vectors),
            )
            cluster = self._nearest_cluster(vector, clusters)
            recommendations.append(
                {
                    "id": row.get("id"),
                    "score": round(score, 4),
                    "confidence": round(confidence, 4),
                    "not_relevant_score": round(not_relevant_score, 4) if not_relevant_vectors else None,
                    "reviewed_relevant_score": round(reviewed_relevant_score, 4)
                    if reviewed_relevant_vectors
                    else None,
                    "margin": round(not_relevant_score - reviewed_relevant_score, 4)
                    if not_relevant_vectors and reviewed_relevant_vectors
                    else None,
                    "suggested_label": label,
                    "irrelevance_cluster": cluster,
                    "model_name": self.model_name,
                }
            )

        return {
            "available": True,
            "model_name": self.model_name,
            "recommendations": recommendations,
            **self._count_payload(seed_counts, clusters),
        }

    def cluster_summaries(
        self,
        not_relevant_annotations: dict[str, dict],
        reviewed_relevant_annotations: dict[str, dict],
    ) -> dict[str, Any]:
        try:
            self.refresh_seed_sets(not_relevant_annotations, reviewed_relevant_annotations)
        except Exception as exc:
            return self._unavailable_payload(str(exc))

        with self._lock:
            clusters = [
                {
                    "cluster_id": cluster["cluster_id"],
                    "size": cluster["size"],
                    "exemplar": cluster["exemplar"],
                    "examples": cluster["examples"],
                }
                for cluster in self._clusters
            ]
            seed_counts = dict(self._seed_counts)

        return {
            "available": True,
            "model_name": self.model_name,
            "clusters": clusters,
            **self._count_payload(seed_counts, clusters),
        }

    def refresh_seed_sets(
        self,
        not_relevant_annotations: dict[str, dict],
        reviewed_relevant_annotations: dict[str, dict],
    ) -> dict[str, Any]:
        seed_records = self._seed_records(not_relevant_annotations, reviewed_relevant_annotations)
        signature = self._signature_for_records(seed_records)
        with self._lock:
            if signature == self._signature and self._clusters:
                return self.status()

        active_cache_keys = set()
        missing = []
        cache_changed = False
        with self._lock:
            for record in seed_records:
                cache_key = self._seed_cache_key(record["label"], record["key"])
                active_cache_keys.add(cache_key)
                cached = self._seed_vectors.get(cache_key)
                if cached is None or cached.get("text_hash") != record["text_hash"]:
                    missing.append(record)
                else:
                    metadata = {key: value for key, value in record.items() if key != "embedding"}
                    if any(cached.get(key) != value for key, value in metadata.items()):
                        cached.update(metadata)
                        cache_changed = True

        if missing:
            vectors = self._encode([record["text"] for record in missing])
            with self._lock:
                for record, vector in zip(missing, vectors):
                    self._seed_vectors[self._seed_cache_key(record["label"], record["key"])] = {
                        **record,
                        "embedding": vector,
                        "model_name": self.model_name,
                    }
        cache_changed = cache_changed or bool(missing)
        with self._lock:
            for cache_key in list(self._seed_vectors):
                if cache_key not in active_cache_keys:
                    self._seed_vectors.pop(cache_key, None)
                    cache_changed = True
            if cache_changed:
                self._write_seed_cache()

            not_relevant_vector_records = [
                self._seed_vectors[self._seed_cache_key(record["label"], record["key"])]
                for record in seed_records
                if record["label"] == NOT_RELEVANT_LABEL
                and self._seed_cache_key(record["label"], record["key"]) in self._seed_vectors
            ]
            self._clusters = self._build_irrelevance_clusters(not_relevant_vector_records)
            self._seed_counts = {
                NOT_RELEVANT_LABEL: sum(1 for record in seed_records if record["label"] == NOT_RELEVANT_LABEL),
                REVIEWED_RELEVANT_LABEL: sum(
                    1 for record in seed_records if record["label"] == REVIEWED_RELEVANT_LABEL
                ),
            }
            self._signature = signature

        return self.status()

    def _seed_records(
        self,
        not_relevant_annotations: dict[str, dict],
        reviewed_relevant_annotations: dict[str, dict],
    ) -> list[dict[str, Any]]:
        records = []
        not_relevant_keys = set(not_relevant_annotations)
        for key, record in sorted(not_relevant_annotations.items()):
            snippet_text = str(record.get("text", "")).strip()
            if snippet_text:
                records.append(self._seed_record(NOT_RELEVANT_LABEL, key, record, snippet_text))

        for key, record in sorted(reviewed_relevant_annotations.items()):
            if key in not_relevant_keys:
                continue
            snippet_text = str(record.get("text", "")).strip()
            if snippet_text:
                records.append(self._seed_record(REVIEWED_RELEVANT_LABEL, key, record, snippet_text))
        return records

    def _seed_record(self, label: str, key: str, record: dict, snippet_text: str) -> dict[str, Any]:
        return {
            "label": label,
            "key": key,
            "source_file": record.get("source_file"),
            "id": record.get("id"),
            "job_key": record.get("job_key"),
            "job_name": record.get("job_name"),
            "company": record.get("company"),
            "title": record.get("title"),
            "category": record.get("category"),
            "text": snippet_text,
            "text_hash": text_hash(snippet_text),
        }

    def _label_from_scores(
        self,
        not_relevant_score: float,
        reviewed_relevant_score: float,
        has_not_relevant: bool,
        has_reviewed_relevant: bool,
    ) -> tuple[str, float, float]:
        if has_not_relevant and has_reviewed_relevant:
            margin = not_relevant_score - reviewed_relevant_score
            if margin >= 0.015:
                return NOT_RELEVANT_LABEL, not_relevant_score, abs(margin)
            return "relevant", reviewed_relevant_score, abs(margin)
        if has_not_relevant:
            if not_relevant_score >= 0.48:
                return NOT_RELEVANT_LABEL, not_relevant_score, not_relevant_score
            return "relevant", not_relevant_score, 1.0 - not_relevant_score
        return "relevant", reviewed_relevant_score, reviewed_relevant_score

    def _build_irrelevance_clusters(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        cluster_threshold = 0.68

        for record in sorted(records, key=lambda item: str(item["key"])):
            vector = record["embedding"]
            best_index = -1
            best_score = -1.0
            for index, cluster in enumerate(clusters):
                score = cosine(vector, cluster["centroid"])
                if score > best_score:
                    best_score = score
                    best_index = index

            if best_index >= 0 and best_score >= cluster_threshold:
                cluster = clusters[best_index]
                cluster["members"].append(record)
                cluster["centroid"] = self._centroid([member["embedding"] for member in cluster["members"]])
            else:
                clusters.append({"centroid": vector, "members": [record]})

        summaries = []
        for cluster in sorted(clusters, key=lambda item: (-len(item["members"]), str(item["members"][0]["key"]))):
            centroid = cluster["centroid"]
            members = sorted(
                cluster["members"],
                key=lambda member: cosine(member["embedding"], centroid),
                reverse=True,
            )
            summaries.append(
                {
                    "centroid": centroid,
                    "size": len(members),
                    "members": members,
                    "exemplar": self._public_example(members[0]),
                    "examples": [self._public_example(member) for member in members[:4]],
                }
            )

        for index, cluster in enumerate(summaries, start=1):
            cluster["cluster_id"] = f"IR-{index:03d}"
        return summaries

    def _nearest_cluster(self, vector: list[float], clusters: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not vector or not clusters:
            return None
        best_cluster = None
        best_score = -1.0
        for cluster in clusters:
            score = cosine(vector, cluster["centroid"])
            if score > best_score:
                best_score = score
                best_cluster = cluster
        if best_cluster is None:
            return None
        return {
            "cluster_id": best_cluster["cluster_id"],
            "score": round(best_score, 4),
            "size": best_cluster["size"],
            "exemplar": best_cluster["exemplar"],
        }

    def _public_example(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record.get("id"),
            "source_file": record.get("source_file"),
            "company": record.get("company"),
            "title": record.get("title"),
            "category": record.get("category"),
            "text": record.get("text"),
        }

    def _centroid(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dimensions = len(vectors[0])
        sums = [0.0] * dimensions
        for vector in vectors:
            for index, value in enumerate(vector):
                sums[index] += float(value)
        return normalize_vector([value / len(vectors) for value in sums])

    def _nearest(self, vector: list[float], records: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
        best_score = 0.0
        best_record = None
        for record in records:
            score = cosine(vector, record["embedding"])
            if score > best_score:
                best_score = score
                best_record = record
        return best_score, best_record

    def _vectors_for_label(self, label: str) -> list[dict[str, Any]]:
        return [record for record in self._seed_vectors.values() if record.get("label") == label]

    def _unavailable_payload(self, reason: str) -> dict[str, Any]:
        with self._lock:
            load_error = self._model_error or reason
            seed_counts = dict(self._seed_counts)
            clusters = list(self._clusters)
        return {
            "available": False,
            "reason": "embedding_model_unavailable",
            "message": load_error,
            "model_name": self.model_name,
            "recommendations": [],
            **self._count_payload(seed_counts, clusters),
        }

    def _count_payload(self, seed_counts: dict[str, int], clusters: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "not_relevant_seed_count": seed_counts.get(NOT_RELEVANT_LABEL, 0),
            "reviewed_relevant_seed_count": seed_counts.get(REVIEWED_RELEVANT_LABEL, 0),
            "seed_count": sum(seed_counts.values()),
            "irrelevance_cluster_count": len(clusters),
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

    def _load_seed_cache(self) -> None:
        if self.seed_cache_path.exists():
            self._load_cache_file(self.seed_cache_path)
            return
        if self.legacy_not_relevant_cache_path.exists():
            self._load_cache_file(self.legacy_not_relevant_cache_path, legacy_label=NOT_RELEVANT_LABEL)

    def _load_cache_file(self, path: Path, legacy_label: str | None = None) -> None:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
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
                embedding = record.get("embedding")
                key = str(record.get("key", ""))
                if not key or not isinstance(embedding, list):
                    continue
                label = str(record.get("label") or legacy_label or "")
                if label not in {NOT_RELEVANT_LABEL, REVIEWED_RELEVANT_LABEL}:
                    continue
                record["label"] = label
                self._seed_vectors[self._seed_cache_key(label, key)] = record

    def _write_seed_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.seed_cache_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(
                self._seed_vectors.values(),
                key=lambda item: (str(item.get("label", "")), str(item.get("key", ""))),
            ):
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(tmp_path, self.seed_cache_path)

    def _signature_for_records(self, records: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for record in records:
            digest.update(str(record["label"]).encode("utf-8", errors="replace"))
            digest.update(b"\0")
            digest.update(str(record["key"]).encode("utf-8", errors="replace"))
            digest.update(b"\0")
            digest.update(str(record["text_hash"]).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _seed_cache_key(self, label: str, key: str) -> str:
        return f"{label}::{key}"

    def _row_cache_key(self, row: dict, snippet_text: str) -> str:
        source_file = str(row.get("_source_file", ""))
        snippet_id = str(row.get("id", ""))
        return f"{source_file}::{snippet_id}::{text_hash(snippet_text)}"
