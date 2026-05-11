from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class RelevanceScorer:
    """Load a versioned relevance bundle and score job-posting snippets."""

    def __init__(self, bundle_dir: Path | str, device: str | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.device = device
        self.metadata = self._load_json("metadata.json")
        self.thresholds = self._load_json("thresholds.json")
        self.seed_records = self._load_jsonl("seed_records.jsonl")
        self.seed_vectors = self._load_matrix("seed_vectors.npy")
        self.cluster_records = self._load_json("clusters.json").get("clusters", [])
        self.cluster_vectors = self._load_optional_matrix("cluster_vectors.npy")
        self.model_name = str(self.metadata.get("model_name", DEFAULT_MODEL_NAME))
        self._model = None

        if len(self.seed_records) != len(self.seed_vectors):
            raise ValueError("Bundle seed_records.jsonl and seed_vectors.npy lengths do not match.")

        self.not_relevant_indexes = np.array(
            [index for index, record in enumerate(self.seed_records) if record.get("label") == "not_relevant"],
            dtype=np.int64,
        )
        self.reviewed_relevant_indexes = np.array(
            [
                index
                for index, record in enumerate(self.seed_records)
                if record.get("label") == "reviewed_relevant"
            ],
            dtype=np.int64,
        )

    @classmethod
    def from_bundle(cls, bundle_dir: Path | str, device: str | None = None) -> "RelevanceScorer":
        return cls(bundle_dir=bundle_dir, device=device)

    def score_records(
        self,
        records: list[dict[str, Any]],
        *,
        text_field: str = "text",
        id_field: str = "id",
        batch_size: int = 32,
        top_examples: int = 3,
    ) -> list[dict[str, Any]]:
        texts = [str(record.get(text_field, "")) for record in records]
        ids = [str(record.get(id_field, index)) for index, record in enumerate(records)]
        scores = self.score_texts(texts, ids=ids, batch_size=batch_size, top_examples=top_examples)
        output = []
        for record, score in zip(records, scores):
            item = dict(record)
            item["relevance_filter"] = score
            output.append(item)
        return output

    def score_texts(
        self,
        texts: list[str],
        *,
        ids: Iterable[str] | None = None,
        batch_size: int = 32,
        top_examples: int = 3,
    ) -> list[dict[str, Any]]:
        if not texts:
            return []

        identifiers = list(ids) if ids is not None else [str(index) for index in range(len(texts))]
        query_vectors = self.encode_texts(texts, batch_size=batch_size)
        results = []
        for snippet_id, text, vector in zip(identifiers, texts, query_vectors):
            results.append(self.score_vector(snippet_id=str(snippet_id), text=text, vector=vector, top_examples=top_examples))
        return results

    def score_vector(
        self,
        *,
        snippet_id: str,
        text: str,
        vector: np.ndarray,
        top_examples: int = 3,
    ) -> dict[str, Any]:
        not_relevant_score, not_relevant_examples = self._nearest_examples(
            vector,
            self.not_relevant_indexes,
            top_examples,
        )
        reviewed_relevant_score, reviewed_relevant_examples = self._nearest_examples(
            vector,
            self.reviewed_relevant_indexes,
            top_examples,
        )
        label, confidence, margin = self._label_from_scores(not_relevant_score, reviewed_relevant_score)
        cluster = self._nearest_cluster(vector)

        return {
            "bundle_version": self.metadata.get("bundle_version"),
            "model_name": self.model_name,
            "id": snippet_id,
            "label": label,
            "confidence": round(confidence, 6),
            "margin": round(margin, 6) if margin is not None else None,
            "not_relevant_score": round(not_relevant_score, 6) if not_relevant_score is not None else None,
            "reviewed_relevant_score": round(reviewed_relevant_score, 6)
            if reviewed_relevant_score is not None
            else None,
            "irrelevance_cluster": cluster,
            "nearest_not_relevant_examples": not_relevant_examples,
            "nearest_reviewed_relevant_examples": reviewed_relevant_examples,
            "input_text": text,
        }

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _label_from_scores(
        self,
        not_relevant_score: float | None,
        reviewed_relevant_score: float | None,
    ) -> tuple[str, float, float | None]:
        not_relevant_margin = float(self.thresholds.get("not_relevant_margin", 0.02))
        reviewed_relevant_margin = float(self.thresholds.get("reviewed_relevant_margin", 0.02))
        min_not_relevant_score = float(self.thresholds.get("min_not_relevant_score_without_relevant", 0.50))

        if not_relevant_score is not None and reviewed_relevant_score is not None:
            margin = not_relevant_score - reviewed_relevant_score
            if margin >= not_relevant_margin:
                return "not_relevant", min(1.0, abs(margin) / max(not_relevant_margin, 1e-6)), margin
            if margin <= -reviewed_relevant_margin:
                return "relevant", min(1.0, abs(margin) / max(reviewed_relevant_margin, 1e-6)), margin
            return "uncertain", 1.0 - (abs(margin) / max(not_relevant_margin, reviewed_relevant_margin, 1e-6)), margin

        if not_relevant_score is not None:
            if not_relevant_score >= min_not_relevant_score:
                return "not_relevant", not_relevant_score, None
            return "uncertain", 1.0 - not_relevant_score, None

        if reviewed_relevant_score is not None:
            return "relevant", reviewed_relevant_score, None

        return "uncertain", 0.0, None

    def _nearest_examples(
        self,
        vector: np.ndarray,
        indexes: np.ndarray,
        limit: int,
    ) -> tuple[float | None, list[dict[str, Any]]]:
        if indexes.size == 0:
            return None, []

        matrix = self.seed_vectors[indexes]
        scores = matrix @ vector
        best_count = min(max(limit, 1), len(scores))
        top_local = np.argpartition(scores, -best_count)[-best_count:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        examples = []
        for local_index in top_local:
            seed_index = int(indexes[int(local_index)])
            examples.append(
                {
                    "score": round(float(scores[int(local_index)]), 6),
                    **self._public_seed_record(self.seed_records[seed_index]),
                }
            )
        return float(scores[int(top_local[0])]), examples

    def _nearest_cluster(self, vector: np.ndarray) -> dict[str, Any] | None:
        if self.cluster_vectors is None or len(self.cluster_records) == 0:
            return None
        scores = self.cluster_vectors @ vector
        best_index = int(np.argmax(scores))
        cluster = self.cluster_records[best_index]
        return {
            "cluster_id": cluster.get("cluster_id"),
            "score": round(float(scores[best_index]), 6),
            "size": cluster.get("size"),
            "category": cluster.get("category"),
            "category_label": cluster.get("category_label"),
            "exemplar": cluster.get("exemplar"),
        }

    def _public_seed_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "label": record.get("label"),
            "id": record.get("id"),
            "source_file": record.get("source_file"),
            "company": record.get("company"),
            "title": record.get("title"),
            "category": record.get("category"),
            "text": record.get("text"),
        }

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - runtime dependency guard
                raise RuntimeError("sentence-transformers is required to score new text.") from exc

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def _load_json(self, filename: str) -> dict[str, Any]:
        path = self.bundle_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing bundle file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self.bundle_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing bundle file: {path}")
        records = []
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _load_matrix(self, filename: str) -> np.ndarray:
        path = self.bundle_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing bundle file: {path}")
        return np.load(path).astype(np.float32)

    def _load_optional_matrix(self, filename: str) -> np.ndarray | None:
        path = self.bundle_dir / filename
        if not path.exists():
            return None
        return np.load(path).astype(np.float32)
