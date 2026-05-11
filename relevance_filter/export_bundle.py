from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_CACHE = ROOT / "annotations" / "embedding_cache" / "seed_vectors.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "relevance_bundles"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CLUSTER_THRESHOLD = 0.68

CATEGORY_LABELS = {
    "about_company": "About company / mission / culture",
    "compensation_benefits": "Compensation and benefits",
    "section_headers": "Section headers",
    "citizenship_clearance": "Citizenship, clearance, authorization",
    "job_title_restatement": "Job-title restatement / role label",
    "application_process": "Application / recruiter process",
    "equal_opportunity_legal": "EEO, legal, accommodation boilerplate",
    "generic_role_context": "Generic role context / team setup",
    "other": "Other or mixed",
}

KEYWORDS = {
    "compensation_benefits": ["compensation", "salary", "pay range", "benefits", "401", "bonus", "equity", "insurance", "pto", "retirement"],
    "citizenship_clearance": ["citizen", "citizenship", "clearance", "top secret", "public trust", "itar", "work authorization", "visa", "sponsorship"],
    "section_headers": ["requirements", "qualifications", "responsibilities", "nice to have", "preferred", "benefits", "about you", "experience", "skills"],
    "about_company": ["about", "company", "mission", "culture", "values", "global", "leader", "we are", "we're", "our mission", "our values", "platform"],
    "application_process": ["apply", "application", "recruiter", "interview", "contact you", "candidate", "applicant", "recruiting"],
    "equal_opportunity_legal": ["equal opportunity", "eoe", "disability", "reasonable accommodation", "protected veteran", "discrimination", "diversity"],
    "generic_role_context": ["collaborate", "stakeholders", "cross-functional", "team", "engineers", "researchers", "product groups", "fast-paced"],
}

STOPWORDS = {"the", "and", "for", "with", "that", "you", "your", "our", "are", "will", "this", "from", "have", "work", "role", "job", "company", "team"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a portable relevance scorer bundle.")
    parser.add_argument("--seed-cache", type=Path, default=DEFAULT_SEED_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--version", default=None, help="Bundle version name. Defaults to UTC timestamp.")
    parser.add_argument("--latest", action="store_true", help="Also copy this bundle to output-root/latest.")
    return parser.parse_args()


def load_seed_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Seed cache not found: {path}")
    records = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("label") in {"not_relevant", "reviewed_relevant"} and isinstance(record.get("embedding"), list):
                records.append(record)
    if not records:
        raise ValueError(f"No usable seed records found in {path}")
    return records


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": record.get("label"),
        "key": record.get("key"),
        "id": record.get("id"),
        "source_file": record.get("source_file"),
        "company": record.get("company"),
        "title": record.get("title"),
        "category": record.get("category"),
        "text": record.get("text"),
        "text_hash": record.get("text_hash"),
    }


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not norm:
        return vector
    return vector / norm


def build_clusters(seed_records: list[dict[str, Any]], seed_vectors: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    not_relevant = [
        (index, record)
        for index, record in enumerate(seed_records)
        if record.get("label") == "not_relevant"
    ]
    clusters: list[dict[str, Any]] = []
    for seed_index, record in sorted(not_relevant, key=lambda item: str(item[1].get("key", ""))):
        vector = seed_vectors[seed_index]
        best_index = -1
        best_score = -1.0
        for cluster_index, cluster in enumerate(clusters):
            score = float(cluster["centroid"] @ vector)
            if score > best_score:
                best_index = cluster_index
                best_score = score
        if best_index >= 0 and best_score >= CLUSTER_THRESHOLD:
            cluster = clusters[best_index]
            cluster["seed_indexes"].append(seed_index)
            cluster["centroid"] = normalize(seed_vectors[np.array(cluster["seed_indexes"])].mean(axis=0))
        else:
            clusters.append({"centroid": vector, "seed_indexes": [seed_index]})

    clusters = sorted(clusters, key=lambda item: (-len(item["seed_indexes"]), str(seed_records[item["seed_indexes"][0]].get("key", ""))))
    cluster_vectors = []
    cluster_records = []
    for cluster_number, cluster in enumerate(clusters, start=1):
        members = [seed_records[index] for index in cluster["seed_indexes"]]
        category = classify_cluster(members)
        center = cluster["centroid"]
        ranked_indexes = sorted(cluster["seed_indexes"], key=lambda index: float(seed_vectors[index] @ center), reverse=True)
        cluster_id = f"IR-{cluster_number:03d}"
        cluster_vectors.append(center)
        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "size": len(cluster["seed_indexes"]),
                "category": category,
                "category_label": CATEGORY_LABELS.get(category, category),
                "top_terms": top_terms(members),
                "seed_indexes": cluster["seed_indexes"],
                "exemplar": public_record(seed_records[ranked_indexes[0]]),
                "examples": [public_record(seed_records[index]) for index in ranked_indexes[:6]],
            }
        )
    matrix = np.asarray(cluster_vectors, dtype=np.float32) if cluster_vectors else np.empty((0, seed_vectors.shape[1]), dtype=np.float32)
    return cluster_records, matrix


def classify_cluster(members: list[dict[str, Any]]) -> str:
    joined = "\n".join(str(member.get("text", "")).lower() for member in members)
    scores = Counter()
    for category, words in KEYWORDS.items():
        for word in words:
            scores[category] += joined.count(word)
    short_count = sum(1 for member in members if len(tokenize(str(member.get("text", "")))) <= 7)
    if short_count >= max(2, len(members) * 0.45):
        scores["section_headers"] += short_count * 2
    title_restatement_count = sum(1 for member in members if title_overlap(member))
    if title_restatement_count >= max(2, len(members) * 0.35):
        scores["job_title_restatement"] += title_restatement_count * 3
    if not scores:
        return "other"
    category, score = scores.most_common(1)[0]
    return category if score > 1 else "other"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+\-']*", text.lower())


def title_overlap(member: dict[str, Any]) -> bool:
    title_tokens = set(tokenize(str(member.get("title", ""))))
    text_tokens = set(tokenize(str(member.get("text", ""))))
    if not title_tokens or not text_tokens:
        return False
    return len(title_tokens & text_tokens) / max(1, len(title_tokens)) >= 0.65 and len(text_tokens) <= len(title_tokens) + 5


def top_terms(members: list[dict[str, Any]], limit: int = 8) -> list[str]:
    counts = Counter()
    for member in members:
        for token in tokenize(str(member.get("text", ""))):
            if len(token) >= 3 and token not in STOPWORDS:
                counts[token] += 1
    return [term for term, _ in counts.most_common(limit)]


def write_bundle(records: list[dict[str, Any]], output_dir: Path, version: str, seed_cache: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_records = [public_record(record) for record in records]
    seed_vectors = np.asarray([record["embedding"] for record in records], dtype=np.float32)
    cluster_records, cluster_vectors = build_clusters(seed_records, seed_vectors)

    with (output_dir / "seed_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in seed_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    np.save(output_dir / "seed_vectors.npy", seed_vectors)
    np.save(output_dir / "cluster_vectors.npy", cluster_vectors)
    (output_dir / "clusters.json").write_text(
        json.dumps({"clusters": cluster_records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    thresholds = {
        "not_relevant_margin": 0.02,
        "reviewed_relevant_margin": 0.02,
        "min_not_relevant_score_without_relevant": 0.50,
    }
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "bundle_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "dimension": int(seed_vectors.shape[1]),
        "seed_count": len(seed_records),
        "not_relevant_seed_count": sum(1 for record in seed_records if record.get("label") == "not_relevant"),
        "reviewed_relevant_seed_count": sum(1 for record in seed_records if record.get("label") == "reviewed_relevant"),
        "cluster_count": len(cluster_records),
        "cluster_threshold": CLUSTER_THRESHOLD,
        "source_seed_cache": str(seed_cache),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    version = args.version or f"job_snippet_relevance_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = args.output_root / version
    records = load_seed_cache(args.seed_cache)
    write_bundle(records, output_dir, version, args.seed_cache)
    if args.latest:
        latest_dir = args.output_root / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(output_dir, latest_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
