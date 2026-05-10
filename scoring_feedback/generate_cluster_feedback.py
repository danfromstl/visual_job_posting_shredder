from __future__ import annotations

import html
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scoring_feedback"
SEED_CACHE = ROOT / "annotations" / "embedding_cache" / "seed_vectors.jsonl"
NOT_RELEVANT_FILE = ROOT / "annotations" / "not_relevant_tags.jsonl"
REVIEWED_RELEVANT_FILE = ROOT / "annotations" / "reviewed_relevant_tags.jsonl"
CLUSTER_THRESHOLD = 0.68


CATEGORY_COLORS = {
    "about_company": "#2f6f73",
    "compensation_benefits": "#b84a3a",
    "section_headers": "#7c5fb5",
    "citizenship_clearance": "#9a7b1f",
    "job_title_restatement": "#3d6ea8",
    "application_process": "#6f7b35",
    "equal_opportunity_legal": "#8c5a3b",
    "generic_role_context": "#4f7d57",
    "other": "#777777",
}

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
    "compensation_benefits": [
        "compensation",
        "salary",
        "pay range",
        "base pay",
        "benefits",
        "401",
        "bonus",
        "stock",
        "equity",
        "insurance",
        "medical",
        "dental",
        "vision",
        "pto",
        "paid time",
        "vacation",
        "wellness",
        "retirement",
    ],
    "citizenship_clearance": [
        "citizen",
        "citizenship",
        "clearance",
        "security clearance",
        "top secret",
        "secret clearance",
        "public trust",
        "itar",
        "export control",
        "work authorization",
        "authorized to work",
        "visa",
        "sponsorship",
    ],
    "section_headers": [
        "requirements",
        "required",
        "qualifications",
        "responsibilities",
        "what you",
        "nice to have",
        "preferred",
        "benefits",
        "about you",
        "about the role",
        "experience",
        "skills",
        "education",
        "overview",
        "minimum",
    ],
    "about_company": [
        "about",
        "company",
        "mission",
        "culture",
        "values",
        "founded",
        "global",
        "leader",
        "we are",
        "we're",
        "our team",
        "our mission",
        "our values",
        "our company",
        "customers",
        "industry",
        "platform",
        "product-market",
    ],
    "application_process": [
        "apply",
        "application",
        "recruiter",
        "interview",
        "contact you",
        "candidate",
        "applicant",
        "hiring process",
        "talent acquisition",
        "recruiting",
    ],
    "equal_opportunity_legal": [
        "equal opportunity",
        "eoe",
        "disability",
        "reasonable accommodation",
        "protected veteran",
        "discrimination",
        "diversity",
        "inclusion",
        "background check",
        "drug screen",
    ],
    "generic_role_context": [
        "collaborate",
        "stakeholders",
        "cross-functional",
        "team",
        "engineers",
        "researchers",
        "product groups",
        "cutting-edge",
        "fast-paced",
        "environment",
    ],
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "you",
    "your",
    "our",
    "are",
    "will",
    "this",
    "from",
    "have",
    "has",
    "all",
    "not",
    "can",
    "may",
    "their",
    "they",
    "work",
    "role",
    "job",
    "company",
    "team",
    "experience",
    "skills",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def centroid(records: list[dict[str, Any]]) -> list[float]:
    if not records:
        return []
    dimensions = len(records[0]["embedding"])
    sums = [0.0] * dimensions
    for record in records:
        for index, value in enumerate(record["embedding"]):
            sums[index] += float(value)
    return normalize([value / len(records) for value in sums])


def build_clusters(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item.get("key", ""))):
        vector = record["embedding"]
        best_index = -1
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            score = cosine(vector, cluster["centroid"])
            if score > best_score:
                best_index = index
                best_score = score

        if best_index >= 0 and best_score >= CLUSTER_THRESHOLD:
            cluster = clusters[best_index]
            cluster["members"].append(record)
            cluster["centroid"] = centroid(cluster["members"])
        else:
            clusters.append({"centroid": vector, "members": [record]})

    clusters = sorted(clusters, key=lambda item: (-len(item["members"]), str(item["members"][0].get("key", ""))))
    for index, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = f"IR-{index:03d}"
        cluster["size"] = len(cluster["members"])
        cluster["category"] = classify_cluster(cluster)
        cluster["top_terms"] = top_terms(cluster["members"])
        cluster["examples"] = exemplar_members(cluster, 6)
    return clusters


def classify_cluster(cluster: dict[str, Any]) -> str:
    members = cluster["members"]
    texts = [str(member.get("text", "")) for member in members]
    joined = "\n".join(text.lower() for text in texts)
    scores = Counter()

    for category, words in KEYWORDS.items():
        for word in words:
            scores[category] += joined.count(word)

    short_count = sum(1 for text in texts if len(tokenize(text)) <= 7 and len(text) <= 85)
    titlelike_count = sum(1 for text in texts if looks_like_section_header(text))
    if short_count >= max(2, len(texts) * 0.45) or titlelike_count >= max(2, len(texts) * 0.35):
        scores["section_headers"] += max(3, titlelike_count * 3)

    title_restatement_count = sum(1 for member in members if title_overlap(member))
    if title_restatement_count >= max(2, len(members) * 0.35):
        scores["job_title_restatement"] += title_restatement_count * 3

    if not scores:
        return "other"
    category, score = scores.most_common(1)[0]
    if score <= 1:
        return "other"
    return category


def looks_like_section_header(text: str) -> bool:
    cleaned = text.strip().strip(":")
    if not cleaned:
        return False
    words = tokenize(cleaned)
    if len(words) > 8:
        return False
    lowered = cleaned.lower()
    if any(word in lowered for word in KEYWORDS["section_headers"]):
        return True
    letters = [char for char in cleaned if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio > 0.42


def title_overlap(member: dict[str, Any]) -> bool:
    title_tokens = set(tokenize(str(member.get("title", ""))))
    text_tokens = set(tokenize(str(member.get("text", ""))))
    if not title_tokens or not text_tokens:
        return False
    overlap = len(title_tokens & text_tokens) / max(1, len(title_tokens))
    return overlap >= 0.65 and len(text_tokens) <= len(title_tokens) + 5


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+\-']*", text.lower())


def top_terms(members: list[dict[str, Any]], limit: int = 8) -> list[str]:
    counts = Counter()
    for member in members:
        for token in tokenize(str(member.get("text", ""))):
            if len(token) < 3 or token in STOPWORDS:
                continue
            counts[token] += 1
    return [term for term, _ in counts.most_common(limit)]


def exemplar_members(cluster: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    center = cluster["centroid"]
    ranked = sorted(
        cluster["members"],
        key=lambda member: cosine(member["embedding"], center),
        reverse=True,
    )
    examples = []
    seen_texts = set()
    for member in ranked:
        normalized = " ".join(str(member.get("text", "")).lower().split())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        examples.append(public_record(member))
        if len(examples) >= limit:
            break
    if len(examples) < limit:
        for member in ranked:
            record = public_record(member)
            if record not in examples:
                examples.append(record)
            if len(examples) >= limit:
                break
    return examples


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "company": record.get("company"),
        "title": record.get("title"),
        "category": record.get("category"),
        "source_file": record.get("source_file"),
        "text": record.get("text"),
    }


def category_rollup(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rollup: dict[str, dict[str, Any]] = defaultdict(lambda: {"cluster_count": 0, "snippet_count": 0})
    for cluster in clusters:
        category = cluster["category"]
        rollup[category]["cluster_count"] += 1
        rollup[category]["snippet_count"] += cluster["size"]
    return [
        {
            "category": category,
            "label": CATEGORY_LABELS.get(category, category),
            **values,
        }
        for category, values in sorted(rollup.items(), key=lambda item: (-item[1]["snippet_count"], item[0]))
    ]


def pca_points(clusters: list[dict[str, Any]]) -> list[tuple[float, float]]:
    try:
        import numpy as np
    except Exception:
        count = max(1, len(clusters))
        return [
            (math.cos(index * 2 * math.pi / count), math.sin(index * 2 * math.pi / count))
            for index, _ in enumerate(clusters)
        ]

    matrix = np.array([cluster["centroid"] for cluster in clusters], dtype=float)
    matrix = matrix - matrix.mean(axis=0)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    projected = matrix @ vh[:2].T
    return [(float(row[0]), float(row[1])) for row in projected]


def scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if source_max == source_min:
        return (target_min + target_max) / 2
    return target_min + ((value - source_min) / (source_max - source_min)) * (target_max - target_min)


def write_cluster_size_svg(clusters: list[dict[str, Any]]) -> None:
    top = clusters[:24]
    width = 1200
    row_height = 32
    margin_left = 245
    margin_right = 60
    height = 70 + row_height * len(top)
    max_size = max(cluster["size"] for cluster in top) if top else 1
    lines = [
        svg_header(width, height),
        text_svg(24, 34, "Top irrelevance clusters by seed count", 22, "#22201d", "700"),
    ]
    for index, cluster in enumerate(top):
        y = 62 + index * row_height
        bar_width = scale(cluster["size"], 0, max_size, 8, width - margin_left - margin_right)
        color = CATEGORY_COLORS.get(cluster["category"], CATEGORY_COLORS["other"])
        label = f'{cluster["cluster_id"]} | {CATEGORY_LABELS.get(cluster["category"], cluster["category"])}'
        lines.append(text_svg(24, y + 18, label, 13, "#22201d", "700"))
        lines.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="22" rx="4" fill="{color}">'
            f"<title>{html.escape(cluster['cluster_id'])}: {cluster['size']} snippets</title></rect>"
        )
        lines.append(text_svg(margin_left + bar_width + 8, y + 17, str(cluster["size"]), 12, "#6b665c", "700"))
    lines.append("</svg>")
    (OUT_DIR / "cluster_size_bars.svg").write_text("\n".join(lines), encoding="utf-8")


def write_category_svg(rollup: list[dict[str, Any]]) -> None:
    width = 1100
    row_height = 38
    margin_left = 285
    height = 70 + row_height * len(rollup)
    max_count = max(item["snippet_count"] for item in rollup) if rollup else 1
    lines = [
        svg_header(width, height),
        text_svg(24, 34, "Heuristic category rollup", 22, "#22201d", "700"),
    ]
    for index, item in enumerate(rollup):
        y = 62 + index * row_height
        category = item["category"]
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"])
        bar_width = scale(item["snippet_count"], 0, max_count, 8, width - margin_left - 80)
        label = item["label"]
        lines.append(text_svg(24, y + 18, label, 13, "#22201d", "700"))
        lines.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="24" rx="4" fill="{color}" />')
        lines.append(
            text_svg(
                margin_left + bar_width + 8,
                y + 18,
                f"{item['snippet_count']} snippets / {item['cluster_count']} clusters",
                12,
                "#6b665c",
                "700",
            )
        )
    lines.append("</svg>")
    (OUT_DIR / "category_rollup.svg").write_text("\n".join(lines), encoding="utf-8")


def write_cluster_map_svg(clusters: list[dict[str, Any]]) -> None:
    shown = clusters[:90]
    points = pca_points(shown)
    xs = [point[0] for point in points] or [0.0]
    ys = [point[1] for point in points] or [0.0]
    width = 1100
    height = 780
    plot_left = 70
    plot_top = 80
    plot_width = 820
    plot_height = 620
    lines = [
        svg_header(width, height),
        text_svg(24, 36, "Cluster map from MPNet embedding space", 22, "#22201d", "700"),
        text_svg(24, 58, "PCA projection of the 90 largest irrelevance clusters; bubble size = cluster size.", 13, "#6b665c", "400"),
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#fffefa" stroke="#d9d4c8" />',
    ]
    for cluster, (x_value, y_value) in zip(shown, points):
        x = scale(x_value, min(xs), max(xs), plot_left + 24, plot_left + plot_width - 24)
        y = scale(y_value, min(ys), max(ys), plot_top + plot_height - 24, plot_top + 24)
        radius = max(4.5, min(23, math.sqrt(cluster["size"]) * 3.2))
        color = CATEGORY_COLORS.get(cluster["category"], CATEGORY_COLORS["other"])
        title = f"{cluster['cluster_id']} | {cluster['size']} | {CATEGORY_LABELS.get(cluster['category'], cluster['category'])}"
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.78" '
            f'stroke="#22201d" stroke-opacity="0.22"><title>{html.escape(title)}</title></circle>'
        )
        if cluster["size"] >= 8:
            lines.append(text_svg(x + radius + 3, y + 4, cluster["cluster_id"], 11, "#22201d", "700"))

    legend_x = 925
    legend_y = 96
    lines.append(text_svg(legend_x, legend_y - 26, "Legend", 16, "#22201d", "700"))
    for index, (category, label) in enumerate(CATEGORY_LABELS.items()):
        y = legend_y + index * 28
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"])
        lines.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" rx="3" fill="{color}" />')
        lines.append(text_svg(legend_x + 24, y + 1, label, 12, "#22201d", "400"))
    lines.append("</svg>")
    (OUT_DIR / "cluster_map.svg").write_text("\n".join(lines), encoding="utf-8")


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#f6f4ef" />'
    )


def text_svg(x: float, y: float, value: str, size: int, color: str, weight: str) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(value)}</text>'
    )


def write_html_report(summary: dict[str, Any]) -> None:
    top_clusters = summary["clusters"][:20]
    category_rows = "\n".join(
        f"<tr><td>{html.escape(item['label'])}</td><td>{item['snippet_count']}</td><td>{item['cluster_count']}</td></tr>"
        for item in summary["category_rollup"]
    )
    cluster_sections = []
    for cluster in summary["clusters"][:3]:
        examples = "\n".join(f"<li>{html.escape(example['text'] or '')}</li>" for example in cluster["examples"])
        cluster_sections.append(
            f"""
            <section>
              <h2>{cluster['cluster_id']} - {html.escape(CATEGORY_LABELS.get(cluster['category'], cluster['category']))}</h2>
              <p><strong>{cluster['size']}</strong> snippets. Top terms: {html.escape(', '.join(cluster['top_terms']))}</p>
              <ol>{examples}</ol>
            </section>
            """
        )
    cluster_rows = "\n".join(
        f"<tr><td>{cluster['cluster_id']}</td><td>{cluster['size']}</td><td>{html.escape(CATEGORY_LABELS.get(cluster['category'], cluster['category']))}</td><td>{html.escape(', '.join(cluster['top_terms']))}</td></tr>"
        for cluster in top_clusters
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Irrelevance Cluster Feedback</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 28px; background: #f6f4ef; color: #22201d; line-height: 1.45; }}
    h1, h2 {{ letter-spacing: 0; }}
    img {{ max-width: 100%; border: 1px solid #d9d4c8; background: #fffdf8; margin: 12px 0 24px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fffdf8; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d9d4c8; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eee8da; }}
    section {{ margin: 26px 0; }}
  </style>
</head>
<body>
  <h1>Irrelevance Cluster Feedback</h1>
  <p>Generated {html.escape(summary['generated_at'])}. Based on {summary['not_relevant_seed_count']} not-relevant seeds and {summary['reviewed_relevant_seed_count']} reviewed-relevant seeds.</p>
  <img src="cluster_size_bars.svg" alt="Top cluster sizes">
  <img src="category_rollup.svg" alt="Category rollup">
  <img src="cluster_map.svg" alt="Cluster map">
  <h2>Category Rollup</h2>
  <table><tr><th>Category</th><th>Snippets</th><th>Clusters</th></tr>{category_rows}</table>
  <h2>Top 20 Clusters</h2>
  <table><tr><th>Cluster</th><th>Size</th><th>Heuristic label</th><th>Top terms</th></tr>{cluster_rows}</table>
  <h2>Largest Cluster Examples</h2>
  {''.join(cluster_sections)}
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(content, encoding="utf-8")


def write_markdown_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Irrelevance Cluster Feedback",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"- Not-relevant seeds: {summary['not_relevant_seed_count']}",
        f"- Reviewed-relevant seeds: {summary['reviewed_relevant_seed_count']}",
        f"- Irrelevance clusters: {summary['cluster_count']}",
        f"- Clustering threshold: cosine >= {CLUSTER_THRESHOLD}",
        "",
        "## Visuals",
        "",
        "- `cluster_size_bars.svg`: top irrelevance clusters by seed count",
        "- `category_rollup.svg`: heuristic category rollup",
        "- `cluster_map.svg`: PCA map of the largest clusters",
        "- `index.html`: browser-friendly report that embeds all visuals",
        "",
        "## Category Rollup",
        "",
        "| Category | Snippets | Clusters |",
        "| --- | ---: | ---: |",
    ]
    for item in summary["category_rollup"]:
        lines.append(f"| {item['label']} | {item['snippet_count']} | {item['cluster_count']} |")

    lines.extend(["", "## Top 20 Clusters", "", "| Cluster | Size | Heuristic label | Top terms |", "| --- | ---: | --- | --- |"])
    for cluster in summary["clusters"][:20]:
        lines.append(
            f"| {cluster['cluster_id']} | {cluster['size']} | {CATEGORY_LABELS.get(cluster['category'], cluster['category'])} | {', '.join(cluster['top_terms'])} |"
        )

    lines.extend(["", "## Examples From The 3 Largest Clusters"])
    for cluster in summary["clusters"][:3]:
        lines.extend(
            [
                "",
                f"### {cluster['cluster_id']} - {CATEGORY_LABELS.get(cluster['category'], cluster['category'])}",
                "",
                f"Size: {cluster['size']}",
                f"Top terms: {', '.join(cluster['top_terms'])}",
                "",
            ]
        )
        for index, example in enumerate(cluster["examples"], start=1):
            company = example.get("company") or "Unknown company"
            title = example.get("title") or "Unknown title"
            lines.append(f"{index}. {example.get('text', '')}  ")
            lines.append(f"   Source: {company} / {title}")

    lines.extend(
        [
            "",
            "## Read",
            "",
            "The labels above are intentionally provisional. They are keyword-assisted names over embedding clusters, not final ontology labels. The useful signal is which examples are drifting together and which clusters keep growing as you review more postings.",
        ]
    )
    (OUT_DIR / "cluster_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    seed_records = load_jsonl(SEED_CACHE)
    not_relevant_records = [
        record
        for record in seed_records
        if record.get("label") == "not_relevant" and isinstance(record.get("embedding"), list)
    ]
    clusters = build_clusters(not_relevant_records)
    rollup = category_rollup(clusters)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "not_relevant_seed_count": len(load_jsonl(NOT_RELEVANT_FILE)),
        "reviewed_relevant_seed_count": len(load_jsonl(REVIEWED_RELEVANT_FILE)),
        "cluster_count": len(clusters),
        "cluster_threshold": CLUSTER_THRESHOLD,
        "category_rollup": rollup,
        "clusters": [
            {
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
                "category": cluster["category"],
                "category_label": CATEGORY_LABELS.get(cluster["category"], cluster["category"]),
                "top_terms": cluster["top_terms"],
                "examples": cluster["examples"],
            }
            for cluster in clusters
        ],
    }
    (OUT_DIR / "cluster_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_cluster_size_svg(clusters)
    write_category_svg(rollup)
    write_cluster_map_svg(clusters)
    write_markdown_report(summary)
    write_html_report(summary)
    print(json.dumps({k: summary[k] for k in ["not_relevant_seed_count", "reviewed_relevant_seed_count", "cluster_count"]}, indent=2))


if __name__ == "__main__":
    main()
