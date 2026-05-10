from __future__ import annotations

import argparse
import json
import mimetypes
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, unquote, urlparse

from tagger_embeddings import EmbeddingRecommender


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sourceFiles"
APP_DIR = ROOT / "tagger_app"
ANNOTATION_DIR = ROOT / "annotations"
ANNOTATION_FILE = ANNOTATION_DIR / "not_relevant_tags.jsonl"
REVIEWED_RELEVANT_FILE = ANNOTATION_DIR / "reviewed_relevant_tags.jsonl"

ANNOTATION_LOCK = RLock()
RECOMMENDER = EmbeddingRecommender(ANNOTATION_DIR / "embedding_cache")


def available_source_files() -> dict[str, Path]:
    if not SOURCE_DIR.exists():
        return {}
    return {path.name: path for path in sorted(SOURCE_DIR.glob("*.jsonl"))}


def source_file_or_404(name: str) -> Path | None:
    return available_source_files().get(name)


def count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return sum(1 for _ in handle)


def annotation_key(source_file: str, snippet_id: str) -> str:
    return f"{source_file}::{snippet_id}"


def read_tag_annotations(path: Path, required_field: str) -> dict[str, dict]:
    annotations: dict[str, dict] = {}
    if not path.exists():
        return annotations

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            source_file = str(record.get("source_file", ""))
            snippet_id = str(record.get("id", ""))
            if source_file and snippet_id and record.get(required_field) is True:
                annotations[annotation_key(source_file, snippet_id)] = record

    return annotations


def read_annotations() -> dict[str, dict]:
    return read_tag_annotations(ANNOTATION_FILE, "not_relevant")


def read_reviewed_relevant_annotations() -> dict[str, dict]:
    return read_tag_annotations(REVIEWED_RELEVANT_FILE, "reviewed_relevant")


def write_tag_annotations(path: Path, annotations: dict[str, dict]) -> None:
    ANNOTATION_DIR.mkdir(exist_ok=True)
    tmp_path = path.with_suffix(".jsonl.tmp")
    records = sorted(
        annotations.values(),
        key=lambda item: (str(item.get("source_file", "")), str(item.get("id", ""))),
    )

    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    os.replace(tmp_path, path)


def write_annotations(annotations: dict[str, dict]) -> None:
    write_tag_annotations(ANNOTATION_FILE, annotations)


def write_reviewed_relevant_annotations(annotations: dict[str, dict]) -> None:
    write_tag_annotations(REVIEWED_RELEVANT_FILE, annotations)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length) if length else b"{}"
    return json.loads(body.decode("utf-8"))


def posting_key_for_row(row: dict, line_number: int) -> str:
    return str(
        row.get("job_key")
        or row.get("job_id")
        or row.get("job_name")
        or row.get("source_file")
        or f"posting-{line_number}"
    )


def read_source_rows(
    source_file: str,
    annotations: dict[str, dict] | None = None,
    reviewed_relevant_annotations: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    path = source_file_or_404(source_file)
    if path is None:
        return [], [{"line": 0, "error": "Unknown source file"}]

    if annotations is None:
        annotations = {}
    if reviewed_relevant_annotations is None:
        reviewed_relevant_annotations = {}

    rows = []
    errors = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue

            snippet_id = str(row.get("id", f"line-{line_number}"))
            key = annotation_key(source_file, snippet_id)
            row["_source_file"] = source_file
            row["_line_number"] = line_number
            row["_posting_key"] = posting_key_for_row(row, line_number)
            row["_annotation_key"] = key
            row["_not_relevant"] = key in annotations
            row["_reviewed_relevant"] = key in reviewed_relevant_annotations and key not in annotations
            row["_reviewed"] = row["_not_relevant"] or row["_reviewed_relevant"]
            rows.append(row)

    return rows, errors


class TaggerHandler(BaseHTTPRequestHandler):
    server_version = "JobSnippetTagger/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/files":
            self.handle_files()
            return
        if parsed.path == "/api/snippets":
            query = parse_qs(parsed.query)
            self.handle_snippets(query.get("file", [""])[0])
            return
        if parsed.path == "/api/annotations":
            self.handle_annotations()
            return
        if parsed.path == "/api/recommender-status":
            self.handle_recommender_status()
            return
        if parsed.path == "/api/recommendations":
            query = parse_qs(parsed.query)
            self.handle_recommendations(
                query.get("file", [""])[0],
                query.get("posting_key", [""])[0],
            )
            return
        if parsed.path == "/api/irrelevance-clusters":
            self.handle_irrelevance_clusters()
            return
        self.handle_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/tags":
            self.handle_tag_update()
            return
        if parsed.path == "/api/review-posting":
            self.handle_posting_review()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: dict | list, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_files(self) -> None:
        files = []
        for name, path in available_source_files().items():
            files.append({"name": name, "line_count": count_jsonl_lines(path)})
        self.send_json(
            {
                "files": files,
                "annotation_file": str(ANNOTATION_FILE.relative_to(ROOT)),
                "reviewed_relevant_file": str(REVIEWED_RELEVANT_FILE.relative_to(ROOT)),
            }
        )

    def handle_snippets(self, source_file: str) -> None:
        path = source_file_or_404(source_file)
        if path is None:
            self.send_json({"error": "Unknown source file"}, HTTPStatus.NOT_FOUND)
            return

        with ANNOTATION_LOCK:
            annotations = read_annotations()
            reviewed_relevant_annotations = read_reviewed_relevant_annotations()

        rows, errors = read_source_rows(source_file, annotations, reviewed_relevant_annotations)
        self.send_json({"source_file": source_file, "rows": rows, "errors": errors})

    def handle_annotations(self) -> None:
        with ANNOTATION_LOCK:
            annotations = read_annotations()
            reviewed_relevant_annotations = read_reviewed_relevant_annotations()
        self.send_json(
            {
                "annotations": list(annotations.values()),
                "count": len(annotations),
                "reviewed_relevant_annotations": list(reviewed_relevant_annotations.values()),
                "reviewed_relevant_count": len(reviewed_relevant_annotations),
            }
        )

    def handle_recommender_status(self) -> None:
        self.send_json(RECOMMENDER.status())

    def handle_recommendations(self, source_file: str, posting_key: str) -> None:
        if source_file_or_404(source_file) is None or not posting_key:
            self.send_json({"error": "Invalid source_file or posting_key"}, HTTPStatus.BAD_REQUEST)
            return

        with ANNOTATION_LOCK:
            annotations = read_annotations()
            reviewed_relevant_annotations = read_reviewed_relevant_annotations()
        rows, errors = read_source_rows(source_file, annotations, reviewed_relevant_annotations)
        posting_rows = [row for row in rows if row.get("_posting_key") == posting_key]
        payload = RECOMMENDER.score_rows(annotations, reviewed_relevant_annotations, posting_rows)
        payload.update(
            {
                "source_file": source_file,
                "posting_key": posting_key,
                "rows_scored": len(posting_rows),
                "errors": errors,
            }
        )
        self.send_json(payload)

    def handle_irrelevance_clusters(self) -> None:
        with ANNOTATION_LOCK:
            annotations = read_annotations()
            reviewed_relevant_annotations = read_reviewed_relevant_annotations()
        self.send_json(RECOMMENDER.cluster_summaries(annotations, reviewed_relevant_annotations))

    def handle_tag_update(self) -> None:
        try:
            payload = read_json_body(self)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
            return

        source_file = str(payload.get("source_file", ""))
        snippet_id = str(payload.get("id", ""))
        not_relevant = bool(payload.get("not_relevant", False))
        if source_file_or_404(source_file) is None or not snippet_id:
            self.send_json({"error": "Invalid source_file or id"}, HTTPStatus.BAD_REQUEST)
            return

        key = annotation_key(source_file, snippet_id)
        now = datetime.now(timezone.utc).isoformat()

        annotations_snapshot: dict[str, dict]
        reviewed_relevant_snapshot: dict[str, dict]
        with ANNOTATION_LOCK:
            annotations = read_annotations()
            reviewed_relevant_annotations = read_reviewed_relevant_annotations()
            if not_relevant:
                annotations[key] = {
                    "source_file": source_file,
                    "id": snippet_id,
                    "tag": "not_relevant",
                    "not_relevant": True,
                    "updated_at": now,
                    "job_key": payload.get("job_key"),
                    "job_name": payload.get("job_name"),
                    "company": payload.get("company"),
                    "title": payload.get("title"),
                    "category": payload.get("category"),
                    "text": payload.get("text"),
                }
                reviewed_relevant_annotations.pop(key, None)
            else:
                annotations.pop(key, None)
            write_annotations(annotations)
            write_reviewed_relevant_annotations(reviewed_relevant_annotations)
            annotations_snapshot = dict(annotations)
            reviewed_relevant_snapshot = dict(reviewed_relevant_annotations)

        RECOMMENDER.queue_refresh(annotations_snapshot, reviewed_relevant_snapshot)

        self.send_json(
            {
                "ok": True,
                "not_relevant": not_relevant,
                "count": len(annotations),
                "reviewed_relevant_count": len(reviewed_relevant_snapshot),
            }
        )

    def handle_posting_review(self) -> None:
        try:
            payload = read_json_body(self)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
            return

        source_file = str(payload.get("source_file", ""))
        posting_key = str(payload.get("posting_key", ""))
        if source_file_or_404(source_file) is None or not posting_key:
            self.send_json({"error": "Invalid source_file or posting_key"}, HTTPStatus.BAD_REQUEST)
            return

        now = datetime.now(timezone.utc).isoformat()
        with ANNOTATION_LOCK:
            annotations = read_annotations()
            reviewed_relevant_annotations = read_reviewed_relevant_annotations()
            rows, errors = read_source_rows(source_file, annotations, reviewed_relevant_annotations)
            posting_rows = [row for row in rows if row.get("_posting_key") == posting_key]
            marked_relevant = 0
            skipped_not_relevant = 0

            for row in posting_rows:
                snippet_id = str(row.get("id", ""))
                if not snippet_id:
                    continue
                key = annotation_key(source_file, snippet_id)
                if key in annotations:
                    reviewed_relevant_annotations.pop(key, None)
                    skipped_not_relevant += 1
                    continue

                reviewed_relevant_annotations[key] = {
                    "source_file": source_file,
                    "id": snippet_id,
                    "tag": "reviewed_relevant",
                    "reviewed_relevant": True,
                    "updated_at": now,
                    "job_key": row.get("job_key"),
                    "job_name": row.get("job_name"),
                    "company": row.get("company"),
                    "title": row.get("title"),
                    "category": row.get("category"),
                    "text": row.get("text"),
                }
                marked_relevant += 1

            write_reviewed_relevant_annotations(reviewed_relevant_annotations)
            annotations_snapshot = dict(annotations)
            reviewed_relevant_snapshot = dict(reviewed_relevant_annotations)

        RECOMMENDER.queue_refresh(annotations_snapshot, reviewed_relevant_snapshot)
        self.send_json(
            {
                "ok": True,
                "source_file": source_file,
                "posting_key": posting_key,
                "rows_reviewed": len(posting_rows),
                "marked_relevant": marked_relevant,
                "skipped_not_relevant": skipped_not_relevant,
                "not_relevant_count": len(annotations_snapshot),
                "reviewed_relevant_count": len(reviewed_relevant_snapshot),
                "errors": errors,
            }
        )

    def handle_static(self, request_path: str) -> None:
        relative = unquote(request_path).lstrip("/")
        if not relative:
            relative = "index.html"

        path = (APP_DIR / relative).resolve()
        if APP_DIR.resolve() not in path.parents and path != APP_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not path.exists() or path.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_type, _ = mimetypes.guess_type(path.name)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local job snippet tagging UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        raise SystemExit(f"Missing source directory: {SOURCE_DIR}")
    if not APP_DIR.exists():
        raise SystemExit(f"Missing app directory: {APP_DIR}")

    server = ThreadingHTTPServer((args.host, args.port), TaggerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Job snippet tagger running at {url}")
    print(f"Annotations will be saved to {ANNOTATION_FILE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping tagger.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
