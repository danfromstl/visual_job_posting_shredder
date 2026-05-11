from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .scorer import RelevanceScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score job-posting JSONL rows for relevance.")
    parser.add_argument("--bundle", type=Path, required=True, help="Path to a relevance scorer bundle directory.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128, help="Rows to score and flush at a time.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-examples", type=int, default=3)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N rows.")
    parser.add_argument(
        "--drop-not-relevant",
        action="store_true",
        help="Only write rows whose relevance label is not 'not_relevant'.",
    )
    parser.add_argument(
        "--drop-confidence",
        type=float,
        default=0.0,
        help="When dropping, require label=not_relevant and confidence >= this value.",
    )
    return parser.parse_args()


def iter_jsonl(
    path: Path,
    *,
    text_field: str = "text",
    id_field: str = "id",
    limit: int | None = None,
) -> Iterable[dict]:
    yielded = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get(id_field) or not row.get(text_field):
                raise ValueError(f"Line {line_number}: missing {id_field!r} or {text_field!r}")
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def should_write_row(row: dict, *, drop_not_relevant: bool, drop_confidence: float) -> bool:
    if not drop_not_relevant:
        return True
    relevance = row.get("relevance_filter", {})
    return not (
        relevance.get("label") == "not_relevant"
        and float(relevance.get("confidence", 0.0)) >= drop_confidence
    )


def write_rows(handle, rows: list[dict], *, drop_not_relevant: bool, drop_confidence: float) -> int:
    written = 0
    for row in rows:
        if not should_write_row(
            row,
            drop_not_relevant=drop_not_relevant,
            drop_confidence=drop_confidence,
        ):
            continue
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        written += 1
    handle.flush()
    return written


def score_jsonl_file(
    *,
    bundle: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    batch_size: int = 32,
    chunk_size: int = 128,
    device: str | None = None,
    top_examples: int = 3,
    text_field: str = "text",
    id_field: str = "id",
    limit: int | None = None,
    drop_not_relevant: bool = False,
    drop_confidence: float = 0.0,
) -> tuple[int, int]:
    scorer = RelevanceScorer.from_bundle(bundle, device=device)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    buffer: list[dict] = []
    chunk_size = max(1, chunk_size)

    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in iter_jsonl(
            input_jsonl,
            text_field=text_field,
            id_field=id_field,
            limit=limit,
        ):
            buffer.append(row)
            if len(buffer) < chunk_size:
                continue

            scored = scorer.score_records(
                buffer,
                text_field=text_field,
                id_field=id_field,
                batch_size=batch_size,
                top_examples=top_examples,
            )
            processed += len(buffer)
            written += write_rows(
                handle,
                scored,
                drop_not_relevant=drop_not_relevant,
                drop_confidence=drop_confidence,
            )
            print(f"Scored {processed} rows; wrote {written} rows...", flush=True)
            buffer = []

        if buffer:
            scored = scorer.score_records(
                buffer,
                text_field=text_field,
                id_field=id_field,
                batch_size=batch_size,
                top_examples=top_examples,
            )
            processed += len(buffer)
            written += write_rows(
                handle,
                scored,
                drop_not_relevant=drop_not_relevant,
                drop_confidence=drop_confidence,
            )
            print(f"Scored {processed} rows; wrote {written} rows...", flush=True)

    return processed, written


def main() -> int:
    args = parse_args()
    processed, written = score_jsonl_file(
        bundle=args.bundle,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=args.device,
        top_examples=args.top_examples,
        text_field=args.text_field,
        id_field=args.id_field,
        limit=args.limit,
        drop_not_relevant=args.drop_not_relevant,
        drop_confidence=args.drop_confidence,
    )
    print(f"Done. Scored {processed} rows; wrote {written} rows to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
