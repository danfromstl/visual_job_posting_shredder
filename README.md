# visual_job_posting_shredder
There's a lot of fluff and non-standard language, let standardize

## Local not-relevant tagger

Run the local review UI:

```powershell
python .\tagger_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

The UI reads JSONL files from `sourceFiles` and saves not-relevant tags to
`annotations/not_relevant_tags.jsonl`. Source JSONL files are not modified.
When you leave a posting, snippets in that posting that are not marked
not-relevant are saved as reviewed-relevant examples in
`annotations/reviewed_relevant_tags.jsonl`.

The recommendation scorer uses `sentence-transformers/all-mpnet-base-v2` and
the saved reviewed examples as seed data. It scores each new posting against
both not-relevant and reviewed-relevant embeddings, then shows the nearest
irrelevance cluster for each row. If `sentence-transformers` is not installed,
run this once:

```powershell
python -m pip install -r .\requirements.txt
```

The first recommendation pass can take a little while because the model may
need to download and load.

Useful keys:

- `R`: mark the selected row not relevant and move down
- `U`: clear the selected row and move down
- `Space`: toggle the selected row and move down
- `ArrowUp` / `ArrowDown`: move selection
- `ArrowLeft` / `ArrowRight`: previous or next posting
- `K` / `J`: move selection

Tag saves are optimistic: the selection advances immediately, and individual
rows show a small saving badge while the JSONL annotation write finishes. When
you move to another posting, the app briefly shows scoring progress while it
finishes queued saves, stores reviewed-relevant examples, and scores the next
posting.
