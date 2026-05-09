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

Useful keys:

- `R`: mark the selected row not relevant and move down
- `U`: clear the selected row and move down
- `Space`: toggle the selected row and move down
- `ArrowUp` / `ArrowDown`: move selection
- `K` / `J`: move selection
