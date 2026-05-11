# Portable Job Snippet Relevance Filter

This package exports the tagger's reviewed examples into a versioned scorer
bundle and scores new JSONL snippets before SOC/O*NET task matching.

The scorer is portable because the bundle contains the frozen seed examples,
seed vectors, cluster summaries, thresholds, and model name. The live tagging UI
and its annotation cache are not needed at scoring time.

Export a bundle from the tagging repo:

```powershell
python -m relevance_filter.export_bundle --latest
```

Score a JSONL file and keep every row:

```powershell
python -m relevance_filter.score_jsonl `
  --bundle relevance_bundles\latest `
  --input-jsonl sourceFiles\ai_linkedin_results_itemized_for_embeddings.jsonl `
  --output-jsonl scoring_feedback\ai_linkedin_relevance_scored.jsonl
```

Score and drop high-confidence not-relevant rows:

```powershell
python -m relevance_filter.score_jsonl `
  --bundle relevance_bundles\latest `
  --input-jsonl sourceFiles\ai_linkedin_results_itemized_for_embeddings.jsonl `
  --output-jsonl scoring_feedback\ai_linkedin_relevance_filtered.jsonl `
  --drop-not-relevant `
  --drop-confidence 0.75
```

Python API:

```python
from relevance_filter import RelevanceScorer

scorer = RelevanceScorer.from_bundle("relevance_bundles/latest")
rows = [{"id": "example", "text": "Compensation ranges will vary based on experience."}]
scored = scorer.score_records(rows)
print(scored[0]["relevance_filter"])
```

Labels are `not_relevant`, `relevant`, and `uncertain`. Downstream matchers can
either keep all rows and inspect `relevance_filter`, or pass a filtered JSONL
into the SOC/O*NET matching step.
