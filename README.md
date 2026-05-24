# victoria-emotion-ds — Review Tool

Human review tool für den [victoria-emotion-de-synthetic](https://huggingface.co/datasets/dida-80b/victoria-emotion-de-synthetic) Datensatz.

## Setup

```bash
pip install fastapi uvicorn
git clone git@github.com:dida-80b/victoria-emotion-ds.git
cd victoria-emotion-ds
```

## Review starten

```bash
python3 tools/qc_review_server.py --reviewer <dein-name>
```

Browser: `http://127.0.0.1:8091`

**Shortcuts:** `K` Keep · `D` Delete · `R` Rerender · `N`/`→` Weiter · `B`/`←` Zurück · `Leertaste` Play/Pause

Wenn fertig: **Sync-Button** oben rechts drücken → pushed automatisch auf GitHub.

## Ergebnisse zusammenführen

```bash
python3 tools/qc_merge_reviews.py --state-dir reviews --output-dir merge --replace
```

Output: `merge/consensus_keep.jsonl`, `merge/consensus_delete.jsonl`, `merge/consensus_rerender.jsonl`, `merge/conflicts.jsonl`
