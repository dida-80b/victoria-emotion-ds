# victoria-emotion-ds — Manuelles Review-Tool

Dieses Repository enthält das Tool für das manuelle Abhören und Bewerten des [victoria-emotion-de-synthetic](https://huggingface.co/datasets/dida-80b/victoria-emotion-de-synthetic) Datensatzes. Mehrere Reviewer können unabhängig voneinander bewerten — ihre Entscheidungen werden automatisch über GitHub zusammengeführt.

---

## Was macht das Tool?

- Lädt die Clip-Liste direkt vom HuggingFace-Datensatz (kein lokaler Download nötig)
- Spielt die Audio-Clips im Browser ab
- Du bewertest jeden Clip mit **Keep**, **Delete** oder **Rerender**
- Deine Entscheidungen werden lokal gespeichert und per Knopfdruck auf GitHub gepusht
- Wenn mehrere Reviewer bewertet haben, kann ein Merge-Script Übereinstimmungen und Konflikte herausfiltern

---

## Voraussetzungen

- Python 3.10 oder neuer
- Git mit SSH-Zugang zu GitHub (oder HTTPS mit Token)
- Internetverbindung (Audio wird von HuggingFace gestreamt)

---

## Setup (einmalig)

### 1. Repository klonen

```bash
git clone git@github.com:dida-80b/victoria-emotion-ds.git
cd victoria-emotion-ds
```

### 2. Abhängigkeiten installieren

```bash
pip install fastapi uvicorn
```

---

## Review starten

```bash
python3 tools/qc_review_server.py --reviewer dein-name
```

Dann im Browser öffnen: **http://127.0.0.1:8091**

`dein-name` wird als Dateiname für deine Entscheidungen verwendet (z.B. `reviews/dein-name.json`). Wähle einen eindeutigen Namen, z.B. deinen Vornamen.

---

## Bedienung

### Tastaturkürzel

| Taste | Aktion |
|-------|--------|
| `K` | Keep — Clip ist gut, behalten |
| `D` | Delete — Clip ist fehlerhaft, löschen |
| `R` | Rerender — Clip neu generieren |
| `N` oder `→` | Nächster Clip |
| `B` oder `←` | Vorheriger Clip |
| `Leertaste` | Play / Pause |

### Filter (linke Sidebar)

- **Emotion** — zeigt nur Clips einer bestimmten Emotion
- **Status** — zeigt nur unbewertete, beibehaltene, zu löschende oder neu zu generierende Clips
- **Suchfeld** — filtert nach Text oder Dateiname

### Andere Reviewer sehen

Wenn andere Reviewer ihre Entscheidungen bereits gepusht haben, werden diese beim nächsten Sync automatisch geladen und neben jedem Clip angezeigt:

- Kleine farbige Punkte in der Liste zeigen die Entscheidungen anderer
- In der Detailansicht steht z.B. **alex: keep · max: delete**
- Bei Meinungsverschiedenheit erscheint ein roter **Konflikt**-Badge
- Bei Übereinstimmung ein grüner **Einig**-Badge

---

## Entscheidungen auf GitHub synchronisieren

Oben rechts im Browser auf **↑ Sync** klicken.

Das Tool macht automatisch:
1. Deine neuen Entscheidungen committen
2. Neueste Entscheidungen der anderen Reviewer von GitHub holen (`git pull`)
3. Alles hochladen (`git push`)
4. Die Anzeige der anderen Reviewer aktualisieren

---

## Ergebnisse zusammenführen (nach Abschluss aller Reviews)

Wenn alle Reviewer fertig sind und gepusht haben:

```bash
python3 tools/qc_merge_reviews.py \
  --state-dir reviews \
  --output-dir merge \
  --min-reviewers 2 \
  --replace
```

### Output-Dateien in `merge/`

| Datei | Inhalt |
|-------|--------|
| `consensus_keep.jsonl` | Alle Reviewer einig: behalten |
| `consensus_delete.jsonl` | Alle Reviewer einig: löschen |
| `consensus_rerender.jsonl` | Alle Reviewer einig: neu generieren |
| `conflicts.jsonl` | Meinungsverschiedenheit — manuell klären |
| `partial.jsonl` | Noch nicht von allen Reviewern bewertet |
| `summary.json` | Statistik: Counts pro Entscheidung und Reviewer |

Konflikte werden auch direkt im Terminal ausgegeben mit allen Votes.

---

## Tipps

- Du kannst jederzeit aufhören und später weitermachen — dein Fortschritt wird automatisch gespeichert
- Falls du eine Entscheidung ändern möchtest, einfach den Clip nochmal aufrufen und neu bewerten
- Der Filter **Status: unreviewed** zeigt dir schnell was noch offen ist
- Autoplay (oben rechts) startet den nächsten Clip automatisch nach einer Entscheidung
