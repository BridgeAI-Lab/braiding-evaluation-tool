# Braiding Annotation Tool

Flask web app for human evaluation of braided narratives.

## Setup

```bash
cd braiding-evaluation-tool
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

Then open http://127.0.0.1:5000

## Default admin

- **Username:** admin
- **Password:** admin

(Change in production!)

## Usage

1. **Admin:** Log in as admin → Upload Excel (columns: narrative_number, dataset, Review1, Review2, braided_narrative, gt_trust, gpt_trust) → Assign annotators → View/download results.
2. **Annotators:** Register → Log in → Annotate assigned files (trust scores + Fluency/Coherency 1–5).

## Excel format

Upload: `narrative_number`, `dataset`, `Review1`, `Review2`, `braided_narrative`, `gt_trust`, `gpt_trust`

Download: Same columns + one column per annotator with format `[70, 30] | F:4 | C:5` (trust | Fluency | Coherency).
