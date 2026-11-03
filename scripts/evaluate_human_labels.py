"""Score frozen human labels with the pinned model, never infer gold labels."""
from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.nlp_ei.authentic import TokenCompleteFinbert,evaluate_labels,sha256
root=Path(__file__).resolve().parents[1];out=root/"outputs/authentic"
source=out/"human_annotation.csv";raw=source.read_bytes()
rows=list(csv.DictReader(raw.decode("utf-8").splitlines()))
if not rows or any(row[key] not in ("negative","neutral","positive") for row in rows for key in ("annotator_a","annotator_b","adjudicated_label")):
    raise SystemExit("Complete independent human annotation and adjudication before evaluation")
config=json.loads((root/"config/authentic-study.json").read_text())
model=TokenCompleteFinbert(config["model"],config["model_revision"],root/"data/model-cache")
for row in rows:row["predicted_label"]=model.score(row["text"])["label"]
result=evaluate_labels(rows);result["human_label_sha256"]=sha256(raw);result["model_revision"]=config["model_revision"]
(out/"human_evaluation.json").write_text(json.dumps(result,indent=2)+"\n", newline="\n", encoding="utf-8")
print(json.dumps(result,indent=2))
