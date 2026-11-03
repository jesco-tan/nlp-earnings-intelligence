"""Real-model execution check on authored fixtures, not authentic study evidence."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.nlp_ei.authentic import TokenCompleteFinbert
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/"config/authentic-study.json").read_text())
model=TokenCompleteFinbert(cfg["model"],cfg["model_revision"],root/"data/model-cache")
text="Revenue increased from the prior year. "+"The company reported its results for the financial year. "*100
score=model.score(text)
assert score["input_tokens"]==score["processed_tokens"] and score["truncated_tokens"]==0 and score["chunks"]>1
out=root/"outputs/authentic";out.mkdir(exist_ok=True)
(out/"backend-smoke.json").write_text(json.dumps({"status":"real_model_executed","scope":"Authored test fixture only; no authentic filing result or independent label evaluation","model":cfg["model"],"revision":cfg["model_revision"],"score":score},indent=2)+"\n", newline="\n", encoding="utf-8")
print("Real FinBERT token-complete smoke passed")
