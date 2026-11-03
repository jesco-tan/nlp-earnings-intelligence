"""Bounded authentic annual-filing study. Contact passed via SEC_CONTACT env only."""
import csv,importlib.metadata,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.nlp_ei.authentic import SecClient,TokenCompleteFinbert,extract_sections,sha256,evaluate_labels
ROOT=Path(__file__).resolve().parents[1]

def main():
    config=json.loads((ROOT/"config/authentic-study.json").read_text())
    out=ROOT/"outputs/authentic";out.mkdir(parents=True,exist_ok=True)
    contact=os.environ.get("SEC_CONTACT")
    if not contact:
        (out/"manifest.json").write_text(json.dumps({"status":"waiting_for_sec_contact","independent_evaluation":"waiting_for_human_labels","configuration_sha256":sha256((ROOT/"config/authentic-study.json").read_bytes())},indent=2)+"\n", newline="\n", encoding="utf-8")
        raise SystemExit("Set SEC_CONTACT to the approved real contact email before ingestion")
    client=SecClient(contact,ROOT/"data/authentic-cache")
    records=[]
    for ticker,cik in config["banks"].items():
        source=f"https://data.sec.gov/submissions/CIK{cik}.json"
        submissions=json.loads(client.get(source))
        recent=submissions["filings"]["recent"]
        candidates=[dict(zip(recent,row)) for row in zip(*recent.values())]
        for item in submissions["filings"].get("files",[]):
            if item["filingFrom"][:4] <= "2025" and item["filingTo"][:4] >= "2023":
                extra=json.loads(client.get("https://data.sec.gov/submissions/"+item["name"]))
                candidates.extend(dict(zip(extra,row)) for row in zip(*extra.values()))
        for year in config["report_years"]:
            matching=[x for x in candidates if x["form"]=="10-K" and x["reportDate"]==f"{year}-12-31"]
            if len(matching)!=1:raise ValueError(f"Expected one annual filing for {ticker} {year}, got {len(matching)}")
            f=matching[0];accession=f["accessionNumber"]
            url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-','')}/{f['primaryDocument']}"
            raw=client.get(url);sections=extract_sections(raw.decode("utf-8",errors="replace"))
            for section,extracted in sections.items():records.append({"ticker":ticker,"cik":cik,"report_period":f["reportDate"],"acceptance_time":f["acceptanceDateTime"],"accession":accession,"document_url":url,"source_sha256":sha256(raw),"section":section,**extracted})
    (out/"extracted_sections.json").write_text(json.dumps(records,indent=2)+"\n", newline="\n", encoding="utf-8")
    revision=config["model_revision"]
    if not revision:raise SystemExit("Pin model_revision in config before inference")
    cache={}
    old_manifest=out/"manifest.json";old_scores=out/"section_scores.json"
    if old_manifest.exists() and old_scores.exists():
        previous=json.loads(old_manifest.read_text())
        if previous.get("model_revision")==revision and previous.get("outputs",{}).get("section_scores.json")==sha256(old_scores.read_bytes()):
            cache={(r["accession"],r["section"],r["text"]):r for r in json.loads(old_scores.read_text())}
    model=None;scored=[]
    for record in records:
        if record["status"]=="missing":continue
        key=(record["accession"],record["section"],record["text"])
        if key in cache:
            result=cache[key]
        else:
            if model is None:model=TokenCompleteFinbert(config["model"],revision,ROOT/"data/model-cache")
            result={**record,**model.score(record["text"])}
        scored.append(result)
        print(f"Scored {record['ticker']} {record['report_period']} {record['section']}",flush=True)
    (out/"section_scores.json").write_text(json.dumps(scored,indent=2)+"\n", newline="\n", encoding="utf-8")
    keys=["ticker","report_period","acceptance_time","accession","document_url","source_sha256","section","positive","negative","neutral","score","label","input_tokens","processed_tokens","truncated_tokens","chunks"]
    summary={"status":"preliminary_model_outputs_pending_independent_human_evaluation","model":config["model"],"model_revision":revision,"annual_filings_downloaded":9,"sections_requested":18,"sections_scored":len(scored),"input_tokens":sum(r["input_tokens"] for r in scored),"records":[{k:r[k] for k in keys} for r in scored]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n", newline="\n", encoding="utf-8")
    # Uniform held-out sentence sample within each filing/section, no label-based selection.
    import random
    rng=random.Random(config["seed"]);samples=[]
    for record in records:
        sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+",record["text"]) if len(s.split())>=8]
        for text in rng.sample(sentences,min(5,len(sentences))):
            samples.append({"id":sha256((record["accession"]+record["section"]+text).encode())[:16],"ticker":record["ticker"],"accession":record["accession"],"section":record["section"],"text":text,"annotator_a":"","annotator_b":"","adjudicated_label":""})
    sample=out/"human_annotation.csv"
    if sample.exists():
        with sample.open(encoding="utf-8",newline="") as f:existing=list(csv.DictReader(f))
        if {r["id"] for r in existing}!={r["id"] for r in samples}:
            raise ValueError("Existing annotation sample differs from current extraction; preserve labels and reconcile explicitly")
    else:
        with sample.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(samples[0]),lineterminator="\n");writer.writeheader();writer.writerows(samples)
    manifest={"status":"authentic_scoring_complete_pending_manual_extraction_review_and_human_evaluation","model_revision":revision,"configuration_sha256":sha256((ROOT/"config/authentic-study.json").read_bytes()),"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},"outputs":{p.name:sha256(p.read_bytes()) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"},"no_anomaly_claim":"Only three annual periods; no past-only alert baseline is validated"}
    manifest["source_hashes"]={name:sha256((ROOT/name).read_bytes()) for name in ["scripts/run_authentic.py","src/nlp_ei/authentic.py","scripts/evaluate_human_labels.py","scripts/plot_authentic.py","config/authentic-study.json","requirements-authentic.txt"]}
    manifest["input_documents"]={r["document_url"]:r["source_sha256"] for r in records}
    manifest["model_cache_hashes"]={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()) for p in (ROOT/"data/model-cache").rglob("*") if p.is_file() and "/snapshots/" in p.as_posix()}
    manifest["execution_note"]="Hash-verified previous section scores may be reused when model revision, accession, section and full text match."
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n", newline="\n", encoding="utf-8")

if __name__=="__main__":main()
