"""Authentic-study ingestion, extraction, token-complete inference and evaluation."""
from pathlib import Path
import hashlib,json,re,time
import numpy as np
import requests


def sha256(data):return hashlib.sha256(data).hexdigest()


def extract_sections(html):
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(html,"html.parser")
    for tag in soup(["script","style","ix:header"]):tag.decompose()
    # Restrict delimiters to short block headings, excluding inline cross-references.
    pattern=re.compile(r"^ITEM\s+(1A|1B|7A|7)\s*[.:](?:\s|$)",re.I)
    candidates=[]
    for node in soup.find_all(["div","p","h1","h2","h3","h4","td"]):
        label=re.sub(r"\s+"," ",node.get_text(" ",strip=True))
        match=pattern.match(label)
        if match and len(label)<=250:
            candidates.append((node,match.group(1).upper()))
    # Nested duplicate headings receive one marker at the outermost heading block.
    nodes={id(node) for node,_ in candidates}
    for node,item in candidates:
        if not any(id(parent) in nodes for parent in node.parents):
            node.insert_before(f" __SEC_ITEM_{item}__ ")
    text=re.sub(r"\s+"," ",soup.get_text(" ",strip=True)).replace("\u00a0"," ")
    result={}
    for name,start,end in [("risk_factors","1A","1B"),("mda","7","7A")]:
        spans=[]
        for match in re.finditer(f"__SEC_ITEM_{start}__",text):
            stop=text.find(f"__SEC_ITEM_{end}__",match.end())
            if stop>=0:spans.append((match.end(),stop))
        valid=[(a,b) for a,b in spans if b-a>=1000]
        if not valid:
            reason="incorporated_by_reference_or_not_present" if any("incorporat" in text[a:b].lower() for a,b in spans) else "heading_not_found_or_section_too_short"
            result[name]={"status":"missing","reason":reason,"text":"","characters":0}
        else:
            a,b=max(valid,key=lambda span:span[1]-span[0])
            section=re.sub(r"__SEC_ITEM_\w+__","",text[a:b]).strip()
            result[name]={"status":"requires_manual_boundary_review","text":section,"characters":len(section),"start_offset":a,"end_offset":b}
    return result


class SecClient:
    def __init__(self,contact,cache):
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+",contact):raise ValueError("A real SEC contact email is required")
        self.session=requests.Session();self.session.headers["User-Agent"]="Jesco Tan research "+contact
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
    def get(self,url):
        name=self.cache/(sha256(url.encode())+".bin")
        if name.exists():return name.read_bytes()
        time.sleep(.25) # <=4 requests/s, sequential, stop on HTTP errors
        response=self.session.get(url,timeout=90);response.raise_for_status()
        name.write_bytes(response.content);return response.content


class TokenCompleteFinbert:
    def __init__(self,model_name,revision,cache_dir):
        if not re.fullmatch("[a-f0-9]{40}",revision or ""):raise ValueError("Pin an immutable model/tokenizer revision")
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.torch=torch;torch.manual_seed(42);torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision,cache_dir=cache_dir)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name,revision=revision,cache_dir=cache_dir).eval()
        self.revision=revision
    def score(self,text):
        tokens=self.tokenizer.encode(text,add_special_tokens=False,truncation=False,verbose=False)
        if not tokens:raise ValueError("Cannot score empty text")
        size=min(512,int(self.model.config.max_position_embeddings))-self.tokenizer.num_special_tokens_to_add(pair=False)
        chunks=[tokens[i:i+size] for i in range(0,len(tokens),size)]
        weighted={name:0. for name in ["positive","negative","neutral"]}
        for i in range(0,len(chunks),8):
            batch=chunks[i:i+8]
            # Pinned FinBERT uses BERT [CLS] content [SEP] framing.
            framed=[[self.tokenizer.cls_token_id]+c+[self.tokenizer.sep_token_id] for c in batch]
            width=max(map(len,framed))
            encoded={"input_ids":self.torch.tensor([c+[self.tokenizer.pad_token_id]*(width-len(c)) for c in framed]),
                     "attention_mask":self.torch.tensor([[1]*len(c)+[0]*(width-len(c)) for c in framed])}
            with self.torch.inference_mode():probs=self.model(**encoded).logits.softmax(-1).cpu().numpy()
            for chunk,row in zip(batch,probs):
                for j,p in enumerate(row):weighted[self.model.config.id2label[j].lower()]+=float(p)*len(chunk)
        result={key:value/len(tokens) for key,value in weighted.items()}
        result.update(score=result["positive"]-result["negative"],label=max(weighted,key=weighted.get),input_tokens=len(tokens),processed_tokens=sum(map(len,chunks)),truncated_tokens=0,chunks=len(chunks),aggregation="non-overlapping content-token-weighted mean")
        return result


def evaluate_labels(rows):
    classes=["negative","neutral","positive"]
    matrix=np.zeros((3,3),dtype=int)
    disagreements=0
    for row in rows:
        a,b=row["annotator_a"],row["annotator_b"]
        if a not in classes or b not in classes:raise ValueError("Two independent human labels are required on every sentence")
        disagreements+=a!=b
        gold=row["adjudicated_label"]
        if gold not in classes:raise ValueError("Adjudicated human gold labels are required")
        matrix[classes.index(gold),classes.index(row["predicted_label"])]+=1
    metrics={}
    for i,name in enumerate(classes):
        precision=matrix[i,i]/matrix[:,i].sum() if matrix[:,i].sum() else None
        recall=matrix[i,i]/matrix[i,:].sum() if matrix[i,:].sum() else None
        metrics[name]={"support":int(matrix[i,:].sum()),"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0.}
    return {"n":len(rows),"classes":classes,"confusion_matrix":matrix.tolist(),"per_class":metrics,"human_disagreement_rate":disagreements/len(rows)}
