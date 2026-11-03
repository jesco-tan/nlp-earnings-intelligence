"""Render current authentic section scores without model execution."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/"outputs/authentic/section_scores.json").read_text(encoding="utf-8"))
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for ax,ticker in zip(axes,["BAC","JPM"]):
    for section,offset,color,label in [("risk_factors",-.18,"#b94c3b","Risk Factors"),("mda",.18,"#397b9c","MD&A")]:
        subset=[x for x in rows if x["ticker"]==ticker and x["section"]==section]
        if subset:ax.bar([int(x["report_period"][:4])+offset for x in subset],[x["score"] for x in subset],width=.34,color=color,label=label)
    ax.set_xticks([2022,2023,2024]);ax.set_title(ticker);ax.axhline(0,color="#999",linewidth=.8);ax.set_ylim(-1,.15);ax.legend()
axes[0].set_ylabel("Net model score (positive minus negative)")
axes[1].text(.5,.94,"MD&A unavailable in primary text",transform=axes[1].transAxes,ha="center",fontsize=9)
fig.suptitle("Authentic filing sections: preliminary FinBERT outputs")
fig.text(.5,.01,"WFC sections unavailable. Token-complete scoring; independent human sentiment evaluation pending.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(ROOT/"outputs/authentic_section_scores.png",dpi=150);plt.close(fig)
