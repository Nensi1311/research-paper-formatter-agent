"""fix_papers.py — adds task4_citations ground truth to all 3 papers"""
import json
from pathlib import Path

t4 = {
    "paper_001": [
        {"id": "ref_1", "citation_number": "1", "raw": "Devlin et al. (2018). BERT. NAACL.", "status": "valid", "year": 2018, "injected": False},
        {"id": "ref_2", "citation_number": "2", "raw": "Liu et al. (2019). RoBERTa. arXiv.", "status": "valid", "year": 2019, "injected": False},
        {"id": "ref_3", "citation_number": "3", "raw": "Zhang et al. (2023). SuperBERT: 99.8% on all NLP. ICML.", "status": "ghost", "year": 2023, "injected": True},
        {"id": "ref_4", "citation_number": "4", "raw": "Joshi et al. (2020). SpanBERT. TACL.", "status": "valid", "year": 2020, "injected": False},
        {"id": "ref_5", "citation_number": "5", "raw": "Loshchilov & Hutter (2019). AdamW. ICLR.", "status": "valid", "year": 2019, "injected": False},
    ],
    "paper_002": [
        {"id": "ref_1", "citation_number": "1", "raw": "Long et al. (2015). FCN. CVPR.", "status": "valid", "year": 2015, "injected": False},
        {"id": "ref_2", "citation_number": "2", "raw": "Eigen et al. (2014). Depth Prediction. NeurIPS.", "status": "valid", "year": 2014, "injected": False},
        {"id": "ref_3", "citation_number": "3", "raw": "Xie et al. (2021). SegFormer. NeurIPS.", "status": "valid", "year": 2021, "injected": False},
        {"id": "ref_4", "citation_number": "4", "raw": "Patel & Rodriguez (2022). UltraSegNet 200fps. ECCV.", "status": "ghost", "year": 2022, "injected": True},
    ],
    "paper_003": [
        {"id": "ref_1", "citation_number": "1", "raw": "Crawshaw (2020). MTL Survey. arXiv.", "status": "valid", "year": 2020, "injected": False},
        {"id": "ref_2", "citation_number": "2", "raw": "Yu et al. (2020). Gradient Surgery. NeurIPS.", "status": "valid", "year": 2020, "injected": False},
        {"id": "ref_3", "citation_number": "3", "raw": "Raffel et al. (2020). T5. JMLR.", "status": "valid", "year": 2020, "injected": False},
        {"id": "ref_4", "citation_number": "4", "raw": "Kumar & Singh (2024). OmniTask-7B: All NLP. Nature MI.", "status": "ghost", "year": 2024, "injected": True},
        {"id": "ref_5", "citation_number": "5", "raw": "Bach et al. (2022). PromptSource. ACL.", "status": "valid", "year": 2022, "injected": False},
    ],
}

for pid, refs in t4.items():
    path = Path("data") / "papers" / f"{pid}.json"
    d = json.loads(path.read_text())
    d["ground_truth"]["task4_citations"] = refs
    path.write_text(json.dumps(d, indent=2))
    keys = list(d["ground_truth"].keys())
    print(f"Fixed {pid}: {keys}")

print("\nDone. Verify with: python verify_papers.py")
