"""
Synthetic research paper dataset.
Each paper is in some partially-formatted state that the agent must fix.
"""

from typing import Dict, List
from models import Section, SectionType, Reference, AuthorInfo, ConferenceFormat


# ──────────────────────────────────────────────
# Paper definitions
# ──────────────────────────────────────────────

PAPERS: Dict[str, dict] = {

    # ── EASY: NeurIPS → IEEE, mostly structural issues ──
    "paper_neurips_2024": {
        "title": "Efficient Transformers via Sparse Attention Mechanisms",
        "current_format": ConferenceFormat.NeurIPS,
        "authors": [
            AuthorInfo(name="Alice Johnson", affiliation="MIT CSAIL", email="alice@mit.edu", order=0),
            AuthorInfo(name="Bob Chen", affiliation="Stanford AI Lab", email="bob@stanford.edu", order=1),
            AuthorInfo(name="Carol Williams", affiliation="Google DeepMind", order=2),
        ],
        "sections": [
            Section(name="Abstract", section_type=SectionType.ABSTRACT, word_count=220,
                    content_snippet="We propose SparseFormer, a novel transformer architecture that reduces attention complexity from O(n²) to O(n log n)..."),
            Section(name="Introduction", section_type=SectionType.INTRODUCTION, word_count=650,
                    content_snippet="Transformer models have revolutionized natural language processing, but their quadratic attention cost..."),
            Section(name="Related Work", section_type=SectionType.RELATED_WORK, word_count=400,
                    content_snippet="Efficient transformers have been studied extensively. Linformer (Wang et al., 2020) projects keys..."),
            Section(name="Method", section_type=SectionType.METHODOLOGY, word_count=800,
                    content_snippet="Our sparse attention mechanism selects the top-k most relevant tokens per query position..."),
            Section(name="Experiments", section_type=SectionType.EXPERIMENTS, word_count=600,
                    content_snippet="We evaluate SparseFormer on GLUE, SuperGLUE, and ImageNet benchmarks...", has_tables=True),
            Section(name="Results", section_type=SectionType.RESULTS, word_count=350,
                    content_snippet="SparseFormer achieves 87.3 GLUE score, matching dense attention at 3.4× speedup...", has_tables=True, has_figures=True),
            Section(name="Conclusion", section_type=SectionType.CONCLUSION, word_count=200,
                    content_snippet="We presented SparseFormer, achieving competitive performance with 3.4× speedup..."),
            Section(name="References", section_type=SectionType.REFERENCES, word_count=300,
                    content_snippet=""),
        ],
        "section_order": ["Abstract", "Introduction", "Related Work", "Method", "Experiments", "Results", "Conclusion", "References"],
        "references": [
            Reference(index=1, authors=["Vaswani, A.", "Shazeer, N."], title="Attention Is All You Need",
                      venue="NeurIPS", year=2017, style="APA"),
            Reference(index=2, authors=["Wang, S.", "Li, B. Z."], title="Linformer: Self-Attention with Linear Complexity",
                      venue="arXiv", year=2020, style="APA"),
            Reference(index=3, authors=["Kitaev, N.", "Kaiser, L."], title="Reformer: The Efficient Transformer",
                      venue="ICLR", year=2020, style="APA"),
        ],
        "abstract_word_count": 220,   # NeurIPS allows 250, IEEE only 150 → must truncate
        "column_layout": 1,           # NeurIPS is 1-col; IEEE needs 2
        "title_case_style": "title_case",
        "citation_style": "author_year",  # NeurIPS style; IEEE needs numeric
    },

    # ── MEDIUM: ACM → NeurIPS, author format + references + layout ──
    "paper_acm_systems": {
        "title": "Distributed Training of Large Language Models on Heterogeneous Clusters",
        "current_format": ConferenceFormat.ACM,
        "authors": [
            AuthorInfo(name="D. Zhang", affiliation="CMU Systems Lab", order=0),
            AuthorInfo(name="E. Patel", affiliation="UC Berkeley RISE", order=1),
            AuthorInfo(name="F. Kim", affiliation="Microsoft Research", order=2),
            AuthorInfo(name="G. Martinez", affiliation="AWS AI", order=3),
        ],
        "sections": [
            Section(name="Abstract", section_type=SectionType.ABSTRACT, word_count=185,
                    content_snippet="Training large language models requires massive compute. We present HeteroTrain, a system for distributed training across heterogeneous GPU clusters..."),
            Section(name="1. Introduction", section_type=SectionType.INTRODUCTION, word_count=700,
                    content_snippet="The cost of training foundation models has grown exponentially. GPT-3 required 355 GPU-years..."),
            Section(name="2. Background", section_type=SectionType.RELATED_WORK, word_count=450,
                    content_snippet="Prior systems like Megatron-LM and DeepSpeed assume homogeneous hardware..."),
            Section(name="3. System Design", section_type=SectionType.METHODOLOGY, word_count=900,
                    content_snippet="HeteroTrain uses adaptive pipeline scheduling to balance load across mixed A100/V100/T4 nodes...", has_figures=True),
            Section(name="4. Evaluation", section_type=SectionType.EXPERIMENTS, word_count=700,
                    content_snippet="We train GPT-2 1.5B, LLaMA-7B, and a custom 13B model across 128 heterogeneous GPUs...", has_tables=True, has_figures=True),
            Section(name="5. Related Work", section_type=SectionType.RELATED_WORK, word_count=350,
                    content_snippet="Shoeybi et al. (2019) introduced Megatron-LM for tensor parallelism..."),
            Section(name="6. Conclusion", section_type=SectionType.CONCLUSION, word_count=180,
                    content_snippet="HeteroTrain reduces LLM training costs by up to 42% on heterogeneous clusters..."),
            Section(name="References", section_type=SectionType.REFERENCES, word_count=400,
                    content_snippet=""),
        ],
        "section_order": ["Abstract", "1. Introduction", "2. Background", "3. System Design",
                          "4. Evaluation", "5. Related Work", "6. Conclusion", "References"],
        "references": [
            Reference(index=1, authors=["D. Zhang", "E. Patel"], title="Pipeline Parallelism for DNN Training",
                      venue="MLSys", year=2021, style="ACM"),
            Reference(index=2, authors=["N. Shazeer", "Y. Cheng"], title="Mesh TensorFlow: Deep Learning for Supercomputers",
                      venue="NeurIPS", year=2018, style="ACM"),
            Reference(index=3, authors=["S. Rajbhandari", "J. Rasley"], title="ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
                      venue="SC", year=2020, style="ACM"),
            Reference(index=4, authors=["M. Shoeybi", "M. Patwary"], title="Megatron-LM: Training Multi-Billion Parameter Language Models",
                      venue="arXiv", year=2019, style="ACM"),
        ],
        "abstract_word_count": 185,
        "column_layout": 2,
        "title_case_style": "title_case",
        "citation_style": "numeric",  # ACM uses numeric; NeurIPS uses author_year
    },

    # ── HARD: Multi-issue IEEE→ICML with wrong sections, bad refs, wrong authors, wrong layout ──
    "paper_ieee_ml": {
        "title": "NEURAL ARCHITECTURE SEARCH VIA DIFFERENTIABLE RELAXATION",
        "current_format": ConferenceFormat.IEEE,
        "authors": [
            AuthorInfo(name="H. Liu", affiliation="Tsinghua Univ.", order=0),
            AuthorInfo(name="K. Simonyan", affiliation="DeepMind", order=1),
            AuthorInfo(name="Y. Yang", affiliation="Tsinghua Univ.", order=2),
        ],
        "sections": [
            Section(name="Abstract", section_type=SectionType.ABSTRACT, word_count=145,
                    content_snippet="Neural architecture search (NAS) automates the design of neural network architectures. We propose DARTS, which relaxes the discrete search space..."),
            Section(name="I. Introduction", section_type=SectionType.INTRODUCTION, word_count=550,
                    content_snippet="Designing neural architectures requires significant human expertise. Recent NAS methods..."),
            Section(name="II. Related Work", section_type=SectionType.RELATED_WORK, word_count=380,
                    content_snippet="NAS methods include reinforcement learning [1], evolutionary algorithms [2]..."),
            Section(name="III. Methodology", section_type=SectionType.METHODOLOGY, word_count=750,
                    content_snippet="We relax the categorical choice of operation to a softmax over all possible operations...", has_equations=True),
            Section(name="IV. Experiments", section_type=SectionType.EXPERIMENTS, word_count=680,
                    content_snippet="We search on CIFAR-10 and transfer to ImageNet classification...", has_tables=True, has_figures=True),
            Section(name="V. Discussion", section_type=SectionType.RESULTS, word_count=250,
                    content_snippet="The discovered architectures show competitive performance..."),
            Section(name="VI. Conclusion", section_type=SectionType.CONCLUSION, word_count=190,
                    content_snippet="DARTS efficiently searches for architectures using gradient descent..."),
            Section(name="Acknowledgment", section_type=SectionType.ACKNOWLEDGMENTS, word_count=50,
                    content_snippet="This work was supported by..."),
            Section(name="References", section_type=SectionType.REFERENCES, word_count=450,
                    content_snippet=""),
        ],
        "section_order": ["Abstract", "I. Introduction", "II. Related Work", "III. Methodology",
                          "IV. Experiments", "V. Discussion", "VI. Conclusion", "Acknowledgment", "References"],
        "references": [
            Reference(index=1, authors=["Zoph, B.", "Le, Q. V."], title="Neural Architecture Search with Reinforcement Learning",
                      venue="ICLR", year=2017, style="IEEE"),
            Reference(index=2, authors=["Real, E.", "Moore, S."], title="Large-Scale Evolution of Image Classifiers",
                      venue="ICML", year=2017, style="IEEE"),
            Reference(index=3, authors=["Pham, H.", "Guan, M."], title="Efficient Neural Architecture Search via Parameter Sharing",
                      venue="ICML", year=2018, style="IEEE"),
            Reference(index=4, authors=["Liu, H.", "Simonyan, K.", "Yang, Y."], title="DARTS: Differentiable Architecture Search",
                      venue="ICLR", year=2019, style="IEEE"),
        ],
        "abstract_word_count": 215,  # IEEE allows up to 150... wait, 145 is fine for ICML (200) — inflate to test
        "column_layout": 1,          # IEEE is 2-col but this paper was exported wrong; ICML needs 2
        "title_case_style": "upper", # ALL CAPS title — must fix to title_case
        "citation_style": "numeric", # IEEE style; ICML needs author_year
    },
}


def get_paper(paper_id: str) -> dict:
    if paper_id not in PAPERS:
        raise ValueError(f"Unknown paper_id: {paper_id}. Available: {list(PAPERS.keys())}")
    import copy
    return copy.deepcopy(PAPERS[paper_id])


def list_papers() -> List[str]:
    return list(PAPERS.keys())
