#!/usr/bin/env python3
"""
scripts/generate_corpus.py — Generate synthetic annotated papers for ScholarEnv.

Produces 3 papers covering:
  paper_001 — NLP benchmark paper (easy inconsistencies, clear table refs)
  paper_002 — Computer vision survey (medium, more tables, injected discrepancies)
  paper_003 — Multi-task learning paper (hard, nested claims, subtle mismatches)

Each paper is a realistic synthetic document with:
  - Well-structured sections (abstract, intro, methods, results, discussion, refs)
  - Tables with numerical data
  - Ground truth annotations for Tasks 1, 2, and 3
  - Injected discrepancies (text says X, table says Y)

Run:
  python scripts/generate_corpus.py

Outputs:
  data/papers/paper_001.json
  data/papers/paper_002.json
  data/papers/paper_003.json
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("data/papers")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Paper 001: NLP Benchmark — LanguageNet ────────────────────────────────────

PAPER_001 = {
    "id":    "paper_001",
    "title": "LanguageNet: A Multi-Task Pre-Training Framework for Natural Language Understanding",
    "source": "arxiv:synthetic_001",
    "license": "CC-BY 4.0",
    "difficulty_score": 0.35,
    "sections": {
        "abstract": (
            "We present LanguageNet, a multi-task pre-training framework for natural language "
            "understanding (NLU). Our model achieves state-of-the-art results on the GLUE "
            "benchmark, reaching an overall score of 94.3, outperforming prior methods by "
            "2.1 points. The model was trained on 128 billion tokens using a mixture of "
            "masked language modelling, next sentence prediction, and span boundary objectives. "
            "We evaluate across eight downstream tasks and report consistent improvements. "
            "LanguageNet demonstrates that joint training across heterogeneous NLU tasks "
            "provides complementary supervision signals that improve generalisation. "
            "We release model weights and training code to facilitate reproducibility."
        ),
        "introduction": (
            "Natural language understanding (NLU) encompasses a wide range of tasks including "
            "sentiment analysis, textual entailment, and question answering. The GLUE benchmark "
            "(Wang et al., 2018) provides a standardised evaluation suite across eight tasks. "
            "Recent models such as BERT [1], RoBERTa [2], and DeBERTa [3] have pushed performance "
            "significantly. In this work, we propose LanguageNet, which extends the pre-training "
            "paradigm with three complementary objectives. Our main contributions are as follows: "
            "(1) a novel multi-task pre-training objective combining three learning signals, "
            "(2) a curriculum scheduling strategy that adapts task weights during training, "
            "and (3) comprehensive ablation studies demonstrating the contribution of each component. "
            "We achieve a GLUE score of 94.3, establishing a new state of the art."
        ),
        "methods": (
            "LanguageNet is built on a 340M parameter transformer architecture. We use a "
            "vocabulary of 50,265 byte-pair encoding (BPE) tokens. Pre-training uses three "
            "objectives simultaneously: (1) masked language modelling (MLM) with a masking "
            "probability of 15%, (2) next sentence prediction (NSP), and (3) span boundary "
            "objective (SBO) [4]. Training runs for 1 million steps on 128 billion tokens "
            "drawn from a mixture of BookCorpus, English Wikipedia, CC-News, and OpenWebText. "
            "We use the AdamW optimiser [5] with a peak learning rate of 1e-4, linear warmup "
            "over 10,000 steps, and polynomial decay. Batch size is 8,192 sequences of 512 tokens. "
            "Fine-tuning follows the standard protocol: we add a task-specific classification head "
            "and train for 3 epochs with a learning rate of 2e-5."
        ),
        "results": (
            "Table 1 reports GLUE benchmark results. LanguageNet achieves an average score of "
            "91.7 across all eight tasks, with particular strength on MNLI (90.2) and QQP (92.5). "
            "On the SST-2 sentiment task, our model reaches 97.1 accuracy. "
            "Table 2 presents ablation results showing the contribution of each pre-training "
            "objective. Removing SBO reduces GLUE score by 1.8 points, while removing NSP "
            "reduces it by 0.9 points, confirming the value of our multi-task design. "
            "Training time was 14 days on 64 NVIDIA A100 GPUs. "
            "Our model uses 340 million parameters, comparable to BERT-Large."
        ),
        "discussion": (
            "The results confirm that multi-task pre-training provides complementary supervision. "
            "The SBO objective appears most valuable for tasks requiring span-level reasoning "
            "such as SQuAD. The curriculum scheduler reduces training instability during "
            "the early stages where task gradients conflict. One limitation of our approach "
            "is the increased computational cost compared to single-objective pre-training. "
            "Future work will explore parameter-efficient adaptation and distillation "
            "to smaller model sizes."
        ),
        "references": (
            "[1] Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.\n"
            "[2] Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv.\n"
            "[3] He et al. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR.\n"
            "[4] Joshi et al. (2020). SpanBERT. TACL.\n"
            "[5] Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization. ICLR."
        ),
    },
    "tables": {
        "Table 1": {
            "caption": "Table 1: GLUE benchmark results. Best results per task in bold.",
            "data": {
                "MNLI": "90.2",
                "QQP": "92.5",
                "QNLI": "95.3",
                "SST-2": "97.1",
                "CoLA": "67.4",
                "STS-B": "91.8",
                "MRPC": "89.6",
                "RTE": "87.0",
                "Average": "91.7",
            },
        },
        "Table 2": {
            "caption": "Table 2: Ablation study. Each row removes one pre-training objective.",
            "data": {
                "Full model": "91.7",
                "w/o SBO":    "89.9",
                "w/o NSP":    "90.8",
                "w/o MLM":    "73.2",
            },
        },
    },
    "figures": {
        "Figure 1": {
            "caption": "Figure 1: Training loss curves for LanguageNet.",
            "type":    "line_chart",
        },
    },
    "ground_truth": {
        "task1_violations": [
            {"rule": "citation_format_ieee",
             "note": "Uses (Author, Year) style; IEEE requires [N]"},
            {"rule": "abstract_max_words",
             "actual": 105, "limit": 100,
             "note": "Abstract slightly over IEEE 100-word limit"},
        ],
        "task2_inconsistencies": [
            {
                "id":         "IC_001",
                "type":       "number_mismatch",
                "location_a": "abstract",
                "claim_a":    "reaching an overall score of 94.3",
                "location_b": "results",
                "claim_b":    "LanguageNet achieves an average score of 91.7",
                "injected":   False,
                "note":       "Abstract claims 94.3 but results section and Table 1 show 91.7",
            },
            {
                "id":         "IC_002",
                "type":       "contribution_count",
                "location_a": "introduction",
                "claim_a":    "three complementary objectives",
                "location_b": "methods",
                "claim_b":    "three objectives simultaneously",
                "injected":   False,
                "note":       "Consistent count — this is NOT an inconsistency (control entry)",
            },
        ],
        "task3_discrepancies": [
            {
                "id":           "D_001",
                "type":         "table_text_mismatch",
                "text_location": "abstract",
                "text_claim":   "reaching an overall score of 94.3",
                "table_id":     "Table 1",
                "table_value":  "91.7",
                "injected":     True,
                "note":         "Abstract inflated by 2.6 points vs Table 1 Average",
            },
        ],
    },
}


# ── Paper 002: Computer Vision Survey ────────────────────────────────────────

PAPER_002 = {
    "id":    "paper_002",
    "title": "DenseVision: Efficient Dense Prediction with Hierarchical Feature Aggregation",
    "source": "arxiv:synthetic_002",
    "license": "CC-BY 4.0",
    "difficulty_score": 0.60,
    "sections": {
        "abstract": (
            "We introduce DenseVision, an efficient architecture for dense prediction tasks "
            "including semantic segmentation and depth estimation. DenseVision employs "
            "hierarchical feature aggregation across four resolution scales, achieving "
            "mIoU of 56.2 on the ADE20K dataset while running at 47 frames per second "
            "on a single NVIDIA RTX 3090. Compared to prior efficient methods, our model "
            "reduces memory consumption by 38% while maintaining competitive accuracy. "
            "We evaluate on three benchmarks: ADE20K, Cityscapes, and NYU-Depth-v2."
        ),
        "introduction": (
            "Dense prediction tasks require per-pixel understanding of scene content. "
            "Semantic segmentation assigns a class label to every pixel [1], while monocular "
            "depth estimation predicts a continuous depth map [2]. State-of-the-art methods "
            "such as SegFormer [3] and BEiT [4] achieve high accuracy but are computationally "
            "expensive. We propose DenseVision, designed for real-time inference. "
            "Our contributions: "
            "(1) a hierarchical feature pyramid that aggregates multi-scale context efficiently, "
            "(2) a lightweight attention mechanism that reduces quadratic complexity to linear, "
            "(3) a joint training protocol for segmentation and depth simultaneously, and "
            "(4) comprehensive benchmarks across three standard datasets. "
            "The model runs at 47 fps, meeting real-time constraints for autonomous driving."
        ),
        "methods": (
            "DenseVision consists of a MobileNetV3 backbone followed by four aggregation "
            "stages at stride 4, 8, 16, and 32. Each stage produces a feature map that is "
            "upsampled and summed with the preceding level. The lightweight attention module "
            "decomposes the attention matrix into two low-rank factors, achieving O(n) "
            "complexity. The joint loss combines cross-entropy for segmentation and "
            "scale-invariant log-RMSE for depth, weighted 0.7:0.3. We train for 160,000 "
            "iterations with batch size 16 on 4 A100 GPUs. Learning rate follows a poly "
            "schedule from 6e-5 to 0. Data augmentation includes random horizontal flip, "
            "random crop to 512×512, and colour jitter."
        ),
        "results": (
            "Table 1 reports semantic segmentation results on ADE20K. DenseVision achieves "
            "mIoU of 54.8 — competitive with SegFormer-B2 (51.8) while running 3.2× faster. "
            "Table 2 reports depth estimation results on NYU-Depth-v2. Our model achieves "
            "delta1 accuracy of 0.921 and RMSE of 0.341. "
            "Table 3 shows inference speed vs. accuracy on Cityscapes. DenseVision "
            "achieves 78.3 mIoU at 47 fps. Compared to prior efficient models, "
            "memory usage is reduced by 38%. The parameter count is 31.2 million, "
            "substantially smaller than SegFormer-B5 (82M)."
        ),
        "discussion": (
            "DenseVision demonstrates that hierarchical aggregation without heavy attention "
            "is sufficient for competitive dense prediction. The 0.7:0.3 joint loss weighting "
            "was found empirically — adjusting to 0.5:0.5 degraded segmentation by 1.1 mIoU "
            "while improving depth RMSE by 0.012. The linear attention approximation "
            "introduces a small accuracy gap (0.3 mIoU) but enables real-time inference. "
            "Limitations: the joint training may not generalise to all dense prediction tasks."
        ),
        "references": (
            "[1] Long et al. (2015). Fully Convolutional Networks. CVPR.\n"
            "[2] Eigen et al. (2014). Depth Map Prediction. NeurIPS.\n"
            "[3] Xie et al. (2021). SegFormer. NeurIPS.\n"
            "[4] Bao et al. (2021). BEiT. ICLR."
        ),
    },
    "tables": {
        "Table 1": {
            "caption": "Table 1: Semantic segmentation on ADE20K validation set.",
            "data": {
                "DenseVision":   {"mIoU": "54.8", "params": "31.2M", "fps": "47"},
                "SegFormer-B2":  {"mIoU": "51.8", "params": "25M",   "fps": "15"},
                "SegFormer-B5":  {"mIoU": "56.1", "params": "82M",   "fps": "4"},
            },
        },
        "Table 2": {
            "caption": "Table 2: Depth estimation on NYU-Depth-v2.",
            "data": {
                "DenseVision": {"delta1": "0.921", "RMSE": "0.341"},
                "BEiT-Large":  {"delta1": "0.956", "RMSE": "0.270"},
            },
        },
        "Table 3": {
            "caption": "Table 3: Speed vs. accuracy on Cityscapes.",
            "data": {
                "DenseVision":  {"mIoU": "78.3", "fps": "47", "memory_MB": "3240"},
                "DeepLabV3+":   {"mIoU": "80.1", "fps": "12", "memory_MB": "5200"},
            },
        },
    },
    "figures": {
        "Figure 1": {"caption": "Figure 1: DenseVision architecture.", "type": "architecture"},
        "Figure 2": {"caption": "Figure 2: Qualitative segmentation results.", "type": "samples"},
    },
    "ground_truth": {
        "task1_violations": [
            {"rule": "keywords_section_present", "note": "No Keywords section"},
            {"rule": "author_block_present",      "note": "No author affiliation block"},
        ],
        "task2_inconsistencies": [
            {
                "id":         "IC_001",
                "type":       "number_mismatch",
                "location_a": "abstract",
                "claim_a":    "mIoU of 56.2 on the ADE20K dataset",
                "location_b": "results",
                "claim_b":    "DenseVision achieves mIoU of 54.8",
                "injected":   False,
                "note":       "Abstract inflated by 1.4 mIoU",
            },
            {
                "id":         "IC_002",
                "type":       "contribution_count",
                "location_a": "introduction",
                "claim_a":    "four contributions listed",
                "location_b": "methods",
                "claim_b":    "three methodological elements described",
                "injected":   False,
                "note":       "Intro promises 4 contributions, methods only implements 3",
            },
        ],
        "task3_discrepancies": [
            {
                "id":           "D_001",
                "type":         "table_text_mismatch",
                "text_location": "abstract",
                "text_claim":   "mIoU of 56.2 on the ADE20K dataset",
                "table_id":     "Table 1",
                "table_value":  "54.8",
                "injected":     True,
            },
            {
                "id":           "D_002",
                "type":         "table_text_mismatch",
                "text_location": "results",
                "text_claim":   "DenseVision achieves mIoU of 54.8 — competitive with SegFormer-B2 (51.8)",
                "table_id":     "Table 1",
                "table_value":  "SegFormer-B2 mIoU=51.8",
                "injected":     False,
                "note":         "This one IS consistent — control entry",
            },
        ],
    },
}


# ── Paper 003: Multi-Task Learning ────────────────────────────────────────────

PAPER_003 = {
    "id":    "paper_003",
    "title": "UnifiedLM: Scaling Multi-Task Language Models with Adaptive Gradient Balancing",
    "source": "arxiv:synthetic_003",
    "license": "CC-BY 4.0",
    "difficulty_score": 0.80,
    "sections": {
        "abstract": (
            "We present UnifiedLM, a large-scale multi-task language model trained on "
            "23 diverse NLP tasks simultaneously. UnifiedLM-3B achieves an average improvement "
            "of 4.7% over single-task baselines across all evaluated tasks. On SuperGLUE, "
            "UnifiedLM scores 91.2, surpassing human performance (89.8) by 1.4 points. "
            "Our adaptive gradient balancing (AGB) algorithm dynamically reweights task "
            "gradients to prevent dominated tasks from collapsing. We train models at "
            "three scales: 350M, 1B, and 3B parameters."
        ),
        "introduction": (
            "Multi-task learning (MTL) in NLP seeks to share representations across tasks, "
            "improving data efficiency and generalisation. Classic challenges include negative "
            "transfer [1] and gradient conflict between tasks [2]. We introduce UnifiedLM "
            "and the AGB algorithm. Contributions: "
            "(1) AGB — an adaptive gradient balancing algorithm that provably reduces "
            "gradient conflict in MTL settings, "
            "(2) a unified training protocol across 23 NLP tasks without task-specific "
            "hyperparameter tuning, "
            "(3) state-of-the-art SuperGLUE results at the 3B parameter scale. "
            "UnifiedLM-3B achieves 91.2 on SuperGLUE."
        ),
        "methods": (
            "UnifiedLM is based on a T5 [3] encoder-decoder backbone. We train on 23 tasks "
            "from the FLAN collection [4] and additional tasks from PromptSource [5]. "
            "The AGB algorithm computes per-task gradient norms at each step and reweights "
            "gradients to equalise their magnitudes. Specifically, for K tasks, task k "
            "receives weight w_k = median_norm / ||g_k||. This prevents any single task "
            "from dominating the shared parameter updates. Models are trained at three "
            "scales: 350M, 1B, and 3B parameters. Training uses Adafactor with a "
            "learning rate of 5e-4 for 500,000 steps."
        ),
        "results": (
            "Table 1 shows SuperGLUE results. UnifiedLM-3B achieves 91.2. "
            "Table 2 compares average improvement over single-task baselines across "
            "all 23 tasks. UnifiedLM-3B improves by 3.9% on average — a significant "
            "and consistent gain across task families. "
            "Table 3 ablates the AGB algorithm. Removing AGB reduces SuperGLUE score "
            "by 2.3 points (from 91.2 to 88.9). Replacing AGB with PCGrad [2] gives "
            "90.1, confirming AGB's superiority. "
            "At the 1B scale, UnifiedLM scores 88.4 on SuperGLUE. "
            "Figure 1 shows gradient conflict reduction over training steps."
        ),
        "discussion": (
            "AGB's provable gradient conflict reduction translates to consistent accuracy "
            "gains. The 3B model is our strongest, but the 1B model offers a better "
            "efficiency-accuracy trade-off. One limitation is the computational overhead "
            "of computing per-task gradient norms at each step, adding approximately 8% "
            "to training time. Future work will explore second-order AGB variants "
            "and extension to vision-language tasks."
        ),
        "references": (
            "[1] Crawshaw (2020). Multi-Task Learning with Deep Neural Networks. arXiv.\n"
            "[2] Yu et al. (2020). Gradient Surgery for Multi-Task Learning. NeurIPS.\n"
            "[3] Raffel et al. (2020). Exploring the Limits of Transfer Learning with T5. JMLR.\n"
            "[4] Wei et al. (2022). Finetuned Language Models Are Zero-Shot Learners. ICLR.\n"
            "[5] Bach et al. (2022). PromptSource. ACL."
        ),
    },
    "tables": {
        "Table 1": {
            "caption": "Table 1: SuperGLUE benchmark results.",
            "data": {
                "UnifiedLM-350M": "85.3",
                "UnifiedLM-1B":   "88.4",
                "UnifiedLM-3B":   "91.2",
                "Human baseline": "89.8",
                "T5-11B":         "90.3",
            },
        },
        "Table 2": {
            "caption": "Table 2: Average improvement over single-task baselines (23 tasks).",
            "data": {
                "UnifiedLM-350M": "+2.1%",
                "UnifiedLM-1B":   "+3.2%",
                "UnifiedLM-3B":   "+3.9%",
            },
        },
        "Table 3": {
            "caption": "Table 3: Ablation of gradient balancing algorithm.",
            "data": {
                "UnifiedLM-3B (full)": "91.2",
                "w/o AGB":             "88.9",
                "w/ PCGrad":           "90.1",
            },
        },
    },
    "figures": {
        "Figure 1": {
            "caption": "Figure 1: Gradient conflict (cosine similarity) over training.",
            "type": "line_chart",
        },
    },
    "ground_truth": {
        "task1_violations": [
            {"rule": "abstract_max_words", "actual": 118, "limit": 100},
            {"rule": "citation_format_ieee", "note": "Uses [N] inline but references not IEEE formatted"},
        ],
        "task2_inconsistencies": [
            {
                "id":         "IC_001",
                "type":       "number_mismatch",
                "location_a": "abstract",
                "claim_a":    "average improvement of 4.7% over single-task baselines",
                "location_b": "results",
                "claim_b":    "UnifiedLM-3B improves by 3.9% on average",
                "injected":   False,
                "note":       "Abstract says 4.7% but results and Table 2 say 3.9%",
            },
            {
                "id":         "IC_002",
                "type":       "contribution_count",
                "location_a": "introduction",
                "claim_a":    "three contributions listed in introduction",
                "location_b": "abstract",
                "claim_b":    "abstract does not enumerate contributions",
                "injected":   False,
                "note":       "Intro lists 3 contributions; verify methods covers all 3",
            },
        ],
        "task3_discrepancies": [
            {
                "id":           "D_001",
                "type":         "table_text_mismatch",
                "text_location": "abstract",
                "text_claim":   "average improvement of 4.7% over single-task baselines",
                "table_id":     "Table 2",
                "table_value":  "+3.9%",
                "injected":     True,
                "note":         "Abstract says 4.7%, Table 2 shows 3.9% for UnifiedLM-3B",
            },
            {
                "id":           "D_002",
                "type":         "table_text_mismatch",
                "text_location": "results",
                "text_claim":   "UnifiedLM-3B improves by 3.9% on average",
                "table_id":     "Table 2",
                "table_value":  "+3.9%",
                "injected":     False,
                "note":         "CONSISTENT — control; should NOT be reported",
            },
        ],
    },
}


# ── Write papers ──────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    import sys
    force = force or "--force" in sys.argv
    papers = [PAPER_001, PAPER_002, PAPER_003]

    # Skip if all JSON files already exist — avoids overwriting hand-annotated GT
    if not force and all((OUT_DIR / f"{p['id']}.json").exists() for p in papers):
        print(f"  Corpus already present in {OUT_DIR.resolve()} — skipping.")
        print("  Pass --force to regenerate from scratch.")
        return
    for paper in papers:
        out_path = OUT_DIR / f"{paper['id']}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(paper, f, indent=2, ensure_ascii=False)
        print(f"  Written: {out_path}")
    print(f"\n✓ {len(papers)} papers written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
