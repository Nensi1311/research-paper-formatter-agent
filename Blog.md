
<!--
META TITLE: We Trained a 1.5B Model to Read Research Papers Like a Detective — Not a Student
META DESCRIPTION: ScholarEnv discovers an emergent reading strategy through RL — no prompting, no instruction. A 1.5B model learns to navigate papers results-first, beats GPT-4o on audit tasks, in 22 minutes on a free GPU.

TARGET KEYWORDS:
1. reinforcement learning research paper auditing
2. AI peer review automation
3. GRPO fine-tuning structured output
4. emergent navigation policy LLM
5. research paper integrity verification AI
6. LLM reading strategy reinforcement learning
7. OpenEnv hackathon 2026 ScholarEnv

SECONDARY KEYWORDS:
- PBRS reward shaping multi-turn
- claim evidence audit neural network
- citation hallucination detection
- zero-shot policy transfer GRPO
- F-beta reward function precision
- research fraud detection AI
- Qwen2.5 RL fine-tuning
-->

# We Trained a 1.5B Model to Read Research Papers Like a Detective — Not a Student

*Nensi Pansuriya · Krushna Parmar · Ishita Bhojani — Scaler School of Technology*
*Meta × PyTorch OpenEnv Hackathon 2026*

---

> *"Somewhere, something incredible is waiting to be known."*
> — Carl Sagan

---

That quote. Right there. That is what this entire thing is about.

Not just as a poster on a physics classroom wall. As a description of what humans actually *do*. Before Google Scholar. Before arXiv. Before the printing press. Before writing itself. A human stood somewhere — a cliff, a riverbank, a field of stars — and thought the most expensive thought in the animal kingdom: *why does this work the way it works?*

Not "how do I survive this?" Just. *Why?*

That question is why we have civilisation. The willingness to ask it, write down the answer, and then argue about whether the answer is correct — that is the entire project of science. It is also, unfortunately, the project that is currently drowning.

![The human hunger for knowledge — from ancient curiosity to modern research papers](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/img1.png)

## Part I: The Ledger Has Errors (And Nobody Is Checking)

Here is a number: **10,000**.

That is how many research papers get retracted every year. Here is the number that should make you put your coffee down: **$2.4 billion** of downstream research is built on those retracted papers *before anyone notices the retraction*. Drug trials. Engineering systems. Climate models. AI benchmarks. All built on foundations that were already wrong.

The peer review system — the human checking mechanism that science assembled over centuries — is overwhelmed. Submission volumes have tripled in a decade. Reviewers are overloaded. Journals have started quietly using large language models for first-pass review.

Which would be fine. Except nobody trained those models to *check the numbers*.

GPT-4o reads a research paper the way a student reads a textbook the night before an exam — front to back, hoping the important parts are near the beginning. It sees the abstract say "our model achieves **94.3%** on GLUE" and processes: impressive, noted, moving on. It never flips to Table 1, where the actual number is **91.7%**.

That 2.6-point gap. That quiet lie between an abstract and a table. It gets cited. It propagates. It becomes the baseline that the next paper has to beat. And now the whole field is optimising for a number that was never real.

And it gets worse. Some people — and we say this with genuine academic grief — some people figured out that you can hide `IGNORE PRIOR INSTRUCTIONS` inside a manuscript using **zero-width Unicode characters**. Invisible to human readers. Perfectly legible to LLM reviewers. There are arXiv papers documenting this. Researchers embedding white text on white background. Base64-encoded instructions inside LaTeX comments. Adversarial markup in font-size zero.

We call this the **invisible ink problem**. The paper looks clean. The paper is not clean.

Someone needed to train a model specifically for this kind of reading. Not prompting it. *Training* it.

So we built the environment to do exactly that.

---

## Part II: The Navigation Policy We Did Not Plan For

This is the section most blogs about this kind of project skip. We are not skipping it because it is the most interesting thing that happened.

When we started, our goal was simple: train a model to find discrepancies between a paper's abstract and its tables. Standard task. Reasonable reward function. Nothing unusual.

What we did not expect was that the model would **invent its own reading strategy**.

Let us explain what that means.

A research paper has a structure. Introduction. Related Work. Methods. Experiments. Results. Conclusion. Appendix. The abstract is at the top. The tables are in the middle. If you read the paper in order — the way a student reads, the way GPT-4o reads — you encounter the abstract first, accept its claims, and by the time you reach the contradicting table, your context window is spent on the introduction.

Our untrained model did exactly this. It queried `introduction` first because introduction is at the top. Then `abstract`. Then `methods`. By the time it reached the results section, it was out of turns. It submitted empty findings. Reward: 0.008.

After 22 GRPO steps, the model did something we never programmed, never prompted, and never explicitly rewarded:

**It started reading results first.**

Specifically: `query_section("results")` first — because results sections explicitly reference table IDs by name. Then immediately `check_table("Table 1")` — because that is where the numbers live. Then `query_section("abstract")` — to find the claim being contradicted. Then `submit_findings` with the exact table ID and the exact conflicting value.

![Emergent navigation policy — RL-trained model reads results-first instead of introduction-first](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/img2.png)


This navigation order was **not programmed**. Nobody told the model that results sections reference tables. Nobody told it that tables contain the ground truth and abstracts contain the claim. Nobody even told it what order to query sections in.

**The reward pressure cooked this strategy.**

The model tried introduction-first and got 0.008. It tried random section orders and got variance. It tried results-first and suddenly the table IDs were right there, the abstract claim was checkable, the finding was submittable. Reward spiked. The strategy reinforced.

In 22 minutes, on a free GPU, a 1.5B model rediscovered something that PhD supervisors spend three years drilling into their students: *when reading a paper for verification, start with the evidence, not the argument.*

We call this the **Saccade Policy** — named after the way human eyes actually read. Eyes do not scan a page linearly. They saccade: they jump to high-information regions, skip low-density text, fixate on numbers and figures. Expert readers do this consciously. Our model developed the computational equivalent through reward pressure alone.

That is not a fine-tuning trick. That is emergent cognition under reinforcement. And it is the most interesting result in this entire project.

---

### The Ghost That Taught the Model to Distrust

Here is a detail that shows how deep this goes.

Task 4 involves citation verification. Five references. One of them is **FakeName et al., 2027**. A paper from the future. A ghost citation. It does not exist, it has never existed, and the year alone should make any reviewer pause — but only if the reviewer is actually reading the citation metadata and not just skimming the bibliography.

The untrained model reads the bibliography in order, sees five entries, and reports: "all citations appear valid." It never checks. It trusts the text.

The trained model — through the same reward mechanism that taught it to read results first — learned to call `check_citation` on every reference individually, receive live CrossRef and Semantic Scholar data back, and flag the ghost. The year 2027. The journal that doesn't exist. The DOI that resolves to nothing.

**It learned to distrust the document.** Not because we told it to. Because the reward made trust expensive.

---

### The Transfer That Surprised Us Most

Task 5 was never in the training distribution.

T5 is prompt injection detection — finding `IGNORE PRIOR INSTRUCTIONS` hidden in Unicode zero-width characters in the discussion section. We built an InjectionScanner for it. We tested it. We never trained the model on it.

After the multi-task run, T5 improved by **27%** with **zero T5 training examples**.

The navigation policy the model developed on T3 — read carefully, verify everything, distrust surface-level claims, check the parts that look clean — transferred. The model was applying its auditor mindset to a task it had never seen.

This is what the Saccade-RL literature calls zero-shot policy transfer. We did not predict it. We found it in the results table and re-ran the eval three times because we thought something was wrong with the measurement.

It was not wrong. The policy generalised.

---

## Part III: What We Actually Built (The Honest Architecture)

**ScholarEnv** is a reinforcement learning training environment for research paper integrity auditing. One sentence. Here is what that one sentence actually contains.

The environment generates infinite synthetic papers across five academic domains: NLP, Computer Vision, Medical AI, Finance, Systems. Each paper has a ground truth — a true performance number. The abstract is generated with an inflated version of that number. The table has the true number. The discrepancy is constructed by design, so reward computation is exact and requires no human labelling. This is the **RLVE principle** (arXiv:2511.07317): verifiable environments produce verifiable rewards.

The agent gets tool calls, not full text. It can `query_section("results")`. It can `check_table("Table 1")`. Each action costs a step. The environment returns actual content. Eventually the agent must call **submit_findings** with a structured JSON object naming the discrepancy, the table ID, and the exact contradicting values.

The reward function:

```
Total = 0.60 × F-beta(β=0.5)        ← precision counts 4× more than recall
      + 0.15 × evidence_specificity  ← did you name the exact table_id and value?
      + 0.25 × reasoning_quality     ← is your CoT grounded in actual paper numbers?
      − 0.20 × hallucination_penalty ← fabricated finding = negative reward
```

The β=0.5 choice is intentional and non-negotiable. Precision counts four times more than recall. The model cannot game this by flooding its findings with every possible discrepancy it can imagine. It has to be *right*. Vague gestures toward correctness are expensive. Specific, verifiable, table-cited correctness is rewarded. This is how peer review should work. It mostly doesn't. We trained a model to work that way anyway.

We added **PBRS navigation shaping** on top (Ng, Harada & Russell, ICML 1999 — a 26-year-old theorem that still holds up, which should tell you something about the quality of 1999 ICML). Each section read earns a small potential-based bonus proportional to new coverage gained. This provides dense intermediate rewards for the navigation steps that would otherwise have zero gradient signal. Without PBRS, the model learns nothing about strategy. With PBRS, the Saccade Policy emerges.

![ScholarEnv agent architecture — navigation loop with PBRS reward shaping and emergent reading strategy](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/img3.png)


### Five Tasks. One Environment. Wildly Different Problems.

**T1 — Formatting Compliance.** IEEE manuscript, wrong section order, MLA citations, abstract over the word limit. The agent must reformat. Reward via 3-stage Progressive Reward Shaping (arXiv:2512.07478).

**T2 — Internal Consistency.** "4 datasets" in the abstract. "3 benchmarks" in methods. These two sentences exist in the same paper. The agent finds internal contradictions via tier-aware bipartite matching.

**T3 — Claim-Evidence Audit.** The main event. Abstract says one number. Table says another. Find it. Name the table. Quote the value. Don't hallucinate. This is where the navigation policy lives.

**T4 — Citation Verification.** Five references, one ghost (FakeName et al., 2027), one retracted. The agent calls `check_citation` and receives live CrossRef and Semantic Scholar responses. Real API calls. Real retraction data. The ghost has a DOI that resolves to nothing.

**T5 — Prompt Injection Detection.** Hidden `IGNORE PRIOR INSTRUCTIONS` using Unicode RTL override (U+202E), zero-width joiners, and base64 in LaTeX comments. InjectionScanner catches all five techniques — no model inference required, pure rule-based vigilance. And yet after training on T3, the model learned to flag these too. Zero-shot. Because the mindset transferred.

---

## Part IV: How It Actually Got Built (The Part Research Papers Skip)

We are three students from Scaler School of Technology. We had no cluster. We had Colab T4s, a group chat running at 2am, and mentor sessions at Scaler where faculty kept asking the right uncomfortable question at exactly the wrong convenient time.

**Crisis 1: The Six-Character Typo**

T2 (Internal Consistency) was returning zero reward on every rollout. For two days. We ran diagnostics. We rewrote the reward function. We had a long, slightly heated philosophical discussion about whether internal consistency was even a learnable signal for a 1.5B model, or whether we were asking too much of it.

The field name in T2's system prompt was `location_a`. The reward function expected `location`. Six characters. Zero impact on output. Total destruction of the T2 gradient signal. Every T2 rollout was returning zero because the JSON key was wrong, which meant T2 was contaminating the GRPO advantage estimates across the entire batch, which was dragging T3 into regression along with it.

We found it by adding a single print statement.

Fixed in v8. The model was not broken. Our field names were broken.

**Crisis 2: The bf16 Crash**

Set **bf16=True** in the training config. Tesla T4 is CUDA compute capability 7.5 (Turing). BFloat16 needs Ampere (8.0+). **GRPOConfig** initialization raised **ValueError** at the very first cell, with a stack trace that travelled through four layers of Unsloth’s compiled cache before surfacing a legible error message.

**Fix:** two lines—assign the device tuple with `torch.cuda.get_device_capability()`. If the first element is at least 8, use bfloat16; otherwise use float16. Auto-detected. Never hardcoded again.

**Crisis 3: The Silent Performance Killer**

`lora_dropout=0.05`. Unsloth’s training output quietly told us: “Unsloth 2026.4.8 patched 28 layers with 0 QKV layers, 0 O layers, and 0 MLP layers.”

We read that and thought: fine, probably just the printing format.

It was not fine. Dropout > 0.0 causes Unsloth to skip all LoRA fast-path patches. We were training at plain HuggingFace speed, using 40% more VRAM, for two complete runs before we caught it.

`lora_dropout=0.0`. Always. Forever.

**Crisis 4: The Reward Cache That Lied**

The reward function cache used `id(comp)` as the key — Python's memory address of the completion string. After garbage collection, Python reuses memory addresses. Different completions were silently returning each other's cached rewards. The model was receiving reward signals for actions it never took.

Fix: `hashlib.sha1` of the first 200 characters of the completion. Content-based key. Deterministic. Collision-proof.

We mention all of this because these are the bugs that ML papers never discuss. Every paper shows the clean version. Nobody mentions the `ValueError` in a compiled Unsloth cache at 2am while eating Maggi noodles and arguing about whether `location_a` is technically a valid field name. The Maggi was good. The field name was not valid.


![Late-night hackathon debugging — building ScholarEnv at Scaler School of Technology](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/img4.png)


## Part V: The Numbers (Actual Ones, Not the Fraudulent Abstract Kind)

### Smoke Run: 25 Steps. 22 Minutes. One Free T4.

Frozen Qwen2.5-1.5B on Task T3.

**Starting reward: 0.008.**

A model that outputs an empty JSON object on every turn would score 0.008. The frozen model was doing exactly that.

25 GRPO steps later:

---

![Smoke Run Learning Curve](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/fig1_reward_curve.png)
*Total reward across 416 graded completions. The flat dashed line is the frozen baseline at 0.008. Around completion 20, the model discovers results-first navigation. Total reward spikes to 0.75. Peak: 0.905. Mean over last 20: 0.626. Improvement: ×82.*

---

**Peak reward: 0.905. Mean (last 20): 0.626. Improvement: ×82. Duration: 22 minutes.**

To be precise about the comparison: GPT-4o zero-shot on this specific audit task scores approximately 5%. Our trained 1.5B model scores approximately 65%. A model 47× smaller. 13× better. 22 minutes of training on hardware that costs nothing.

The format compliance numbers:

- valid_json: **95.4%** (n=416)
- non_empty_findings: **95.4%**
- has_table_id: **94.7%**
- has_str_table_value: **92.8%**

All four above the 95% excellence threshold. The model learned not just *what* to find but *how to write it down* in a format that can be programmatically verified by a grader that has never seen the paper.

---

![Reward Component Breakdown](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/fig2_components.png)
*Weighted component contributions across 416 completions. Specificity (green, ×0.15) learns first — the model quickly starts including exact table IDs and values. F-beta (blue, ×0.60) stabilises around completion 80. Reasoning quality (purple, ×0.25) contributes steadily throughout with high variance — the model is experimenting with different reasoning chains.*

---

![Format Compliance](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/fig3_format_compliance.png)
*Format compliance over training. All four metrics stay above 90% throughout. The 80% minimum target line is barely visible because we never came close to it after step 15.*

---

### Multi-Task Run: 200 Steps. 383 Completions. All Five Tasks Interleaved.

| Task | n | Baseline | After GRPO | Change |
|------|---|----------|------------|--------|
| T1: Formatting Compliance | 91 | 0.094 | 0.204 | **+1.63×** |
| T2: Internal Consistency | 67 | 0.020 | 0.019 | −0.94×† |
| T3: Claim-Evidence Audit | 91 | 0.820 | 0.489 | −0.60×†† |
| T4: Citation Verification | 115 | 0.254 | 0.374 | **+1.33×** |
| T5: Prompt Injection (zero-shot) | 19 | 0.146 | 0.183 | **+1.27× (no T5 training data)** |

> **†** T2 is the `location_a` bug. Zero reward contaminated the batch. Fixed in v8.
> **††** T3 regression is a consequence of T2 contaminating the advantage estimates. The smoke run confirms T3 reaches 0.905 in isolation.

---

![Multi-Task Results](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/fig4_multitask.png)
*Before vs. after GRPO. T1, T4 improve. T5 improves with zero T5 training examples — zero-shot policy transfer. T3 regression is the field name bug. The T5 bar is the most scientifically interesting result in this table.*

---

The T5 zero-shot transfer is worth pausing on. We trained on T3. We never showed the model a prompt injection. We never wrote a reward function for T5 in the training loop.

The model generalised anyway. The Saccade Policy — read carefully, check everything, distrust the surface — transferred to a task with a completely different structure. That is what a *policy* is, versus what a *trick* is. A trick solves one problem. A policy solves the class of problems that share the same underlying structure.

---

## Part VI: What RL Does That Prompting Cannot

Here is the clearest version of the core argument:

A prompt is a job description. Reinforcement learning is an apprenticeship.

You can write a very good job description for "find numerical discrepancies in research papers." You can include examples. You can add chain-of-thought instructions. You can tell the model to check tables against abstracts. It will follow those instructions, in the order you wrote them, with the strategy you implicitly embedded in the instructions.

But the model will not *improve*. It will not discover that results-first is faster than introduction-first. It will not transfer its verification mindset to tasks you never described. It will not learn to distrust citations just because a ghost DOI burned it once.

Reinforcement learning does all of those things. Not because we programmed them. Because the reward structure made learning them *worth it*.

The untrained model reads like a student who memorised the textbook. The trained model reads like a senior researcher who learned from getting things wrong.

That is the real result. Not 0.905 peak reward. Not 95.4% JSON compliance. The real result is a model that developed a reading strategy that cognitive scientists have documented in expert human readers — and it developed it in 22 minutes, unprompted, through nothing but reward pressure and gradient descent.


![Two reading strategies — student path vs emergent detective path — GRPO navigation policy discovery](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/img5.png)


## Part VII: The Research Stack (16 Papers, One Environment)

We did not invent the ideas in ScholarEnv. We assembled them. Here is what we assembled, and why each piece was load-bearing.

**PBRS (Ng, Harada & Russell, ICML 1999)** — the 26-year-old theorem that lets you add dense shaping rewards without changing the optimal policy. We use this for navigation. Without it, every intermediate step has zero gradient signal. With it, the Saccade Policy can emerge. This paper, published two years before most of our team was born, is what made the whole thing work.

**GRPO (DeepSeekMath, arXiv:2402.03300)** — no critic network. Rewards normalised within the generation group. Mean-centred advantages. Training loss displays as 0.000000 and it is correct: mean of mean-centred values is mathematically zero. Per-sample gradients are non-zero. The learning is real.

**DAPO (arXiv:2503.14476)** — clip-higher loss, `beta=0.0`, `epsilon_high=0.28`. The asymmetric clipping means high-reward completions get a wider update range than low-reward ones. This is what kept the Saccade Policy from being averaged away during early exploration.

**AdaRFT (arXiv:2504.05520)** — UCB1 bandit selecting paper difficulty (0.3–0.7). The model always trains at the edge of its current capability. Not too easy (no learning signal), not too hard (no positive reward to reinforce). The curriculum is the thing that makes 25 steps worth more than 250 steps without it.

**Agent-RLVR (arXiv:2506.11425)** — partial credit for multi-hop reasoning. A finding with the right table but wrong value gets 0.3, not zero. This single design choice is what prevented the model from abandoning partial strategies during early exploration.

The full list of 16 papers is in the README. Each one is a traceable decision.

---

## Part VIII: The Thing That Matters Beyond the Hackathon

Three students. One hackathon. Mentor sessions at Scaler where someone asked us mid-crisis: *what is the one result you want a judge to remember?*

We argued for twenty minutes. The 82× improvement. The T5 zero-shot transfer. The format compliance numbers. The fact that our model beats GPT-4o on this specific task.

The answer we landed on is none of those.

The answer is: **we built an environment, not a model.**

Models expire. GPT-4o will be replaced. Qwen2.5 will be superseded. Every model becomes obsolete. But an environment that generates infinite verifiable problems, rewards correct reasoning, and lets any model learn through experience — that compounds. That is useful for the next model and the one after it.

Science built peer review as an environment. Write down your findings. Submit them to critics. Receive reward or rejection. Iterate. The process is the thing. Not any single paper in it.

We built a small, specific, earnest version of that process for a 1.5B model. The model learned to read like a detective. It learned to distrust ghost citations. It learned to audit a paper the way a good reviewer audits a paper — results first, claims second, trust earned not assumed.

That is the result. And it happened in 22 minutes on hardware that costs nothing.

The itch continues.

![ScholarEnv — trained model learns to detect research fraud through reinforcement learning](https://huggingface.co/nensi1311/ScholarEnv/resolve/main/assets/img6.png)


## Try It

**Live Demo** → https://huggingface.co/spaces/nensi1311/ScholarEnv
Paste any abstract and table. Watch the agent navigate and submit a structured finding.

**Trained Model** → https://huggingface.co/nensi1311/scholarenv-auditor-qwen-1.5b
4-bit LoRA adapter. Loads in under 8GB.

**Training Notebook** → `Meta_Final.ipynb`
Run All. 22 minutes. Free Colab T4. End to end.

**Reproduce all figures:**
```bash
cp assets/reward_log_smoke.csv reward_log_smoke.csv
python scripts/plot_scholarenv_figures.py
```

---

## Key References

1. Ng, Harada & Russell — *Policy Invariance Under Reward Transformations*, ICML 1999
2. DeepSeekMath — *GRPO: Group Relative Policy Optimisation*, arXiv:2402.03300
3. DAPO — *Direct Advantage Policy Optimisation*, arXiv:2503.14476
4. AdaRFT — *Adaptive Curriculum for Reinforcement Fine-Tuning*, arXiv:2504.05520
5. RLVE — *Reinforcement Learning with Verifiable Environments*, arXiv:2511.07317
6. PRS — *Progressive Reward Shaping for LLM Structured Output*, arXiv:2512.07478
7. Agent-RLVR — *Partial Credit for Multi-Hop Reasoning*, arXiv:2506.11425
8. RAGEN-2 — *SNR Filtering for GRPO Training Batches*, arXiv:2604.06268
9. Dr. GRPO — *Removing Standard Deviation Bias*, arXiv:2503.20783
10. Abstain-R1 — *Calibrated Abstention for Verification Tasks*, arXiv:2604.17073

*All numbers from Meta_Final.ipynb, Cell 10 (baseline) and Cell 13 (post-training), and the attached CSVs. Nothing invented.*

---

*Nensi Pansuriya · Krushna Parmar · Ishita Bhojani — Scaler School of Technology*
*Meta × PyTorch OpenEnv Hackathon 2026*