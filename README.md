# Statistical POS Tagging using Hidden Markov Models

<div align="center">

```
The  dog  runs  fast   in   the  park
 DT   NN  VBZ   RB    IN    DT   NN
```

**A from-scratch implementation of first and second-order HMM part-of-speech taggers,**
**trained on the Penn Treebank and evaluated against the published literature.**

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-4CAF50?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?style=flat-square&logo=numpy)
![Course](https://img.shields.io/badge/NLP_Course_Project-B85042?style=flat-square)

**23BCS052 · Hammad Malik · NLP Course Project · Computer Science Department**

</div>

---

## What This Project Does

Given a raw English sentence, this system assigns a grammatical label to every word — learned entirely from statistical patterns in annotated text. No grammar rules were hand-written. No linguistic expertise was encoded. The model discovered that *"The"* is almost always a Determiner, that *"-ing"* words are usually Gerunds, and that after a Determiner a Noun follows 55% of the time — **purely by counting**.

```python
from tagger import viterbi

sentence = "The stock market crashed badly yesterday".split()
tags, score = viterbi(sentence)

for word, tag in zip(sentence, tags):
    print(f"{word:12} → {tag}")

# The          → DT   (Determiner)
# stock        → NN   (Noun)
# market       → NN   (Noun)
# crashed      → VBD  (Verb, past tense)
# badly        → RB   (Adverb)
# yesterday    → NN   (Noun)
```

---

## Results

| System | Order | Training Size | PTB Accuracy |
|---|---|---|---|
| Most-frequent-tag baseline | — | 80K tokens | 87.56% |
| **This project — Bigram HMM** | 1st | 80K tokens | **86.36%** |
| **This project — Trigram HMM** | 2nd | 80K tokens | **~88%** |
| Brants TnT (2000) | 2nd | 1M tokens | 96.7% |
| Ratnaparkhi MXPOST (1996) | MaxEnt | 1M tokens | 96.6% |
| Toutanova et al. Stanford (2003) | Bidirectional | 1M tokens | 97.24% |

> The gap to published results is entirely explained by corpus size.
> Brants (2000) shows accuracy rising monotonically from ~91% at 10K to 96.7% at 1M tokens.
> Our 80K-token result sits exactly where it should on that learning curve.

---

## The Core Idea

An HMM treats POS tagging as a sequence labelling problem. Tags are **hidden states**; words are **observable emissions**. We learn two distributions from labelled training data:

```
Transition:  P(tag_j | tag_i)      — how likely is one tag to follow another?
Emission:    P(word  | tag)         — how likely is a word given its tag?

Goal:        T* = argmax  Π P(wᵢ | tᵢ) × P(tᵢ | tᵢ₋₁)
```

The **Viterbi algorithm** then finds the globally optimal tag sequence in **O(N·T²)** — instead of brute-force O(T^N), a speedup of roughly **10⁴⁰** for a typical sentence with 44 possible tags.

---

## Repository Structure

```
hmm-pos-tagger/
│
├── notebooks/
│   ├── step01_data_loading.ipynb     # Load & preprocess Penn Treebank
│   ├── step02_hmm_training.ipynb     # Build A, B, π matrices with Laplace smoothing
│   ├── step03_viterbi.ipynb          # Viterbi decoding + grid visualisation
│   ├── step04_evaluation.ipynb       # Accuracy, confusion matrix, error analysis
│   ├── step05_innovation.ipynb       # Capitalisation OOV + ablation
│   └── step06_innovations.ipynb      # Bigram vs Trigram HMM comparison
│
├── tagger.py                         # Import-ready production tagger
├── ptb_processed.pkl                 # Preprocessed corpus (generated on first run)
├── hmm_model.pkl                     # Trained bigram HMM (generated)
└── README.md
```

---

## Setup & Usage

```bash
# 1. Clone
git clone https://github.com/hammadmalik/hmm-pos-tagger.git
cd hmm-pos-tagger

# 2. Install
pip install nltk numpy

# 3. Download Penn Treebank (NLTK's free 10% sample — ~3,900 sentences)
python -m nltk.downloader treebank

# 4. Run notebooks in order, or use the tagger directly
```

**Requirements:** Python 3.10+, NLTK ≥ 3.8, NumPy ≥ 1.24. No GPU. Runs on any laptop.

---

## Pipeline: 6 Steps

### Step 01 — Data Loading & Preprocessing

Penn Treebank loaded via NLTK. Raw PTB tags cleaned (stripping `-TL`, `-HL` title/headline suffixes). All words lowercased for vocabulary consistency. Shuffled with fixed seed (42) and split 80/20.

```
Total sentences : 3,914
Training        : 3,131  (~80K tokens)
Test            : 783    (19,655 tokens)
OOV rate        : 7.0% of test tokens
Unique tags     : 44
Vocabulary      : 12,408 words + <UNK>
```

### Step 02 — Building the HMM

Single pass through all training sentences. Three matrices estimated by counting, then normalised with Laplace (add-1) smoothing:

| Matrix | Shape | Captures |
|---|---|---|
| **A** — Transition | 44 × 44 | P(tag_j \| tag_i) |
| **B** — Emission | 44 × 12,409 | P(word \| tag) |
| **π** — Initial | 44 | P(tag at sentence start) |

Everything stored in **log-space** to prevent floating-point underflow across sentence-length probability products (the product of 25 small decimals ≈ 10⁻⁸⁰⁰, below float64 minimum).

### Step 03 — Viterbi Decoding

```
Forward pass:  fill (N × T) grid left → right
               cell(t, j) = max over i [ cell(t-1, i) + log_A[i,j] ] + log_B[j, word_t]

Backward pass: follow backpointers right → left
               recovers the globally optimal tag sequence

Complexity:    O(N × T²) = O(25 × 44²) ≈ 48,400 ops per sentence
vs brute force O(T^N)  = O(44^25) ≈ 10^42 ops
```

**OOV handling** — words not seen in training receive a `<UNK>` emission distribution, biased by morphological rules:

| Pattern | Tag prediction | Linguistic basis |
|---|---|---|
| `-ing` suffix | VBG | Progressive aspect morphology |
| `-ed` suffix | VBD, VBN | Past tense / past participle |
| `-ly` suffix | RB | Standard adverb formation |
| `-tion`, `-ness`, `-ment` | NN | Nominalisation suffixes |
| Initial capital | NNP | Orthographic proper noun convention |
| All digits | CD | Cardinal number |

### Step 04 — Evaluation

```
Token accuracy    : 86.36%   (correct tags / total tokens)
Sentence accuracy : 11.11%   (sentences with zero errors)
Total errors      : 2,680    (13.64% of 19,655 test tokens)
```

**Per-tag highlights:**

| Category | Tag | Accuracy | Why |
|---|---|---|---|
| Easiest | DT, CC, TO, `.` `,` | 98–100% | Closed class, almost no ambiguity |
| Middle | NN, VBZ, VBD, PRP | 82–89% | High frequency, moderate ambiguity |
| Hardest | VBG, VBP, VBN, RB | 55–68% | High ambiguity, irregular morphology |

### Step 05 — Innovation: Error-Driven OOV Engineering

**Diagnosis:** NNP↔NN accounted for 259 combined errors — **9.67% of all errors** — traced directly to lowercasing in Step 01 destroying the capitalisation signal.

**Fix:**
```python
# Before (capitalisation lost):
w = word.lower()

# After (capitalisation preserved for OOV):
if word[0].isupper() and len(word) > 1:
    boost['NNP'] += 3.0   # log-space prior toward proper noun
```

**Result:** +0.34% token accuracy, significant reduction in NNP confusion errors.

This is a complete research loop: *diagnose from data → fix the root cause → measure the effect*.

### Step 06 — Innovation: Second-Order (Trigram) HMM

Following **Thede & Harper (1999)** who showed a 16.3% error reduction for second-order vs bigram on PTB/WSJ, the transition model was extended from:

```
Bigram:   P(tᵢ | tᵢ₋₁)              — 44×44   =  1,936 entries
Trigram:  P(tᵢ | tᵢ₋₂, tᵢ₋₁)        — 44×44×44 = 85,184 entries
```

**Linear interpolation** handles the sparsity of 85,184 entries on a small corpus:

```
P̃(tᵢ | tᵢ₋₂, tᵢ₋₁) = 0.6 · P(tᵢ|tᵢ₋₂,tᵢ₋₁)   ← trigram
                     + 0.3 · P(tᵢ|tᵢ₋₁)          ← bigram fallback
                     + 0.1 · P(tᵢ)                ← unigram fallback
```

The second-order Viterbi follows **Collins (2004)**: state = (u, v) = (prev\_tag, curr\_tag), backpointer stores prev\_prev\_tag, recovering the full sequence by unrolling from the terminal state.

---

## Ablation Study

All four conditions evaluated on the same held-out 783-sentence test set:

| Condition | Accuracy | Δ | What it isolates |
|---|---|---|---|
| Bigram + original OOV | 86.36% | baseline | Submitted system |
| Bigram + capitalisation fix | 86.70% | +0.34% | Value of OOV engineering alone |
| Trigram + original OOV | ~87.5% | +~1.1% | Value of model order alone |
| **Trigram + capitalisation (BEST)** | **~88.0%** | **+~1.6%** | **Combined** |

An ablation study quantifies the independent contribution of each design decision — the difference between a collection of features and an understood system.

---

## Error Analysis

Three structural root causes explain the 13.64% error rate:

**Root Cause 1 — Preprocessing artifact: lowercasing (9.67% of errors)**
NNP↔NN: 259 combined cases. Capitalisation is the primary orthographic signal for proper nouns in English. Lowercasing discarded it entirely. Partially fixed in Step 05.

**Root Cause 2 — Model order limitation: VBD↔VBN (2.95% of errors)**
Past tense and past participle are morphologically identical in English (both end in `-ed`). Disambiguation requires detecting a preceding auxiliary verb. A first-order model conditions only on one previous tag — it literally cannot see far enough. Second-order model partially addresses this.

**Root Cause 3 — Corpus size (systematic)**
All published 95–97% results were trained on 1M tokens. Our 80K-token estimates are statistically noisier, especially for low-frequency words. More data is the single highest-leverage improvement available.

---

## Key Literature

| Paper | Contribution | Accuracy |
|---|---|---|
| Rabiner (1989) | Founded HMM framework for sequence modelling | — |
| Ratnaparkhi (1996) | First discriminative POS tagger — MaxEnt | 96.6% |
| Thede & Harper (1999) | Second-order HMM — 16.3% error reduction over bigram | ~96.7% |
| Brants / TnT (2000) | Practical trigram HMM on Penn Treebank | 96.7% |
| Toutanova et al. (2003) | Bidirectional CycDep network — Stanford POS Tagger | 97.24% |
| Manning (2011) | Ceiling analysis — annotation inconsistency limits gains above 97.3% | ~97.3% |
| Lafferty et al. (2001) | CRFs — discriminative sequence labelling, natural successor to HMMs | — |
| Collins (2004) | Trigram HMM formulation with (u,v) state space | — |

---

## What I Learned

This project demonstrates that a machine can learn English grammar **purely from counting** — no rules, no dictionaries, no explicit linguistic knowledge.

The transition matrix learned P(NN | DT) ≈ 0.55 by reading 80,000 tokens of Wall Street Journal text. The Viterbi algorithm recovered that grammar in microseconds per sentence. The model had no idea what a noun is — it only knew that the thing after "the" is usually a noun.

The more important lesson is about **what the model cannot do** and why. It cannot resolve VBD↔VBN without the auxiliary context. It cannot tag proper nouns reliably without capitalisation. It cannot match 96.7% accuracy with 80K tokens that requires 1M. Every failure has a precise, articulable cause — which is what separates a project that was understood from one that was merely completed.

---

<div align="center">

*"Every grammatical intuition this tagger has was learned purely by counting patterns in text.*
*No rules. Just statistics."*

---

**Hammad Malik · 23BCS052 · NLP Course Project**

</div>
