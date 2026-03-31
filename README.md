# HMM-POS-Tagger
Statistical POS tagging using Hidden Markov Models - NLP Course
# Statistical POS Tagging using Hidden Markov Models

> NLP Course Project · 23BCS052 · Hammad Malik

---

## Overview

This project implements a **statistical Part-of-Speech (POS) tagger** using **Hidden Markov Models (HMMs)**. Given a raw English sentence, the tagger assigns a grammatical tag (Noun, Verb, Adjective, etc.) to every word — learned entirely from statistical patterns in a large annotated corpus.

The core idea: rather than hand-writing grammar rules, we *learn* the rules from data. The model captures two things:
- **How likely** a word is to appear given a tag (*emission probabilities*)
- **How likely** a tag is to follow another tag (*transition probabilities*)

At inference time, the **Viterbi algorithm** decodes the most probable tag sequence for any new sentence.

---

## Dataset

**Penn Treebank (PTB)** — LDC99T42  
~1 million words of Wall Street Journal text, manually annotated with 36 POS tags.  
Access via [LDC](https://catalog.ldc.upenn.edu/LDC99T42) or the NLTK corpus interface (`nltk.corpus.treebank`).

---

## Project Structure

```
hmm-pos-tagger/
│
├── data/                   # Corpus loading & preprocessing
│   └── loader.py
│
├── model/
│   ├── hmm.py              # HMM: transition & emission matrices
│   └── viterbi.py          # Viterbi decoding algorithm
│
├── evaluation/
│   └── metrics.py          # Accuracy, confusion matrix
│
├── notebooks/
│   └── exploration.ipynb   # Dataset analysis & visualizations
│
├── main.py                 # Train & evaluate the tagger
├── requirements.txt
└── README.md
```

---

## How It Works

### 1 · Training
The model counts tag-to-tag transitions and word-given-tag emissions across all training sentences, then normalizes them into probabilities. Laplace smoothing is applied to handle unseen combinations.

### 2 · Decoding (Viterbi)
For a new sentence, the Viterbi algorithm uses dynamic programming to find the tag sequence `t₁, t₂, ..., tₙ` that maximises:

```
P(t₁...tₙ | w₁...wₙ) ∝ ∏ P(wᵢ | tᵢ) × P(tᵢ | tᵢ₋₁)
```

### 3 · Unknown Words (OOV)
Words not seen during training are handled via morphological heuristics:
- Suffix `-ing` → VBG (gerund/present participle)
- Suffix `-ed` → VBD (past tense)
- Suffix `-ly` → RB (adverb)
- Suffix `-tion`, `-ness`, `-ment` → NN (noun)
- Starts with uppercase → NNP (proper noun)

---

## Results (Target)

| Metric | Score |
|--------|-------|
| Token-level Accuracy | ~95–96% |
| Baseline (Most-frequent tag) | ~92% |
| State-of-the-art (Neural) | ~97–98% |

---

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/hammadmalik/hmm-pos-tagger.git
cd hmm-pos-tagger

# Install dependencies
pip install -r requirements.txt
python -m nltk.downloader treebank

# Train and evaluate
python main.py
```

---

## Requirements

```
nltk>=3.8
numpy>=1.24
scikit-learn>=1.3   # for confusion matrix
matplotlib>=3.7     # for plots
```

---

## Tags Used (Penn Treebank Tagset)

| Tag | Description | Example |
|-----|-------------|---------|
| NN | Noun, singular | dog, city |
| NNS | Noun, plural | dogs, cities |
| NNP | Proper noun | London, John |
| VB | Verb, base form | run, eat |
| VBZ | Verb, 3rd person singular | runs, eats |
| VBD | Verb, past tense | ran, ate |
| VBG | Verb, gerund | running, eating |
| JJ | Adjective | big, beautiful |
| RB | Adverb | quickly, very |
| DT | Determiner | the, a, an |
| IN | Preposition/conjunction | in, on, with |
| PRP | Personal pronoun | he, she, they |
| MD | Modal verb | can, will, should |
| CC | Coordinating conjunction | and, but, or |

---

## References

- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition.* Proceedings of the IEEE.
- Marcus et al. (1993). *Building a Large Annotated Corpus of English: The Penn Treebank.* Computational Linguistics.
- Jurafsky & Martin. *Speech and Language Processing* (3rd ed.), Chapter 8.

---

*NLP Course Project — Computer Science Department*
