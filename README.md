<div align="center">

# 🧬 M5 — cfDNA Glioma Subtype Classifier

**Pleiades-Inspired HAT · Part of [NeuroSight](../)**

*End-to-end clinical AI for GBM treatment monitoring and early detection*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](../LICENSE)

</div>

---

## The Clinical Problem

Glioblastoma (GBM) has a 15-month median survival. That number has not meaningfully changed in 20 years. One of the most critical unmet needs is an early, non-invasive detection tool — something that can flag GBM from a blood draw before symptoms escalate, and distinguish it from lower-grade glioma subtypes that carry very different prognoses and treatment paths.

Plasma cell-free DNA (cfDNA) carries methylation signatures of its cell of origin. In glioma patients, even the vanishingly small tumor-derived fraction of plasma cfDNA carries distinctive epigenomic marks. M5 builds a classifier on this signal.

---

## The Core Challenge: No GBM cfDNA Dataset Exists

This is the central engineering problem of M5. Unlike established cancer cfDNA tasks (lung, colorectal), **no publicly available plasma cfDNA dataset exists for glioblastoma subtype classification**. Real GBM plasma cfDNA is extraordinarily rare due to the blood-brain barrier — the median tumor-derived fraction in glioma plasma is ~3.1 × 10⁻⁵ (~0.003%), orders of magnitude lower than most solid tumors.

The solution: construct a biologically-grounded synthetic cfDNA dataset from first principles, anchored to real patient methylation data and the hg38 reference genome.

---

## Synthetic Dataset Construction

Every fragment in this dataset is:
- Anchored to real hg38 genomic coordinates
- Derived from real patient beta values (TCGA / GEO array data)
- Sized according to nucleosome biology (tumor cfDNA fragments shorter than healthy)
- CpG-enriched to simulate post-cfMeDIP-seq immunoprecipitation
- Restricted to validated Differentially Methylated Regions only
- Tokenized in Pleiades-compatible format

### Step 1 — Real Methylation Data as the Foundation

Four publicly available methylation array datasets provide the biological ground truth:

| Dataset | Source | Samples | Role |
|---|---|---|---|
| **TCGA-GBM** | GDC Portal · Illumina 450K | ~423 | IDH-wildtype GBM beta values |
| **TCGA-LGG** | GDC Portal · Illumina 450K | ~453 | LGG: ~282 Astrocytoma IDH-mut + ~171 Oligodendroglioma |
| **GSE161944 + GSE50022** | GEO · EPIC/450K | ~48 | DMG H3K27M (adult + pediatric) |
| **GSE122126** | GEO · Healthy plasma cell-type atlas | — | Healthy plasma baseline (Moss et al. 2018) |

These datasets give us **beta values (0.0–1.0) at ~450,000 CpG positions** across the genome — methylation levels per locus, per patient. They do not give us DNA sequence. That comes next.

### Step 2 — Filling Sequence from hg38

The 450K array tells us methylation levels at CpG positions. The **hg38 reference genome** provides the actual A/T/G/C nucleotide sequence at every genomic coordinate.

For every CpG site, beta values are converted to methylation tokens using the Pleiades threshold scheme:

```
beta > 0.70  →  <m>   (methylated, high confidence)
beta < 0.30  →  <um>  (unmethylated, high confidence)
0.30 ≤ beta ≤ 0.70  →  probabilistic sampling
                         P(<m>) = beta,  P(<um>) = 1 − beta
```

Non-CpG positions (A, T, G, C) are taken directly from hg38 at the same genomic coordinates. This produces a complete, methylation-aware DNA string at every position — ready for tokenization.

### Step 3 — DMR-First Window Selection

Sequence is not extracted from random genome positions. The pipeline first identifies the **Top 300 Differentially Methylated Regions (DMRs)** per glioma subtype using limma-trend statistical testing (moderated t-statistic, delta-beta ≥ 0.3, FDR < 0.01, ≥ 2 CpGs per region). Only 300bp windows overlapping these DMRs are used.

Windows are restricted to biologically relevant regions: CpG islands, CpG shores (±2kb), CpG shelves (±2kb), and FANTOM5 enhancers. Open-sea regions are downsampled to 5% — matching the Pleiades pretraining protocol.

This DMR-first approach ensures every synthetic fragment carries discriminative signal, not background noise.

### Step 4 — Nucleosome-Aware Fragmentation

cfDNA fragmentation is not random — it is governed by nuclease cleavage at nucleosome linker regions. Tumor-derived cfDNA fragments are systematically **shorter** than healthy cfDNA:

| Parameter | Healthy Plasma | Glioma / Tumor cfDNA |
|---|---|---|
| Mononucleosomal peak | ~166 bp | ~134–144 bp (shorter) |
| Dinucleosomal peak | ~320 bp | ~320 bp |
| Fragment range | 100–400 bp | 100–400 bp |
| Sub-nucleosomal periodicity | ~10 bp | Enriched below 150 bp |

Fragment endpoints are biased toward GC-rich linker boundaries, reflecting DNASE1L3 cleavage preference (~42.9% of plasma cfDNA fragmentation). This encodes tumor vs. healthy signal at the fragmentomics level, not just at the methylation level.

### Step 5 — CpG Content Filtering (cfMeDIP Simulation)

cfMeDIP-seq immunoprecipitates methylated DNA fragments. Fragments with zero methylated CpGs would not be captured. Every fragment in the dataset must contain **2–8 methylated CpGs**, with fragments outside this range discarded. This simulates the enrichment that real cfMeDIP-seq provides and matches the Nassiri et al. (2020) pipeline that achieved AUC = 0.99 for glioma detection.

### Step 6 — ctDNA Fraction Mixing

Raw plasma cfDNA is dominated by healthy cells. The cell-type composition of healthy plasma (Moss et al. 2018) is used for mixing:

| Cell Type | Healthy Fraction |
|---|---|
| White blood cells | ~55% |
| RBC progenitors | ~30% |
| Vascular endothelium | ~10% |
| Hepatocytes | ~1% |
| Other | ~4% |

Tumor-derived fraction in raw plasma: **~0.003%** (median 3.1 × 10⁻⁵). The pipeline models **post-cfMeDIP-seq enriched signal**, which concentrates the tumor fraction substantially above raw plasma levels.

### Step 7 — Pleiades-Compatible Tokenization

Each synthetic patient is serialized as a JSON file with Pleiades-format tokens:

```json
{
  "label": 1,
  "subtype": "GBM",
  "ctdna_fraction": 0.000031,
  "regions": {
    "region_0": {
      "chromosome": 2,
      "genomic_start": 89768000,
      "fragments": [
        {
          "tokens": ["<cfdna>", "A", "T", "<m>", "C", "G", "...", "</cfdna>"],
          "fragment_length": 141,
          "cpg_count": 4,
          "strand": "+"
        }
      ]
    }
  }
}
```

The `genomic_start` coordinate maps directly to hg38. The coordinate `chr2:89,768,000` appears in Pleiades Supplementary Table S2 as a validated cfDNA generation region.

### Dataset Scale

| Label | Class | Source | Target Samples |
|---|---|---|---|
| 0 | Healthy | GSE122126 | 1,000+ |
| 1 | GBM IDH-wildtype | TCGA-GBM | 1,000+ |
| 2 | Astrocytoma IDH-mut | TCGA-LGG | 1,000+ |
| 3 | DMG H3K27M | GSE161944 + GSE50022 | 1,000+ |

500–1,000 fragments per patient, simulating post-cfMeDIP-seq sequencing depth.

---

## Architecture

M5 implements a three-tier Hierarchical Attention Transformer (HAT) directly inspired by the Pleiades architecture (Niki et al., Prima Mente 2025). The design is an independent reimplementation trained from scratch — not a fine-tuned checkpoint of the Pleiades foundation model.

### Architectural Lineage

| Property | Pleiades (Prima Mente 2025) | M5 / NeuroSight |
|---|---|---|
| **Approach** | Pretrained foundation model, fine-tuned on clinical cohorts | Trained from scratch on synthetic GBM cfDNA |
| **Parameters** | 90M / 600M / 7B | ~13M |
| **Pretraining data** | 1.9T tokens (whole-genome methylated DNA, 39 cell-type groups) | No pretraining — direct task training |
| **Hierarchy** | Fragment → Region (1kb) → Sample (HAT) | Fragment → Region (1kb) → Sample (HAT) |
| **Alignment Embeddings** | chr + millions + thousands + ones offsets | ✓ Same decomposition |
| **Token format** | `<cfdna>`, `A/T/C/G`, `<m>`, Alignment Embeddings | ✓ Same vocabulary |
| **Base model** | Autoregressive transformer decoder | Bidirectional transformer encoder |
| **Set model** | Hierarchical Attention Transformer (HAT) | ✓ Same HAT design |
| **Task** | Binary (AD/PD vs control); CToO; generative | 4-class glioma subtype classification |
| **Training setup** | 256 H100s (7B), 64 H200s (600M) | 2× Kaggle T4 GPUs (DDP) |
| **d_model** | 4096 (7B), 1280 (600M), 768 (90M) | 256 |
| **Base layers** | 42 (7B), 32 (600M), 12 (90M) | 6 |
| **Region layers** | 4 | 4 |
| **Sample layers** | 2 | 2 |

> Pleiades is a foundation model pretrained on 1.9T tokens of real human methylome data across 39 cell types, then fine-tuned on clinical disease cohorts. M5 is a ground-up reimplementation of the same hierarchical architecture, trained entirely on the synthetic GBM cfDNA dataset described above. The parameter and compute differential is intentional — M5 is a proof-of-architecture under severe resource constraints, not a reproduction of Pleiades at scale.

### Three-Tier Hierarchy

```
Plasma cfDNA sample
  │
  ├─ Fragment 1: <cfdna> A T <m> C G ... </cfdna>
  ├─ Fragment 2: <cfdna> G C <m> A T ... </cfdna>
  ├─ ...
  │
  ▼
Tier 0 — PleiadesBase (fragment level)
  Input:  token sequence + Alignment Embeddings (chr, M, K, ones)
  Model:  6-layer bidirectional transformer encoder, d_model=256
  Output: [CLS] embedding per fragment  →  shape (d_model,)
        │
        ▼
Tier 1 — PleiadesRegion (region level)
  Input:  all fragment [CLS] vectors within a 1kb genomic window
  Model:  4-layer transformer encoder, d_model=256
  Output: [CLS] embedding per region  →  shape (d_model,)
        │
        ▼
Tier 2 — PleiadesSample (sample level)
  Input:  all region [CLS] vectors for the patient
  Model:  2-layer transformer encoder + classification head
  Output: 4-class logits  →  Healthy / GBM / LGG / DMG H3K27M
```

### Alignment Embeddings

Each nucleotide in every fragment is embedded with its precise hg38 genomic coordinate, decomposed into four components — matching the Pleiades AE scheme exactly:

```
position → [P_chr, P_millions, P_thousands, P_ones]
         → [LearnedEmbedding(chr), LearnedEmbedding(M), LearnedEmbedding(K), LearnedEmbedding(ones)]
         → summed and added to token + position embeddings
```

This gives the model single-nucleotide resolution across all ~3.1 billion positions of hg38 without requiring prohibitively long context windows.

### Model Configuration

```python
Pleiades(
    d_model      = 256,
    n_head       = 8,
    d_ff         = 1024,
    base_layers  = 6,    # PleiadesBase  (L0)
    region_layers= 4,    # PleiadesRegion (L1)
    sample_layers= 2,    # PleiadesSample (L2)
    n_classes    = 4,
    dropout      = 0.1,
)
# Total parameters: ~13M
```

---

## Training

- **Hardware:** 2× Kaggle T4 GPUs, PyTorch DDP (`mp.start_processes`, NCCL backend)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler:** CosineAnnealingLR
- **Precision:** AMP (bfloat16 / float16 mixed)
- **Gradient checkpointing:** enabled on PleiadesBase encoder to reduce VRAM
- **Batch size:** 1 patient per GPU (each patient = variable number of fragments)
- **Loss:** CrossEntropyLoss (4-class)
- **Epochs:** 50 with best-checkpoint saving on val accuracy

---

## Repository Structure

```
m5_cfdna/
├── train_script.py          # Full DDP training loop
├── model.py                 # PleiadesBase, PleiadesRegion, PleiadesSample, Pleiades
├── dataset.py               # cfDNADataset, collate_fn, get_datasets
├── synthetic_pipeline/
│   ├── dmr_selection.py     # limma-trend DMR calling (Top 300 per class)
│   ├── hg38_extraction.py   # Sequence extraction at DMR coordinates
│   ├── beta_to_tokens.py    # Beta value → <m>/<um>/A/T/C/G tokenization
│   ├── fragmentation.py     # Nucleosome-aware fragment size sampling
│   ├── cpg_filter.py        # cfMeDIP simulation — enforce 2–8 CpGs per fragment
│   ├── mixing.py            # ctDNA fraction mixing with healthy baseline
│   └── serialize.py         # Pleiades-compatible JSON output
├── checkpoints/             # Saved model checkpoints
└── README.md
```

---

## Limitations

- **Synthetic training data** — the model has never seen real GBM plasma cfDNA. Performance on real clinical samples is unknown and will require validation.
- **Scale gap** — Pleiades 7B was pretrained on 1.9T tokens of real human methylome data. M5 is 13M parameters trained from scratch on synthetic data. The architectural design is validated; the learned representations are not.
- **4-class scope** — Healthy / GBM / LGG / DMG H3K27M. Finer GBM subtyping (MGMT status, TERT, EGFR amplification) is out of scope pending larger datasets.
- **Blood-brain barrier** — glioma ctDNA fraction in raw plasma (~0.003%) is the lowest of any tumor type. Even post-enrichment, the signal is weak. Real validation will require cfMeDIP-seq on actual GBM plasma cohorts.

---

## References

- Niki P. et al. "Human whole epigenome modelling for clinical applications with Pleiades." *Prima Mente*, 2025. [Preprint]
- Nassiri F. et al. "Detection and discrimination of intracranial tumors using plasma cell-free DNA methylomes." *Nature Medicine* 26, 1044–1047 (2020).
- Mouliere F. et al. "Enhanced detection of circulating tumor DNA by fragment size analysis." *Science Translational Medicine* 10, eaat4921 (2018).
- Moss J. et al. "Comprehensive human cell-type methylation atlas reveals origins of circulating cell-free DNA." *Nature Communications* 9, 5068 (2018).
- Loyfer N. et al. "A DNA methylation atlas of normal human cell types." *Nature* 613, 355–364 (2023).
- *NeuroSight Synthetic GBM cfDNA Generation Rulebook*, Arnav Mishra, 2025.
- *NeuroSight System Master Architecture and Build Documentation*, 2025 (`NeuroSight_Pleiades.docx`).

---

<div align="center">

**NeuroSight Pipeline**

[M1 — 3D Res-U-Net](../m1_segmentation) · [M2 — Fisher-KPP PINN](../m2_pinn) · [M3 — Progression Classifier](../m3_classifier) · [M4 — Clinical RAG](../m4_rag) · **M5 — cfDNA Classifier (this repo)**

*Orchestrated by the NeuroBio Agent*

</div>
