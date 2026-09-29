# RNAseq_pool_MAFLD

Code and derived data to reproduce the analysis in:

> Alshehade SA, Alshawsh MA, Shukri NHH, Bitar AN. *An eight-cohort meta-analysis of MASLD liver
> transcriptomes: disease severity, not the obesity comparator, explains between-cohort
> disagreement.* Manuscript in submission (2026).

The repository recomputes the paper's results from per-cohort RNA-seq data and prints them to
the terminal. Every number is compared with the value reported in the paper and marked
**PASS** or **DIFF**. No figures are produced.

## What is reproduced

| Stage | Paper | What it computes |
|---|---|---|
| 1 Harmonisation | 2.4 | Eight cohorts (605 samples: 111 control, 494 case) on one Ensembl gene namespace; GSE183229's integer counts recovered exactly from its deposited log2(normalised + 1) matrix; the 15,800-gene count and log2 CPM matrices |
| 2 Differential expression | 2.5, Table 1, Table S3 | Per-cohort DESeq2 (PyDESeq2), case vs control, all eight cohorts |
| 3 Meta-analysis | 2.6, 3.4, Table 2, Table S6 | DerSimonian-Laird random-effects pooling; the 73-gene signature (36 up, 37 down); the Hartung-Knapp-Sidik-Jonkman sensitivity |
| 4 Robustness | 3.3, 3.4, Table 3, Figures S4, S6 | Paule-Mandel and REML refits (57 estimator-invariant genes), leave-one-cohort-out pooling, single-cohort replication |
| 5 Batch structure | 3.2, Figure 2 | PCA variance partition, cohort vs disease (eight cohorts, n = 605) |
| 6 Obesity | 3.6, Figure 5A-B | Three contrasts inside GSE126848: MASLD vs lean, MASLD vs obese, obese vs lean |
| 7 Severity | 3.6, Figure 5C | Fibrosis strata against each cohort's own controls |
| 8 Classifier | 2.7, 3.5, Figure 4, Table S4 | Leave-one-cohort-out elastic net and gradient boosting, feature selection inside every fold, discrimination and calibration, and nested-tuned internal cross-validation, each summarised over ten tuning seeds |
| 9 SHAP and negative control | 3.5, Figure 4C | SHAP importance of the elastic net averaged over the ten seed refits (full background); predicting the source cohort from the same features |
| 10 Prioritised core | 2.8, 3.7, Table 4, Table S7 | Three evidence lines, tiers and the twelve-gene validation panel |
| 11 Repurposing | 2.9.2, 3.8, Table 5, Figure 7A, Table S8 | Approval, oral, hepatic, cardiac, direction, consensus and mechanism gates, re-run on the dated query snapshot |

A full run (ten tuning seeds) takes about 15 minutes on a laptop; `--quick` (the pre-set seed only, checked against the seed-1 sensitivity values the paper reports) takes about 7. On the analysis environment every check passes.

## Quick start

```bash
git clone https://github.com/salahalsh/RNAseq_pool_MAFLD.git
cd RNAseq_pool_MAFLD
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Options:

```
python run_all.py --quick               # classifier on the pre-set seed only (about 7 minutes)
python run_all.py --skip-classifier     # stages 1-7 and 11 only (about 3 minutes)
python run_all.py --save                # also write every result table to results/
python run_all.py --use-recomputed-de   # carry this run's DESeq2 tables downstream
```

The exit code is 0 when every check passes.

## Environment

The analysis was run with Python 3.14 and the versions pinned in `requirements.txt`:

- pandas 2.3.3
- numpy 2.4.6
- scipy 1.18.0
- scikit-learn 1.9.0
- PyDESeq2 0.5.4
- LightGBM 4.7.0
- SHAP 0.52.0

Seeds:

- `1` to `10` for model tuning and cross-validation; every classifier quantity is the median over these ten seeds (the paper's decision R2-09), and seed `1` alone is reported as a sensitivity.
- `20260903` for the randomised PCA solver.

## Data (`data/`)

### `cohorts/<GSE>/`

Three files for each of the eight analysed cohorts.

| File | Content |
|---|---|
| `<GSE>_counts.csv.gz` | The deposited GEO matrix, restricted to the analysed samples. The first column is the identifier as deposited, the second is the harmonised Ensembl gene ID (blank where unmappable), then one column per GSM. Values are raw counts, except **GSE183229, which deposits log2(DESeq2-normalised count + 1)**; `masld_pool.io.recover_counts` recovers its integer counts exactly and checks the recovery (every value integer; DESeq2 size factors match). |
| `<GSE>_metadata.csv` | One row per sample. `group_class` (control / case) is the label used in every contrast. `analysis_set` is `primary` for the 605 samples of the primary analysis, and `obesity_stratum_only` for the 12 obese GSE126848 samples used only in the obesity contrasts. Histology, demography, SRA and GEO link columns are as harvested from GEO. |
| `<GSE>_DE.csv.gz` | The per-cohort differential-expression table exactly as it entered the meta-analysis: `baseMean`, `log2FC`, `lfcSE`, `stat`, `pvalue`, `padj`. |

| Cohort | Control / case | Scale | DE method |
|---|--:|---|---|
| GSE126848 | 14 / 31 (+12 obese, obesity stratum only) | counts | DESeq2 |
| GSE135251 | 10 / 206 | counts | DESeq2 |
| GSE162694 | 31 / 112 | counts | DESeq2 |
| GSE183229 | 9 / 7 | log2(normalised + 1), counts recovered | DESeq2 |
| GSE185051 | 5 / 52 | counts | DESeq2 |
| GSE260666 | 6 / 10 | counts | DESeq2 |
| GSE268360 | 6 / 12 | counts (RSEM expected counts) | DESeq2 |
| GSE281797 | 30 / 64 | counts | DESeq2 |

### `reference/`

| File | Why it is needed |
|---|---|
| `gene_universe_counts.tsv` | The 15,800 genes shared by the eight cohorts. It is recomputed and checked. |
| `gene_universe_logexpr.tsv` | The same 15,800 genes; library sizes are taken over them, so log2 CPM is reproduced exactly. |
| `sample_order.tsv` | Sample order of the analysed matrix. It fixes which samples share a cross-validation fold. |
| `string_top500_hub_degree.tsv` | Snapshot (29 September 2026) of the STRING v12.0 network (score ≥ 400, version-pinned address) of the top 500 pooled genes. It supplies the degree ≥ 5 evidence line. |
| `gene_symbols_hgnc.tsv` | Ensembl to HGNC symbol map. |

### `repurposing_snapshot/`

The drug-target edges, network neighbours, drug annotations and mechanism references queried on 29 September 2026 (STRING pinned to v12.0). Files named `expected_*` are the published gate outputs used for comparison. `mechanism_concordance_curated.tsv` is the hand-curated mechanism at each candidate's nominating target, with its source.

`data/MANIFEST.csv` lists every file with its size and SHA-256 checksum.

## Numerical reproducibility

- **Differential expression.** Stage 2 re-runs every cohort from its counts and compares the result with the shipped table: the same genes are tested in every cohort and the significant-gene counts agree to within one. A few genes can differ in standard error in the last digits because of PyDESeq2's numerical dispersion fitting, so downstream stages use the shipped tables by default; `--use-recomputed-de` propagates the recomputed tables instead.
- **Other environments.** Other Python and library versions can change DESeq2 and model-fitting results in the last digits.

## Not re-run here

These steps depend on live external services or on tools outside this repository. Their dated outputs are in the paper's supplementary datasets.

- **Search and screening:** GEO and ArrayExpress searches and screening (Dataset D6).
- **recount3 check:** the quantification-pipeline check (Figure S5).
- **STRING:** enrichment analysis, and the neighbour expansion of the repurposing arm. The neighbour lists are in the snapshot.
- **Drug databases:** Open Targets, DGIdb, ChEMBL, Pharos, BindingDB, Probes&Drugs, TTD and openFDA queries (TTD and Probes&Drugs returned no usable records). Only their results are in the snapshot.
- **Superseded screen:** the signature-reversal screen (LINCS L1000CDS²), the ADMET models and the cardiac label curation.
- **Other sensitivities:** the inferred-sex checks, the prevalence-independent standardisation of the classifier and the nine-cohort pooling; their outputs are in the supplementary datasets.

## Licence

- **Code:** MIT, see `LICENSE`.
- **Cohort data:** derived from public GEO submissions (accessions above). Please cite the original studies when you reuse them.
- **Snapshot files:** records retrieved from third-party resources keep the terms of their sources. These include ChEMBL (CC BY-SA 3.0), Open Targets (CC0), STRING (CC BY 4.0), BindingDB, DGIdb and the sources it aggregates, Pharos, and openFDA. They are provided only so that the published gates can be re-run.

## Contact

Corresponding authors: Mohammed Abdullah Alshawsh (mohammed.alshawsh@monash.edu) and
Salah A. Alshehade (salahshehade@unisza.edu.my).

## Citation

See `CITATION.cff`. Please cite the paper once it is published.
