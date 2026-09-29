"""Fixed settings of the analysis. Every threshold here is the one used in the paper."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
COHORT_DIR = DATA / "cohorts"
REF_DIR = DATA / "reference"
SNAPSHOT_DIR = DATA / "repurposing_snapshot"
RESULTS = ROOT / "results"

# The eight cohorts of the primary analysis, in the order the harmonised matrix holds them.
COHORTS = ["GSE126848", "GSE135251", "GSE162694", "GSE183229",
           "GSE185051", "GSE260666", "GSE268360", "GSE281797"]
# GSE183229 deposits log2(DESeq2-normalised count + 1). Its integer counts are recovered
# exactly from that matrix (harmonise.recover_counts) and it is analysed as counts.
RECOVERED_COUNT_COHORTS = {"GSE183229"}

# Signature rule (fixed before pooling was run)
FDR = 0.05
LFC_MIN = 0.5
K_MIN = 4
LFC_STRICT = 1.0

# Per-cohort differential expression
MIN_COUNT = 10            # a gene is kept if >= min(n_control, n_case) samples reach this count

# Classifier
N_FEATURES = 500          # top genes by pooled p, re-selected inside every fold
SEED = 1                  # the pre-set seed (reported as a sensitivity)
SEEDS = tuple(range(1, 11))   # every classifier quantity is summarised over these ten seeds

# Prioritisation
SHAP_TOP = 200
HUB_MIN_DEGREE = 5
