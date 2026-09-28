"""
O9 -- pure decision logic for the target-based repurposing arm.

Everything here is deterministic and network-free so it can be unit tested. The direction
rule, the reference filter and the pharmacokinetic pseudo-target exclusion are load-bearing
for the paper's claims, so they live in one tested place rather than being reimplemented in
each pipeline script.
"""

INHIBITORY = {"inhibitor", "antagonist", "blocker", "negative modulator",
              "inverse agonist", "suppressor", "negative allosteric modulator",
              "inhibitory allosteric modulator"}
ACTIVATORY = {"activator", "agonist", "positive modulator", "inducer", "stimulator",
              "partial agonist", "positive allosteric modulator", "opener"}

# "fda" is ChEMBL's actual mechanism_refs ref_type value for an FDA label citation (verified
# against live ChEMBL data); "fda label" is kept too in case a differently-cased/spelled source
# ever emits it. Fix round 2, 2026-09-11: the original set had "fda label" only, which silently
# rejected every ChEMBL ref_type=="FDA" reference (241 seen, 0 kept in the first production run)
# and reported drugs whose only primary citation was an FDA label as MECHANISM UNSUPPORTED -- a
# false claim. Deliberately NOT widened beyond this: EMA, KEGG, InterPro, UniProt, ISBN, Expert,
# BNF, DOI, PMC and Wikipedia all stay rejected -- this is a typo fix, not a rule change.
ACCEPTED_REF_TYPES = {"pubmed", "dailymed", "fda label", "fda", "clinicaltrials"}

PK_PSEUDOTARGETS = {"CYP3A4", "CYP2C19"}


def direction_verdict(gene_dir, action_types):
    """MATCH when the drug's action opposes the gene's observed dysregulation."""
    direction = str(gene_dir).strip().upper()
    if direction not in ("UP", "DOWN"):
        return "UNKNOWN"
    acts = {str(a).strip().lower() for a in (action_types or []) if str(a).strip()}
    if not acts:
        return "UNKNOWN"
    has_inh = bool(acts & INHIBITORY)
    has_act = bool(acts & ACTIVATORY)
    if not has_inh and not has_act:
        return "UNKNOWN"
    # A drug annotated as BOTH activator and inhibitor matches either direction
    # because some targets like TRPV4 genuinely carry both annotations in DGIdb.
    if direction == "UP" and has_inh:
        return "MATCH"
    if direction == "DOWN" and has_act:
        return "MATCH"
    return "WRONG DIRECTION"


def accepted_refs(refs):
    """Keep only primary reference types. ChEMBL mechanism_refs includes Wikipedia."""
    out = []
    for r in refs or []:
        if str(r.get("ref_type", "")).strip().lower() in ACCEPTED_REF_TYPES:
            out.append(r)
    return out


def is_pk_pseudotarget(symbol):
    return str(symbol).strip().upper() in PK_PSEUDOTARGETS


def consensus_label(platforms):
    p = sorted({str(x) for x in (platforms or []) if str(x).strip()})
    if len(p) >= 2:
        return "CONSENSUS"
    if len(p) == 1:
        return "SINGLE:%s" % p[0]
    return "NONE"


# Salt/hydrate/form suffix stripping, moved here from o9d_drug_annotate.py (fix round 3,
# 2026-09-11) so a single shared, tested copy backs both o9d's annotation pass and o9e's
# salt-form candidate merge -- two independent copies of a correctness-critical exclusion
# list is one future edit away from silent drift. Matched case-insensitively against the
# last whitespace-delimited token (parentheses stripped first, so "AMPICILLIN (ANHYDROUS)"
# matches the same as "AMPICILLIN ANHYDROUS"). "alfa"/"beta" are deliberately NOT in this
# set: they are biologic product designators, not salt/hydrate forms -- epoetin alfa and
# epoetin beta are genuinely different drugs, so stripping them risks matching a different
# product's identity (o9d fix round 2, Finding 2, the THROMBIN ALFA case).
SALT_SUFFIXES = {
    "hydrochloride", "dihydrochloride", "hydrobromide", "sulfate", "sulphate", "phosphate",
    "maleate", "dimaleate", "fumarate", "tartrate", "bitartrate", "citrate", "mesylate",
    "besylate", "tosylate", "ditosylate", "succinate", "acetate", "sodium", "disodium",
    "potassium", "calcium", "magnesium", "chloride", "bromide", "nitrate", "oxalate",
    "malate", "lactate", "gluconate", "pamoate", "palmitate", "stearate", "valerate",
    "propionate", "benzoate", "carbonate", "bicarbonate", "anhydrous", "hydrate",
    "monohydrate", "dihydrate", "trihydrate",
}


def strip_base_name(name):
    """Strip trailing salt/hydrate/form qualifiers repeatedly, e.g. "X SODIUM HYDRATE" ->
    "X SODIUM" -> "X". Returns the input unchanged if no trailing qualifier matched (the
    caller compares the return value to the input to decide whether a retry/merge is
    warranted, rather than this function signalling it separately). Prefix-form qualifiers
    ("ANHYDROUS TACROLIMUS") are NOT stripped -- only trailing ones."""
    words = name.split()
    while words:
        last = words[-1].strip("()").lower()
        if last in SALT_SUFFIXES:
            words.pop()
        else:
            break
    stripped = " ".join(words).strip()
    return stripped if stripped else name


def oral_verdict(chembl_oral, openfda_routes):
    """Oral only if both sources agree, where both exist."""
    routes = None
    if openfda_routes is not None:
        routes = {str(r).strip().upper() for r in openfda_routes if str(r).strip()}
    # Treat empty or whitespace-only route list as no data
    if routes is not None and not routes:
        routes = None
    fda_oral = None if routes is None else ("ORAL" in routes)
    if chembl_oral is None and fda_oral is None:
        return "UNKNOWN"
    if chembl_oral is None:
        return "ORAL" if fda_oral else "NOT_ORAL"
    if fda_oral is None:
        return "ORAL" if chembl_oral else "NOT_ORAL"
    if bool(chembl_oral) == bool(fda_oral):
        return "ORAL" if chembl_oral else "NOT_ORAL"
    return "DISAGREE"
