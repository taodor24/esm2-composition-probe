"""Fetch, label, and filter reviewed human proteins from UniProt."""
import os
import re
import sys
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/uniprot_human_reviewed.tsv"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream"
QUERY = "organism_id:9606 AND reviewed:true"
# protein_families: UniProt's curated family-membership string (e.g. "GPCR family"),
# used to build homology-aware groups for the family-confound check.
FIELDS = "accession,sequence,cc_subcellular_location,length,protein_families"


def _download():
    import requests

    os.makedirs("data/raw", exist_ok=True)
    try:
        resp = requests.get(
            UNIPROT_URL, params={"query": QUERY, "fields": FIELDS, "format": "tsv"}, timeout=300
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        sys.exit(f"UniProt download failed: {e}. Cannot proceed without real data.")
    with open(RAW_PATH, "w") as f:
        f.write(resp.text)


def _label(location_text):
    """Nucleus xor {Cell membrane, Plasma membrane}; else ambiguous/neither."""
    if not isinstance(location_text, str) or not location_text:
        return "neither"
    nucleus = re.search(r"\bNucleus\b", location_text) is not None
    membrane = re.search(r"\b(Cell membrane|Plasma membrane)\b", location_text) is not None
    if nucleus and membrane:
        return "ambiguous"
    if nucleus:
        return "nucleus"
    if membrane:
        return "membrane"
    return "neither"


def load_data():
    if not os.path.exists(RAW_PATH):
        print("Downloading UniProt data (not cached)...")
        _download()
    else:
        print(f"Using cached raw data at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, sep="\t")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"entry": "accession"})
    n_total = len(df)

    df["label_cat"] = df["subcellular_location_[cc]"].apply(_label)
    n_ambiguous = (df.label_cat == "ambiguous").sum()
    n_neither = (df.label_cat == "neither").sum()
    df = df[df.label_cat.isin(["nucleus", "membrane"])].copy()
    df["label"] = (df.label_cat == "membrane").astype(int)

    df = df.dropna(subset=["sequence"])
    df["sequence"] = df.sequence.str.upper()
    len_ok = df.sequence.str.len().between(50, 1022)
    aa_ok = df.sequence.apply(lambda s: set(s) <= STANDARD_AA)
    n_bad_len_or_aa = (~(len_ok & aa_ok)).sum()
    df = df[len_ok & aa_ok]

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
    n_dup = n_before_dedup - len(df)

    n_kept = len(df)
    n_nucleus = (df.label == 0).sum()
    n_membrane = (df.label == 1).sum()
    majority = max(n_nucleus, n_membrane) / n_kept
    os.makedirs("results", exist_ok=True)

    # family group for the homology-aware split: UniProt's curated family string when
    # present, else a singleton group per protein (no family = its own group of size 1)
    fam = df.protein_families.fillna("").str.strip()
    df["family_group"] = np.where(fam == "", "singleton:" + df.accession, fam)
    family_sizes = df.family_group.value_counts()
    top10 = family_sizes.nlargest(10).index
    frac_nucleus_top10 = (df.loc[df.label == 0, "family_group"].isin(top10)).mean()
    frac_membrane_top10 = (df.loc[df.label == 1, "family_group"].isin(top10)).mean()
    df[df.family_group.isin(top10)].groupby("family_group").agg(
        size=("accession", "count"), n_nucleus=("label", lambda s: (s == 0).sum()), n_membrane=("label", lambda s: (s == 1).sum())
    ).sort_values("size", ascending=False).to_csv("results/family_top10.csv")

    stats = {
        "n_raw_total": n_total,
        "n_dropped_ambiguous": n_ambiguous,
        "n_dropped_neither": n_neither,
        "n_dropped_length_or_nonstandard_aa": n_bad_len_or_aa,
        "n_dropped_duplicate_sequence": n_dup,
        "n_kept_final": n_kept,
        "n_nucleus": n_nucleus,
        "n_membrane": n_membrane,
        "majority_class_fraction": round(majority, 4),
        "n_distinct_families": family_sizes.shape[0],
        "family_size_median": family_sizes.median(),
        "frac_nucleus_in_top10_families": round(frac_nucleus_top10, 4),
        "frac_membrane_in_top10_families": round(frac_membrane_top10, 4),
    }
    print("Data pipeline stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if majority > 0.8:
        print(f"  NOTE: class balance is worse than 80/20 (majority={majority:.1%}); not resampled.")

    pd.DataFrame([stats]).to_csv("results/dataset_stats.csv", index=False)

    return df[["accession", "sequence", "label", "family_group"]].reset_index(drop=True), stats
