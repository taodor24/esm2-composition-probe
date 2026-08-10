"""Single entry point: python run.py reproduces the full analysis."""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_data
from embed import compute_or_load_embeddings
from experiment import run_representations, residual_r2, family_leakage_stats

REP_ORDER = ["A_composition", "B_esm", "C_esm_residualized", "D_length_only", "E_composition_plus_residual"]
REP_LABELS = {
    "A_composition": "A: composition",
    "B_esm": "B: ESM-2",
    "C_esm_residualized": "C: ESM-2, residualized",
    "D_length_only": "D: length only",
    "E_composition_plus_residual": "E: A+C combined",
}


def make_hero_figure(results):
    """Grouped bars: one group per representation, one bar per split (random vs. family),
    so the family-confound check lives in the hero figure instead of a second figure."""
    agg = results.groupby(["representation", "split"])["auc"].agg(["mean", "min", "max"])
    x = np.arange(len(REP_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, split in enumerate(["random", "family"]):
        sub = agg.xs(split, level="split").loc[REP_ORDER]
        means = sub["mean"].values
        err = [means - sub["min"].values, sub["max"].values - means]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=err, capsize=3, label=f"{split} split")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([REP_LABELS[r] for r in REP_ORDER], rotation=15, ha="right")
    ax.set_ylabel("ROC-AUC (mean, range across 5 seeds)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Subcellular localization: AUC by representation and split")
    ax.legend()
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/representation_comparison.png", dpi=150)
    plt.close(fig)


def make_r2_histogram(r2):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2, bins=30, color="#55A868", edgecolor="white")
    ax.set_xlabel("R^2 (ESM dimension ~ composition), test fold")
    ax.set_ylabel("count of ESM dimensions")
    ax.set_title("How much of each ESM dimension is composition?")
    fig.tight_layout()
    fig.savefig("figures/residual_r2_hist.png", dpi=150)
    plt.close(fig)


def main():
    df, _ = load_data()
    esm = compute_or_load_embeddings(df)

    res_r, diff_r = run_representations(df, esm, split_name="random")
    res_f, diff_f = run_representations(df, esm, groups=df.family_group.values, split_name="family")
    results = pd.concat([res_r, res_f], ignore_index=True)
    diffs = pd.concat([diff_r, diff_f], ignore_index=True)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/representations.csv", index=False)
    diffs.to_csv("results/representation_diffs.csv", index=False)

    r2 = residual_r2(df, esm)
    pd.DataFrame({"esm_dim": np.arange(len(r2)), "r2": r2}).to_csv("results/residual_r2.csv", index=False)

    fam_stats = family_leakage_stats(df)
    pd.DataFrame([fam_stats]).to_csv("results/family_stats.csv", index=False)

    make_hero_figure(results)
    make_r2_histogram(r2)

    for split in ["random", "family"]:
        print(f"\n=== AUC summary, {split} split (mean, [min, max] across 5 seeds) ===")
        sub_split = results[results.split == split]
        for rep in REP_ORDER:
            sub = sub_split[sub_split.representation == rep]
            print(f"{REP_LABELS[rep]:28s} mean={sub.auc.mean():.3f}  [{sub.auc.min():.3f}, {sub.auc.max():.3f}]  "
                  f"mean CI=({sub.ci_low.mean():.3f}, {sub.ci_high.mean():.3f})")

    print("\n=== Paired-bootstrap AUC differences (same test proteins -> narrower than marginals) ===")
    print(diffs.groupby(["split", "a", "b"])[["mean_diff", "ci_low", "ci_high"]].mean().round(3))

    print("\n=== Residual R^2 distribution (320 ESM dims, composition -> ESM, random split) ===")
    print(f"median={np.median(r2):.3f}  Q1={np.percentile(r2,25):.3f}  Q3={np.percentile(r2,75):.3f}  "
          f"frac(R2>0.5)={(r2 > 0.5).mean():.3f}")

    print("\n=== Family-split constraint (results/family_stats.csv) ===")
    print(fam_stats)

    print("\nDone. See results/ and figures/.")


if __name__ == "__main__":
    main()
