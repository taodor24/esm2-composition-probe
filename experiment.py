"""Representations, splits, models, and evaluation for the localization task."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 1000


def composition_features(sequences):
    """20 amino-acid frequencies (sum to 1) + log(length). 21 dims."""
    freqs = np.array([[s.count(a) / len(s) for a in AA] for s in sequences])
    loglen = np.log([len(s) for s in sequences]).reshape(-1, 1)
    return np.hstack([freqs, loglen]).astype(np.float64)


def _bootstrap_auc(y_true, scores, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = np.empty(n_boot)
    i = 0
    while i < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots[i] = roc_auc_score(y_true[idx], scores[idx])
        i += 1
    return boots


def fit_eval(Xtr, ytr, Xte, yte, seed):
    """Standardize (fit on train fold only) then logistic regression; same hyperparameters
    for every representation so differences in AUC reflect the representation, not the model.
    Bootstrap uses the same seed for every representation within a split on purpose: all
    representations score the SAME test proteins, so resample index i is the same draw of
    proteins across representations. That pairing is what makes the difference CIs in
    paired_diff() below valid (and narrower than the marginals -- see its docstring)."""
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
    scores = clf.predict_proba(Xte_s)[:, 1]
    auc = roc_auc_score(yte, scores)
    boots = _bootstrap_auc(yte, scores, seed)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi, boots


def residualize(comp_train, esm_train, comp_test, esm_test):
    """Regress each ESM dimension on composition, fit on TRAIN ONLY, then subtract the
    train-fit prediction from both folds. Fitting this on the full dataset (or on the
    test fold) would leak test-set composition-embedding structure into representation
    C and invalidate the AUC(B) vs AUC(C) comparison -- this is the one place the whole
    analysis can silently break."""
    reg = LinearRegression().fit(comp_train, esm_train)
    return esm_train - reg.predict(comp_train), esm_test - reg.predict(comp_test)


def paired_diff(boots_a, boots_b):
    """boots_a/boots_b come from the same resample-index draws (see fit_eval), so this is a
    PAIRED difference, not a difference of independent samples: Var(diff) = Var(a) + Var(b)
    - 2*Cov(a,b), and shared test proteins make Cov(a,b) > 0, so this CI is expected to be
    narrower than either marginal CI -- that is the point of pairing, not an error."""
    diff = boots_a - boots_b
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return diff.mean(), lo, hi


def _split(idx, y, seed, groups):
    """Stratified random split by default; group-aware (no family split across train/test)
    when `groups` is given, to check whether performance survives removing family leakage."""
    if groups is None:
        return train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(idx, y, groups=groups))
    return idx[tr], idx[te]


def run_representations(df, esm, groups=None, split_name="random"):
    y = df.label.values
    comp = composition_features(df.sequence.values)
    loglen = comp[:, -1:]
    idx_all = np.arange(len(df))
    rows, diffs = [], []

    for seed in SEEDS:
        tr, te = _split(idx_all, y, seed, groups)
        ytr, yte = y[tr], y[te]
        esm_res_tr, esm_res_te = residualize(comp[tr], esm[tr], comp[te], esm[te])

        reps = {
            "A_composition": (comp[tr], comp[te]),
            "B_esm": (esm[tr], esm[te]),
            "C_esm_residualized": (esm_res_tr, esm_res_te),
            "D_length_only": (loglen[tr], loglen[te]),
            "E_composition_plus_residual": (np.hstack([comp[tr], esm_res_tr]), np.hstack([comp[te], esm_res_te])),
        }
        boots_by_name = {}
        for name, (Xtr, Xte) in reps.items():
            auc, lo, hi, boots = fit_eval(Xtr, ytr, Xte, yte, seed)
            rows.append({"split": split_name, "representation": name, "seed": seed, "auc": auc, "ci_low": lo, "ci_high": hi})
            boots_by_name[name] = boots

        for a, b in [("B_esm", "C_esm_residualized"), ("B_esm", "A_composition"), ("C_esm_residualized", "D_length_only")]:
            mean_diff, dlo, dhi = paired_diff(boots_by_name[a], boots_by_name[b])
            diffs.append({"split": split_name, "seed": seed, "a": a, "b": b, "mean_diff": mean_diff, "ci_low": dlo, "ci_high": dhi})

    return pd.DataFrame(rows), pd.DataFrame(diffs)


def family_leakage_stats(df):
    """How much the family-aware split actually changes vs. the random one: the fraction
    of proteins that even belong to a non-singleton family, and how often the random split
    puts a same-family protein in train while its sibling is in test (leakage the group
    split removes). Singleton proteins can never match here by construction."""
    sizes = df.family_group.value_counts()
    frac_family_ge2 = df.family_group.map(sizes).ge(2).mean()
    idx_all, y = np.arange(len(df)), df.label.values
    leak_fracs = []
    for seed in SEEDS:
        tr, te = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=seed)
        train_families = set(df.family_group.values[tr])
        leak_fracs.append(np.mean([fg in train_families for fg in df.family_group.values[te]]))
    return {
        "frac_proteins_in_family_size_ge2": round(frac_family_ge2, 4),
        "mean_frac_test_with_same_family_in_train_random_split": round(float(np.mean(leak_fracs)), 4),
    }


def residual_r2(df, esm, seed=0):
    """Per-ESM-dimension R^2 from composition, fit on train and evaluated on the held-out
    test fold (same split scheme as the main analysis, seed 0)."""
    y = df.label.values
    comp = composition_features(df.sequence.values)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed)
    reg = LinearRegression().fit(comp[tr], esm[tr])
    pred = reg.predict(comp[te])
    ss_res = ((esm[te] - pred) ** 2).sum(axis=0)
    ss_tot = ((esm[te] - esm[te].mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot
    return r2
