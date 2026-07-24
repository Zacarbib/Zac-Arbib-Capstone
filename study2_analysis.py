"""
Study 2 analysis
================
Executes the pre-specified plan locked 7 July 2026
(Study2_PreSpecified_Plan.md, sections 7 to 10). Identical measurement
pipeline to Study 1. Do not alter tests or thresholds; deviations must
be logged in study2_deviations.md and reported.

Inputs:  hai-diversity-main/processed_data/  (Study 1 corpus)
         study2_essays.jsonl                 (generated corpus)
Outputs: study2_results.txt, study2_per_essay.csv,
         fig_spectrum.png, entropy_sensitivity.csv

Usage:   python study2_analysis.py            # confirmatory (SBERT)
         python study2_analysis.py --tfidf    # machinery debug only, NOT for the paper
"""

import argparse, json, os, re, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260707
N_PERM = 10_000
N_BOOT = 10_000
ALPHA = 0.05
MTLD_D_BOUND = 0.30
K_PRIMARY = 5
K_SENSITIVITY = [3, 4, 5, 6, 8]
SPECTRUM_ORDER = ["solo", "gpt3", "instructgpt", "persona", "default"]
RNG = np.random.default_rng(SEED)

CANDIDATE_DIRS = ["processed_data", "hai-diversity-main/processed_data"]

# ---------------------------- text utilities ----------------------------
def normalise(t):
    """Normalise typographic punctuation to ASCII so tokenisation is
    comparable across human-typed and model-generated text. Amendment
    logged before confirmatory analysis (see study2_deviations.md)."""
    return (t.replace("\u2019", "'").replace("\u2018", "'")
             .replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2013", "-").replace("\u2014", "-"))

def tokens(t):
    return re.findall(r"[a-z']+", t.lower())

def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.replace("\n", " ")) if s.strip()]

def _mtld_dir(toks, threshold=0.72):
    factors, types, count = 0.0, set(), 0
    for t in toks:
        count += 1
        types.add(t)
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count:
        ttr = len(types) / count
        if ttr < 1:
            factors += (1 - ttr) / (1 - threshold)
    return len(toks) / factors if factors else np.nan

def mtld(t):
    tk = tokens(t)
    if len(tk) < 50:
        return np.nan
    return (_mtld_dir(tk) + _mtld_dir(tk[::-1])) / 2

# ------------------------------ load data -------------------------------
def load_all():
    data_dir = next((d for d in CANDIDATE_DIRS if os.path.isdir(d)), None)
    if not data_dir:
        sys.exit("Cannot find processed_data/ (Study 1 corpus)")
    rows = []
    for c in ["solo", "gpt3", "instructgpt"]:
        with open(f"{data_dir}/essays_{c}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                rows.append(dict(cond=c, prompt=r["title"].strip(),
                                 text=normalise(r["essay"].strip())))
    if not os.path.exists("study2_essays.jsonl"):
        sys.exit("study2_essays.jsonl not found - run study2_generate.py first")
    with open("study2_essays.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append(dict(cond=r["cond"], prompt=r["prompt_title"].strip(),
                             text=normalise(r["text"].strip())))
    rows = [r for r in rows if r["text"]]
    return rows

# ------------------------------ embeddings ------------------------------
def embed(rows, use_tfidf):
    sent_lists = [split_sentences(r["text"]) for r in rows]
    if use_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        vec.fit([s for sl in sent_lists for s in sl])
        enc = lambda sl: np.asarray(vec.transform(sl).todense())
        mode = "TF-IDF (DEBUG - not for the paper)"
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")
        enc = lambda sl: model.encode(sl, normalize_embeddings=True,
                                      show_progress_bar=False)
        mode = "SBERT all-mpnet-base-v2"
    embs = []
    for sl in sent_lists:
        if not sl:
            embs.append(None)
            continue
        v = np.asarray(enc(sl)).mean(axis=0)
        n = np.linalg.norm(v)
        embs.append(v / n if n > 0 else None)
    return embs, mode

# ------------------------- convergence machinery ------------------------
def build_structures(rows, embs):
    prompts = sorted({r["prompt"] for r in rows})
    members, sims = {}, {}
    for p in prompts:
        idx = [i for i, r in enumerate(rows) if r["prompt"] == p and embs[i] is not None]
        members[p] = idx
        M = np.vstack([embs[i] for i in idx])
        sims[p] = M @ M.T
    return prompts, members, sims

def loo_conv(rows, prompts, members, sims, labels):
    conv = np.full(len(rows), np.nan)
    for p in prompts:
        idx = members[p]
        S = sims[p]
        labs = [labels[i] for i in idx]
        for c in set(labs):
            grp = [k for k, l in enumerate(labs) if l == c]
            if len(grp) < 2:
                continue
            for k in grp:
                oth = [g for g in grp if g != k]
                conv[idx[k]] = S[k, oth].mean()
    return conv

def perm_conv(rows, prompts, members, sims, cond_a, cond_b):
    labels = [r["cond"] for r in rows]
    obs = loo_conv(rows, prompts, members, sims, labels)
    a = np.nanmean([obs[i] for i, r in enumerate(rows) if r["cond"] == cond_a])
    b = np.nanmean([obs[i] for i, r in enumerate(rows) if r["cond"] == cond_b])
    t_obs = b - a
    count = 0
    for _ in range(N_PERM):
        lab = list(labels)
        for p in prompts:
            pool = [i for i in members[p] if labels[i] in (cond_a, cond_b)]
            shuf = RNG.permutation([labels[i] for i in pool])
            for i, l in zip(pool, shuf):
                lab[i] = l
        cv = loo_conv(rows, prompts, members, sims, lab)
        pa = np.nanmean([cv[i] for i in range(len(rows)) if lab[i] == cond_a])
        pb = np.nanmean([cv[i] for i in range(len(rows)) if lab[i] == cond_b])
        if abs(pb - pa) >= abs(t_obs):
            count += 1
    return t_obs, (count + 1) / (N_PERM + 1), obs

def perm_scores(rows, prompts, members, vals, cond_a, cond_b):
    labels = [r["cond"] for r in rows]
    va = [vals[i] for i, r in enumerate(rows) if r["cond"] == cond_a and np.isfinite(vals[i])]
    vb = [vals[i] for i, r in enumerate(rows) if r["cond"] == cond_b and np.isfinite(vals[i])]
    t_obs = np.mean(vb) - np.mean(va)
    count = 0
    for _ in range(N_PERM):
        da, db = [], []
        for p in prompts:
            pool = [i for i in members[p] if labels[i] in (cond_a, cond_b)
                    and np.isfinite(vals[i])]
            shuf = RNG.permutation([labels[i] for i in pool])
            for i, l in zip(pool, shuf):
                (da if l == cond_a else db).append(vals[i])
        if abs(np.mean(db) - np.mean(da)) >= abs(t_obs):
            count += 1
    return t_obs, (count + 1) / (N_PERM + 1)

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sp

def boot_ci(a, b):
    a = np.asarray([x for x in a if np.isfinite(x)])
    b = np.asarray([x for x in b if np.isfinite(x)])
    ds = np.empty(N_BOOT)
    for k in range(N_BOOT):
        ds[k] = RNG.choice(b, len(b)).mean() - RNG.choice(a, len(a)).mean()
    return np.percentile(ds, [2.5, 97.5])

def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj, prev = [None] * m, 0.0
    for rank, i in enumerate(order):
        val = min(1.0, max(prev, (m - rank) * ps[i]))
        adj[i], prev = val, val
    return adj

# ------------------------------ entropy ---------------------------------
def entropy_at_k(rows, embs, prompts, k, conds=("solo", "default", "persona")):
    """Normalised Shannon entropy of cluster distribution per condition,
    averaged across prompts. Cluster assignments are label-independent."""
    assign = {}
    for p in prompts:
        idx = [i for i, r in enumerate(rows)
               if r["prompt"] == p and r["cond"] in conds and embs[i] is not None]
        X = np.vstack([embs[i] for i in idx])
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        assign[p] = (idx, km)

    def mean_entropy(labels):
        per_cond = {c: [] for c in conds}
        for p in prompts:
            idx, km = assign[p]
            for c in conds:
                cl = [km[j] for j, i in enumerate(idx) if labels[i] == c]
                if not cl:
                    continue
                counts = np.bincount(cl, minlength=k)
                probs = counts / counts.sum()
                H = -sum(q * np.log2(q) for q in probs if q > 0)
                per_cond[c].append(H / np.log2(k))
        return {c: float(np.mean(v)) for c, v in per_cond.items() if v}

    true = [r["cond"] for r in rows]
    obs = mean_entropy(true)
    pvals = {}
    for a, b in [("solo", "default"), ("solo", "persona"), ("default", "persona")]:
        t_obs = obs[b] - obs[a]
        count = 0
        for _ in range(N_PERM):
            lab = list(true)
            for p in prompts:
                idx, _ = assign[p]
                pool = [i for i in idx if true[i] in (a, b)]
                shuf = RNG.permutation([true[i] for i in pool])
                for i, l in zip(pool, shuf):
                    lab[i] = l
            e = mean_entropy(lab)
            if abs(e[b] - e[a]) >= abs(t_obs):
                count += 1
        pvals[f"{a} vs {b}"] = (t_obs, (count + 1) / (N_PERM + 1))
    return obs, pvals

# -------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf", action="store_true")
    args = ap.parse_args()

    rows = load_all()
    embs, mode = embed(rows, args.tfidf)
    prompts, members, sims = build_structures(rows, embs)
    mtlds = np.array([mtld(r["text"]) for r in rows])

    L = []
    def rpt(s=""):
        print(s); L.append(s)

    rpt(f"STUDY 2 RESULTS - embedding mode: {mode}")
    rpt(f"run: {datetime.now(timezone.utc).isoformat()}  seed: {SEED}  "
        f"N_PERM={N_PERM}  N_BOOT={N_BOOT}")
    rpt("=" * 70)

    labels = [r["cond"] for r in rows]
    conv = loo_conv(rows, prompts, members, sims, labels)
    wc = {c: [len(tokens(r["text"])) for r in rows if r["cond"] == c]
          for c in ["default", "persona"]}

    rpt(f"{'condition':<13}{'n':>4}{'MTLD':>15}{'convergence':>20}{'words':>14}")
    for c in SPECTRUM_ORDER:
        m = [mtlds[i] for i, r in enumerate(rows) if r["cond"] == c and np.isfinite(mtlds[i])]
        v = [conv[i] for i, r in enumerate(rows) if r["cond"] == c and np.isfinite(conv[i])]
        w = f"{np.mean(wc[c]):.0f} ({np.std(wc[c], ddof=1):.0f})" if c in wc else "-"
        rpt(f"{c:<13}{len(v):>4}{np.mean(m):>10.1f} ({np.std(m, ddof=1):.1f})"
            f"{np.mean(v):>13.4f} ({np.std(v, ddof=1):.4f}){w:>14}")
    rpt()

    # ---- confirmatory family ----
    rpt("PRIMARY CONFIRMATORY (Holm family of 2: H-S2a, H-S2b)")
    results = {}
    for tag, cond in [("H-S2a default vs solo", "default"),
                      ("H-S2b persona vs solo", "persona")]:
        t, p, _ = perm_conv(rows, prompts, members, sims, "solo", cond)
        a = [conv[i] for i, r in enumerate(rows) if r["cond"] == "solo"]
        b = [conv[i] for i, r in enumerate(rows) if r["cond"] == cond]
        d = cohens_d(a, b); lo, hi = boot_ci(a, b)
        results[tag] = (t, p, d, lo, hi)
    hp = holm([results[k][1] for k in results])
    for (tag, (t, p, d, lo, hi)), h in zip(results.items(), hp):
        rpt(f"  {tag}: diff={t:+.4f}  d={d:+.2f}  CI[{lo:+.4f},{hi:+.4f}]  "
            f"perm p={p:.4f}  Holm p={h:.4f}")
    s2a_sig = hp[0] < ALPHA and results["H-S2a default vs solo"][2] > 0
    rpt()

    # ---- MTLD bound (outside alpha family, per plan section 8) ----
    t, p = perm_scores(rows, prompts, members, mtlds, "solo", "default")
    a = [mtlds[i] for i, r in enumerate(rows) if r["cond"] == "solo"]
    b = [mtlds[i] for i, r in enumerate(rows) if r["cond"] == "default"]
    d = cohens_d(a, b); lo, hi = boot_ci(a, b)
    bound_ok = (p >= ALPHA) and (abs(d) < MTLD_D_BOUND)
    rpt("MTLD default vs solo (estimation, outside alpha family)")
    rpt(f"  diff={t:+.2f}  d={d:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  p={p:.4f}  "
        f"bound(|d|<{MTLD_D_BOUND}, p>=.05): {'OK' if bound_ok else 'FAIL'}")
    rpt()
    rpt(f"STUDY 2 DISSOCIATION RULE: "
        f"{'SUPPORTED' if (s2a_sig and bound_ok) else 'NOT SUPPORTED'}")
    rpt()

    # ---- secondary ----
    rpt("SECONDARY (labelled, no confirmatory claims)")
    t, p, _ = perm_conv(rows, prompts, members, sims, "persona", "default")
    a = [conv[i] for i, r in enumerate(rows) if r["cond"] == "persona"]
    b = [conv[i] for i, r in enumerate(rows) if r["cond"] == "default"]
    rpt(f"  default vs persona convergence: diff={np.nanmean(b)-np.nanmean(a):+.4f}  "
        f"d={cohens_d(a, b):+.2f}  p={p:.4f}")
    t, p = perm_scores(rows, prompts, members, mtlds, "solo", "persona")
    a = [mtlds[i] for i, r in enumerate(rows) if r["cond"] == "solo"]
    b = [mtlds[i] for i, r in enumerate(rows) if r["cond"] == "persona"]
    rpt(f"  persona vs solo MTLD: d={cohens_d(a, b):+.2f}  p={p:.4f}")
    rpt()

    # ---- exploratory entropy ----
    rpt(f"EXPLORATORY: cluster entropy (k={K_PRIMARY} primary; "
        f"sensitivity in entropy_sensitivity.csv)")
    obs, pvals = entropy_at_k(rows, embs, prompts, K_PRIMARY)
    for c in ["solo", "default", "persona"]:
        rpt(f"  normalised entropy {c:<8}: {obs[c]:.3f}")
    for pair, (t, p) in pvals.items():
        rpt(f"  {pair}: diff={t:+.3f}  perm p={p:.4f}")
    sens = []
    for k in K_SENSITIVITY:
        o, _ = entropy_at_k(rows, embs, prompts, k)
        sens.append(dict(k=k, **o))
    pd.DataFrame(sens).to_csv("entropy_sensitivity.csv", index=False)
    rpt()

    # ---- outputs ----
    pd.DataFrame([dict(cond=r["cond"], prompt=r["prompt"],
                       mtld=mtlds[i], conv=conv[i])
                  for i, r in enumerate(rows)]).to_csv("study2_per_essay.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for x, c in enumerate(SPECTRUM_ORDER):
        v = np.array([conv[i] for i, r in enumerate(rows)
                      if r["cond"] == c and np.isfinite(conv[i])])
        ax.scatter(np.full(len(v), x) + RNG.uniform(-0.12, 0.12, len(v)),
                   v, s=9, alpha=0.3)
        ax.errorbar(x, v.mean(), yerr=1.96 * v.std(ddof=1) / np.sqrt(len(v)),
                    fmt="o", color="black", capsize=4, zorder=3)
    ax.set_xticks(range(len(SPECTRUM_ORDER)),
                  ["human\nsolo", "human +\nGPT-3", "human +\nInstructGPT",
                   "GPT-5.5\npersonas", "GPT-5.5\ndefault"])
    ax.set_ylabel("Semantic convergence (within prompt)")
    ax.set_title("The delegation spectrum")
    fig.tight_layout()
    fig.savefig("fig_spectrum.png", dpi=200)

    open("study2_results.txt", "w").write("\n".join(L))
    print("\nSaved: study2_results.txt, study2_per_essay.csv, "
          "fig_spectrum.png, entropy_sensitivity.csv")

if __name__ == "__main__":
    main()
