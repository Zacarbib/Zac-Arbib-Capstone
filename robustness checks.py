"""
Robustness checks for Capstone Studies 1 and 2
==============================================
Implements the reviewer-requested sensitivity analyses of 20 July 2026.
These are ROBUSTNESS analyses run AFTER the locked confirmatory plans;
label them as such in the write-up. Nothing here alters the confirmatory
pipeline or its outputs.

Checks:
  A. Duplicate-essay sensitivity (Study 1): drop one copy of the verbatim
     duplicated InstructGPT essay, rerun the primary solo-vs-InstructGPT
     convergence and MTLD tests.
  B. Prompt-level robustness: one mean per prompt x condition; paired
     comparison across the ten prompts (Wilcoxon + exact sign-flip
     permutation, 2^10 = 1024 flips). Study 1 primary and Study 2 H-S2a/b.
  C. Length robustness (Study 2):
       C1. correlations between word count and MTLD / convergence within
           conditions that have length variance;
       C2. truncation of every essay to a common token budget (sentence-
           preserving), then MTLD and convergence rerun, solo vs default.
  D. Embedding-model sensitivity: full five-corpus descriptives and the
     H-S2a / H-S2b permutation tests repeated with all-MiniLM-L6-v2.
  E. Holm correction across the three exploratory entropy pairwise tests.

Usage (Colab, same working directory as the confirmatory runs):
  pip install sentence-transformers scipy scikit-learn pandas
  python robustness_checks.py                 # full run
  python robustness_checks.py --nperm 2000    # quicker preview

Inputs:  hai-diversity-main/processed_data/, study2_essays.jsonl
Output:  robustness_results.txt

Seed: 20260721 (this script only; recorded in the output header).
Conventions copied verbatim from study1_pipeline.py / study2_analysis.py:
tokeniser, MTLD, sentence splitter, normalise(), LOO convergence,
stratified within-prompt permutation, Cohen's d, bootstrap CI, Holm.
Study 1 checks use RAW text (matching the standalone Study 1 run);
Study 2 checks use normalise()d text (matching the amended pipeline).
"""

import argparse, itertools, json, os, re, sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

SEED = 20260721
RNG = np.random.default_rng(SEED)
N_BOOT = 10_000
TRUNC_FLOOR = 100          # lower bound for the common token budget
K_ENTROPY = 5

CANDIDATE_DIRS = ["processed_data", "hai-diversity-main/processed_data"]

# ---------- text utilities (identical to the confirmatory scripts) ----------
def normalise(t):
    return (t.replace("\u2019", "'").replace("\u2018", "'")
             .replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2013", "-").replace("\u2014", "-"))

def tokens(t):
    return re.findall(r"[a-z']+", t.lower())

def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.replace("\n", " "))
            if s.strip()]

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

# ------------------------------ data loading -------------------------------
def data_dir():
    d = next((d for d in CANDIDATE_DIRS if os.path.isdir(d)), None)
    if not d:
        sys.exit("Cannot find processed_data/ (Study 1 corpus)")
    return d

def load_study1(raw=True):
    rows = []
    for c in ["solo", "gpt3", "instructgpt"]:
        with open(f"{data_dir()}/essays_{c}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                txt = r["essay"].strip()
                rows.append(dict(cond=c, prompt=r["title"].strip(),
                                 text=txt if raw else normalise(txt)))
    return [r for r in rows if r["text"]]

def load_study2_generated():
    if not os.path.exists("study2_essays.jsonl"):
        sys.exit("study2_essays.jsonl not found")
    rows = []
    with open("study2_essays.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append(dict(cond=r["cond"], prompt=r["prompt_title"].strip(),
                             text=normalise(r["text"].strip())))
    return [r for r in rows if r["text"]]

# ------------------------------ embeddings ---------------------------------
_MODELS = {}
def embed(rows, model_name="all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    if model_name not in _MODELS:
        _MODELS[model_name] = SentenceTransformer(model_name)
    model = _MODELS[model_name]
    embs = []
    for r in rows:
        sl = split_sentences(r["text"])
        if not sl:
            embs.append(None)
            continue
        v = np.asarray(model.encode(sl, normalize_embeddings=True,
                                    show_progress_bar=False)).mean(axis=0)
        n = np.linalg.norm(v)
        embs.append(v / n if n > 0 else None)
    return embs

# ---------------------- convergence + permutation machinery ----------------
def build_structures(rows, embs):
    prompts = sorted({r["prompt"] for r in rows})
    members, sims = {}, {}
    for p in prompts:
        idx = [i for i, r in enumerate(rows) if r["prompt"] == p
               and embs[i] is not None]
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

def perm_conv(rows, prompts, members, sims, cond_a, cond_b, n_perm):
    labels = [r["cond"] for r in rows]
    obs = loo_conv(rows, prompts, members, sims, labels)
    a = np.nanmean([obs[i] for i, r in enumerate(rows) if r["cond"] == cond_a])
    b = np.nanmean([obs[i] for i, r in enumerate(rows) if r["cond"] == cond_b])
    t_obs = b - a
    count = 0
    for _ in range(n_perm):
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
    return t_obs, (count + 1) / (n_perm + 1), obs

def perm_scores(rows, prompts, members, vals, cond_a, cond_b, n_perm):
    labels = [r["cond"] for r in rows]
    va = [vals[i] for i, r in enumerate(rows)
          if r["cond"] == cond_a and np.isfinite(vals[i])]
    vb = [vals[i] for i, r in enumerate(rows)
          if r["cond"] == cond_b and np.isfinite(vals[i])]
    t_obs = np.mean(vb) - np.mean(va)
    count = 0
    for _ in range(n_perm):
        da, db = [], []
        for p in prompts:
            pool = [i for i in members[p] if labels[i] in (cond_a, cond_b)
                    and np.isfinite(vals[i])]
            shuf = RNG.permutation([labels[i] for i in pool])
            for i, l in zip(pool, shuf):
                (da if l == cond_a else db).append(vals[i])
        if abs(np.mean(db) - np.mean(da)) >= abs(t_obs):
            count += 1
    return t_obs, (count + 1) / (n_perm + 1)

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
    ps = list(ps)
    order = np.argsort(ps)
    m = len(ps)
    adj, prev = [None] * m, 0.0
    for rank, i in enumerate(order):
        val = min(1.0, max(prev, (m - rank) * ps[i]))
        adj[i], prev = val, val
    return adj

def cond_vals(rows, arr, cond):
    return [arr[i] for i, r in enumerate(rows)
            if r["cond"] == cond and np.isfinite(arr[i])]

# ------------------------- paired prompt-level test -------------------------
def prompt_level_test(rows, arr, cond_a, cond_b):
    """One mean per prompt x condition; Wilcoxon + exact sign-flip test."""
    prompts = sorted({r["prompt"] for r in rows})
    pa, pb = [], []
    for p in prompts:
        va = [arr[i] for i, r in enumerate(rows)
              if r["prompt"] == p and r["cond"] == cond_a and np.isfinite(arr[i])]
        vb = [arr[i] for i, r in enumerate(rows)
              if r["prompt"] == p and r["cond"] == cond_b and np.isfinite(arr[i])]
        if va and vb:
            pa.append(np.mean(va))
            pb.append(np.mean(vb))
    pa, pb = np.array(pa), np.array(pb)
    diffs = pb - pa
    w, p_w = stats.wilcoxon(pb, pa)
    # exact sign-flip permutation over 2^n assignments
    t_obs = diffs.mean()
    n = len(diffs)
    flips = np.array(list(itertools.product([1, -1], repeat=n)))
    null = (flips * diffs).mean(axis=1)
    p_exact = np.mean(np.abs(null) >= abs(t_obs))
    return dict(n_prompts=n, mean_diff=t_obs,
                n_positive=int((diffs > 0).sum()),
                wilcoxon_p=p_w, signflip_p=p_exact)

# ------------------------------ truncation ----------------------------------
def truncate_text(text, budget):
    """Sentence-preserving truncation to exactly `budget` word tokens."""
    sents = split_sentences(text)
    kept, used = [], 0
    for s in sents:
        tk = tokens(s)
        if used + len(tk) <= budget:
            kept.append(s)
            used += len(tk)
        else:
            need = budget - used
            if need > 0:
                words = s.split()
                partial, cnt = [], 0
                for wtok in words:
                    partial.append(wtok)
                    cnt = len(tokens(" ".join(partial)))
                    if cnt >= need:
                        break
                kept.append(" ".join(partial))
            break
    return " ".join(kept)

# --------------------------------- entropy ----------------------------------
def entropy_holm(rows, embs, n_perm, conds=("solo", "default", "persona")):
    prompts = sorted({r["prompt"] for r in rows})
    assign = {}
    for p in prompts:
        idx = [i for i, r in enumerate(rows)
               if r["prompt"] == p and r["cond"] in conds and embs[i] is not None]
        X = np.vstack([embs[i] for i in idx])
        km = KMeans(n_clusters=K_ENTROPY, n_init=10,
                    random_state=SEED).fit_predict(X)
        assign[p] = (idx, km)

    def mean_entropy(labels):
        per = {c: [] for c in conds}
        for p in prompts:
            idx, km = assign[p]
            for c in conds:
                cl = [km[j] for j, i in enumerate(idx) if labels[i] == c]
                if not cl:
                    continue
                counts = np.bincount(cl, minlength=K_ENTROPY)
                probs = counts / counts.sum()
                H = -sum(q * np.log2(q) for q in probs if q > 0)
                per[c].append(H / np.log2(K_ENTROPY))
        return {c: float(np.mean(v)) for c, v in per.items() if v}

    true = [r["cond"] for r in rows]
    obs = mean_entropy(true)
    pairs = [("solo", "default"), ("solo", "persona"), ("default", "persona")]
    ps, details = [], []
    for a, b in pairs:
        t_obs = obs[b] - obs[a]
        count = 0
        for _ in range(n_perm):
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
        pv = (count + 1) / (n_perm + 1)
        ps.append(pv)
        details.append((f"{a} vs {b}", t_obs, pv))
    adj = holm(ps)
    return obs, [(name, t, p, h) for (name, t, p), h in zip(details, adj)]

# ----------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=10_000)
    args = ap.parse_args()
    NP = args.nperm

    L = []
    def rpt(s=""):
        print(s)
        L.append(s)

    rpt("ROBUSTNESS CHECKS (post-hoc sensitivity analyses, reviewer feedback 20 Jul 2026)")
    rpt(f"run: {datetime.now(timezone.utc).isoformat()}  seed: {SEED}  "
        f"N_PERM={NP}  N_BOOT={N_BOOT}")
    rpt("=" * 78)

    # ================= A. duplicate-essay sensitivity (Study 1) =============
    rpt("\nA. DUPLICATE-ESSAY SENSITIVITY (Study 1, raw text, mpnet)")
    s1 = load_study1(raw=True)
    counts = Counter((r["cond"], r["text"]) for r in s1)
    dups = [(c, t) for (c, t), n in counts.items() if n > 1]
    for c, t in dups:
        rpt(f"  duplicate found in condition '{c}' "
            f"({counts[(c, t)]} copies, first 60 chars: {t[:60]!r})")
    if not dups:
        rpt("  no verbatim duplicates found - check corpus files")
    else:
        seen, s1_dedup = set(), []
        for r in s1:
            key = (r["cond"], r["text"])
            if key in dups and key in seen:
                continue
            seen.add(key)
            s1_dedup.append(r)
        rpt(f"  essays: {len(s1)} -> {len(s1_dedup)} after removing one copy")
        for tag, rows in [("original", s1), ("dedup", s1_dedup)]:
            embs = embed(rows, "all-mpnet-base-v2")
            prompts, members, sims = build_structures(rows, embs)
            mt = np.array([mtld(r["text"]) for r in rows])
            t, p, conv = perm_conv(rows, prompts, members, sims,
                                   "solo", "instructgpt", NP)
            a, b = cond_vals(rows, conv, "solo"), cond_vals(rows, conv, "instructgpt")
            tm, pm = perm_scores(rows, prompts, members, mt,
                                 "solo", "instructgpt", NP)
            am, bm = cond_vals(rows, mt, "solo"), cond_vals(rows, mt, "instructgpt")
            rpt(f"  [{tag}] convergence: diff={t:+.4f} d={cohens_d(a, b):+.2f} "
                f"p={p:.4f} | MTLD: diff={tm:+.2f} d={cohens_d(am, bm):+.2f} p={pm:.4f}")
            if tag == "original":
                s1_embs, s1_conv, s1_mt = embs, conv, mt
                s1_struct = (prompts, members, sims)

    # ================= B. prompt-level robustness ===========================
    rpt("\nB. PROMPT-LEVEL ROBUSTNESS (one mean per prompt x condition)")
    r = prompt_level_test(s1, s1_conv, "solo", "instructgpt")
    rpt(f"  S1 solo vs instructgpt convergence: mean diff={r['mean_diff']:+.4f}, "
        f"{r['n_positive']}/{r['n_prompts']} prompts positive, "
        f"Wilcoxon p={r['wilcoxon_p']:.4f}, exact sign-flip p={r['signflip_p']:.4f}")

    s2 = load_study1(raw=False) + load_study2_generated()
    embs2 = embed(s2, "all-mpnet-base-v2")
    prompts2, members2, sims2 = build_structures(s2, embs2)
    conv2 = loo_conv(s2, prompts2, members2, sims2, [r_["cond"] for r_ in s2])
    mt2 = np.array([mtld(r_["text"]) for r_ in s2])
    for cond in ["default", "persona"]:
        r = prompt_level_test(s2, conv2, "solo", cond)
        rpt(f"  S2 solo vs {cond} convergence: mean diff={r['mean_diff']:+.4f}, "
            f"{r['n_positive']}/{r['n_prompts']} prompts positive, "
            f"Wilcoxon p={r['wilcoxon_p']:.4f}, exact sign-flip p={r['signflip_p']:.4f}")
    r = prompt_level_test(s2, mt2, "solo", "default")
    rpt(f"  S2 solo vs default MTLD: mean diff={r['mean_diff']:+.2f}, "
        f"{r['n_positive']}/{r['n_prompts']} prompts positive, "
        f"Wilcoxon p={r['wilcoxon_p']:.4f}, exact sign-flip p={r['signflip_p']:.4f}")

    # ================= C. length robustness =================================
    rpt("\nC. LENGTH ROBUSTNESS (Study 2 pipeline, normalised text)")
    wc = np.array([len(tokens(r_["text"])) for r_ in s2])
    rpt("  C1. word count vs outcomes within condition "
        "(Pearson r / Spearman rho):")
    for cond in ["solo", "instructgpt", "default", "persona"]:
        idx = [i for i, r_ in enumerate(s2) if r_["cond"] == cond
               and np.isfinite(conv2[i]) and np.isfinite(mt2[i])]
        w = wc[idx]
        if w.std() == 0:
            rpt(f"    {cond:<12}: no length variance, correlations undefined")
            continue
        c_ = conv2[idx]; m_ = mt2[idx]
        rpt(f"    {cond:<12} (len SD={w.std(ddof=1):5.1f}): "
            f"conv r={stats.pearsonr(w, c_)[0]:+.2f}/"
            f"rho={stats.spearmanr(w, c_)[0]:+.2f}  "
            f"MTLD r={stats.pearsonr(w, m_)[0]:+.2f}/"
            f"rho={stats.spearmanr(w, m_)[0]:+.2f}")

    budget = max(TRUNC_FLOOR,
                 int(min(wc[i] for i, r_ in enumerate(s2)
                         if r_["cond"] in ("solo", "default"))))
    rpt(f"  C2. truncation to a common budget of {budget} tokens "
        f"(sentence-preserving; essays shorter than the budget kept whole):")
    s2_tr = [dict(r_, text=truncate_text(r_["text"], budget)) for r_ in s2]
    wc_tr = np.array([len(tokens(r_["text"])) for r_ in s2_tr])
    for cond in ["solo", "default"]:
        w = [wc_tr[i] for i, r_ in enumerate(s2_tr) if r_["cond"] == cond]
        rpt(f"    {cond}: truncated length M={np.mean(w):.0f} "
            f"SD={np.std(w, ddof=1):.1f}")
    embs_tr = embed(s2_tr, "all-mpnet-base-v2")
    prompts_t, members_t, sims_t = build_structures(s2_tr, embs_tr)
    mt_tr = np.array([mtld(r_["text"]) for r_ in s2_tr])
    t, p, conv_tr = perm_conv(s2_tr, prompts_t, members_t, sims_t,
                              "solo", "default", NP)
    a, b = cond_vals(s2_tr, conv_tr, "solo"), cond_vals(s2_tr, conv_tr, "default")
    lo, hi = boot_ci(a, b)
    rpt(f"    convergence solo vs default (truncated): diff={t:+.4f} "
        f"d={cohens_d(a, b):+.2f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f}   "
        f"(untruncated d was reported as +3.24)")
    tm, pm = perm_scores(s2_tr, prompts_t, members_t, mt_tr,
                         "solo", "default", NP)
    am, bm = cond_vals(s2_tr, mt_tr, "solo"), cond_vals(s2_tr, mt_tr, "default")
    rpt(f"    MTLD solo vs default (truncated): diff={tm:+.2f} "
        f"d={cohens_d(am, bm):+.2f} p={pm:.4f}   "
        f"(untruncated d was reported as +2.02)")

    # ================= D. embedding-model sensitivity =======================
    rpt("\nD. EMBEDDING-MODEL SENSITIVITY (all-MiniLM-L6-v2, full pipeline)")
    embs_m = embed(s2, "all-MiniLM-L6-v2")
    prompts_m, members_m, sims_m = build_structures(s2, embs_m)
    conv_m = loo_conv(s2, prompts_m, members_m, sims_m,
                      [r_["cond"] for r_ in s2])
    order = []
    for cond in ["solo", "gpt3", "instructgpt", "persona", "default"]:
        v = cond_vals(s2, conv_m, cond)
        order.append((cond, np.mean(v)))
        rpt(f"    {cond:<12}: convergence M={np.mean(v):.4f} "
            f"(SD={np.std(v, ddof=1):.4f})")
    rpt("    ordering (low to high): "
        + " < ".join(c for c, _ in sorted(order, key=lambda x: x[1])))
    ps = []
    for tag, cond in [("H-S2a default vs solo", "default"),
                      ("H-S2b persona vs solo", "persona")]:
        t, p, _ = perm_conv(s2, prompts_m, members_m, sims_m, "solo", cond, NP)
        a = cond_vals(s2, conv_m, "solo"); b = cond_vals(s2, conv_m, cond)
        ps.append((tag, t, p, cohens_d(a, b)))
    hp = holm([p for _, _, p, _ in ps])
    for (tag, t, p, d), h in zip(ps, hp):
        rpt(f"    {tag}: diff={t:+.4f} d={d:+.2f} perm p={p:.4f} Holm p={h:.4f}")

    # ================= E. entropy multiple-comparison correction ============
    rpt("\nE. ENTROPY PAIRWISE TESTS, HOLM-CORRECTED (k=5, exploratory)")
    obs, rows_e = entropy_holm(s2, embs2, NP)
    for c in ["solo", "persona", "default"]:
        rpt(f"    normalised entropy {c:<8}: {obs[c]:.3f}")
    for name, t, p, h in rows_e:
        rpt(f"    {name}: diff={t:+.3f} perm p={p:.4f} Holm p={h:.4f}")

    rpt("\nDone.")
    with open("robustness_results.txt", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\nSaved: robustness_results.txt")

if __name__ == "__main__":
    main()
