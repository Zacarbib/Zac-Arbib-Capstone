"""
STUDY 1 PIPELINE: Lexical vs ideational diversity in the Padmakumar & He (2024) corpus
========================================================================================
ANALYSIS PLAN — PRE-SPECIFIED 6 JULY 2026, BEFORE THE CONFIRMATORY SBERT RUN
----------------------------------------------------------------------------------------
Data: processed_data/essays_{solo,gpt3,instructgpt}.jsonl (297 essays, 10 prompts).

Measures (per essay):
  1. MTLD (McCarthy & Jarvis, 2010): length-robust lexical diversity. Bidirectional,
     TTR threshold .72, on lowercased word tokens.
  2. Semantic convergence: essay embedded as the mean of its sentence embeddings
     (SBERT all-mpnet-base-v2; sentence-level embedding avoids max-length truncation).
     Convergence = leave-one-out mean cosine similarity to the other essays in the
     SAME prompt x SAME condition cell. Sentence splitting for whole-essay embeddings
     uses one regex splitter for ALL conditions (consistency across conditions).
  3. User-only convergence (gpt3/instructgpt only): same as (2) but embedding only
     sentences labelled 'U' in the corpus's own authorship field.

PRIMARY CONFIRMATORY TESTS (family = 2, Holm-corrected):
  Solo vs InstructGPT on (a) semantic convergence, (b) MTLD.
  Test: stratified permutation test (condition labels shuffled within prompt,
  10,000 permutations, two-sided). Effect size: Cohen's d + 95% bootstrap CI
  (10,000 resamples) on the mean difference.

DECISION RULE for the dissociation:
  Supported if convergence differs significantly (Holm-adjusted p < .05) AND MTLD
  shows no significant difference with |d| < 0.3. NOTE: a non-significant MTLD test
  is evidence of absence only in the weak sense; report d and CI, do not claim
  equivalence without an equivalence test.

SECONDARY / EXPLORATORY (labelled as such, no confirmatory claims):
  - Solo vs GPT3 on both measures (same machinery).
  - Mechanism decomposition: (i) user-only convergence in AI conditions vs solo
    whole-essay convergence; (ii) within-essay paired contrast, whole-essay vs
    user-only convergence (Wilcoxon signed-rank), per AI condition.

All language in the write-up: associational ("convergence is higher under"),
never causal about cognition. Generation of this plan predates the SBERT run.
----------------------------------------------------------------------------------------
Usage:
  pip install sentence-transformers pandas scipy scikit-learn
  python study1_pipeline.py            # full SBERT run (downloads model first time)
  python study1_pipeline.py --tfidf    # fallback/debug mode, no model download
Outputs:
  study1_results.txt   (all statistics, human-readable)
  study1_per_essay.csv (per-essay measures, for figures)
"""
import argparse, json, re, sys
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260706)
N_PERM = 10_000
N_BOOT = 10_000
import os
_CANDIDATE_DIRS = ["processed_data", "hai-diversity-main/processed_data"]
_DATA_DIR = next((d for d in _CANDIDATE_DIRS if os.path.isdir(d)), _CANDIDATE_DIRS[0])
DATA = _DATA_DIR + "/essays_{}.jsonl"
CONDS = ["solo", "gpt3", "instructgpt"]

# ---------------- text utilities ----------------
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text):
    sents = [s.strip() for s in SENT_SPLIT.split(text.replace("\n", " ")) if s.strip()]
    return [s for s in sents if len(s) > 1]

def tokens(text):
    return re.findall(r"[a-z']+", text.lower())

def mtld_dir(toks, threshold=0.72):
    factors, types, count = 0.0, set(), 0
    for t in toks:
        count += 1
        types.add(t)
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count > 0:
        ttr = len(types) / count
        if ttr < 1:
            factors += (1 - ttr) / (1 - threshold)
    return len(toks) / factors if factors > 0 else np.nan

def mtld(text):
    toks = tokens(text)
    if len(toks) < 50:
        return np.nan
    return (mtld_dir(toks) + mtld_dir(toks[::-1])) / 2

# ---------------- load corpus ----------------
def load():
    rows = []
    for c in CONDS:
        with open(DATA.format(c)) as f:
            for line in f:
                r = json.loads(line)
                user_sents = None
                if "authorship" in r and "sentences" in r:
                    user_sents = [s.strip() for s, a in zip(r["sentences"], r["authorship"])
                                  if a == "U" and s.strip()]
                rows.append(dict(cond=c, prompt=r["title"].strip(),
                                 text=r["essay"].strip(), user_sents=user_sents))
    return rows

# ---------------- embeddings ----------------
def embed_all(rows, use_tfidf):
    """Attach essay-level and user-only embeddings to each row."""
    all_sent_lists = [split_sentences(r["text"]) for r in rows]
    if use_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        flat = [s for sl in all_sent_lists for s in sl] + \
               [s for r in rows if r["user_sents"] for s in r["user_sents"]]
        vec = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(flat)
        enc = lambda sents: np.asarray(vec.transform(sents).todense())
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")
        enc = lambda sents: model.encode(sents, show_progress_bar=False,
                                         normalize_embeddings=True)
    for r, sl in zip(rows, all_sent_lists):
        r["emb"] = enc(sl).mean(axis=0) if sl else None
        r["emb_user"] = (enc(r["user_sents"]).mean(axis=0)
                         if r["user_sents"] else None)
    return rows

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---------------- convergence ----------------
def loo_convergence(rows, emb_key="emb", conv_key="conv"):
    """Leave-one-out mean cosine within prompt x condition."""
    for p in sorted({r["prompt"] for r in rows}):
        for c in CONDS:
            grp = [r for r in rows if r["prompt"] == p and r["cond"] == c
                   and r.get(emb_key) is not None]
            if len(grp) < 3:
                continue
            E = np.vstack([r[emb_key] for r in grp])
            S = E @ E.T / (np.linalg.norm(E, axis=1)[:, None]
                           * np.linalg.norm(E, axis=1)[None, :] + 1e-12)
            for i, r in enumerate(grp):
                r[conv_key] = float((S[i].sum() - S[i, i]) / (len(grp) - 1))

# ---------------- statistics ----------------
def perm_test_convergence(rows, cond_a, cond_b):
    """Stratified permutation: shuffle labels within prompt, recompute LOO means."""
    per_prompt = []
    for p in sorted({r["prompt"] for r in rows}):
        grp = [r for r in rows if r["prompt"] == p and r["cond"] in (cond_a, cond_b)
               and r.get("emb") is not None]
        E = np.vstack([r["emb"] for r in grp])
        S = E @ E.T / (np.linalg.norm(E, axis=1)[:, None]
                       * np.linalg.norm(E, axis=1)[None, :] + 1e-12)
        labels = np.array([r["cond"] == cond_b for r in grp])
        per_prompt.append((S, labels))

    def stat(label_sets):
        va, vb = [], []
        for (S, _), lab in zip(per_prompt, label_sets):
            for grp_mask, sink in ((~lab, va), (lab, vb)):
                idx = np.where(grp_mask)[0]
                if len(idx) < 3:
                    continue
                sub = S[np.ix_(idx, idx)]
                loo = (sub.sum(axis=1) - np.diag(sub)) / (len(idx) - 1)
                sink.extend(loo)
        return np.mean(vb) - np.mean(va)

    obs = stat([lab for _, lab in per_prompt])
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        null[k] = stat([RNG.permutation(lab) for _, lab in per_prompt])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    return obs, p

def perm_test_scores(a_scores, a_prompts, b_scores, b_prompts):
    """Stratified permutation on per-essay scores (for MTLD)."""
    df = pd.DataFrame(dict(score=np.r_[a_scores, b_scores],
                           prompt=np.r_[a_prompts, b_prompts],
                           is_b=np.r_[np.zeros(len(a_scores)), np.ones(len(b_scores))]))
    df = df.dropna()
    obs = df.loc[df.is_b == 1, "score"].mean() - df.loc[df.is_b == 0, "score"].mean()
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        perm = df.groupby("prompt")["is_b"].transform(
            lambda x: RNG.permutation(x.values))
        null[k] = df.loc[perm == 1, "score"].mean() - df.loc[perm == 0, "score"].mean()
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    return obs, p

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                 / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sp

def boot_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    diffs = [RNG.choice(b, len(b)).mean() - RNG.choice(a, len(a)).mean()
             for _ in range(N_BOOT)]
    return np.percentile(diffs, [2.5, 97.5])

def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    m = len(pvals)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf", action="store_true", help="debug mode, no SBERT")
    args = ap.parse_args()
    mode = "TF-IDF (DEBUG — not for the paper)" if args.tfidf else "SBERT all-mpnet-base-v2"

    rows = load()
    print(f"Loaded {len(rows)} essays. Embedding with {mode}...")
    for r in rows:
        r["mtld"] = mtld(r["text"])
    embed_all(rows, args.tfidf)
    loo_convergence(rows, "emb", "conv")
    loo_convergence(rows, "emb_user", "conv_user")

    out = [f"STUDY 1 RESULTS — embedding mode: {mode}", "=" * 70]

    def vals(cond, key):
        v = [r.get(key, np.nan) for r in rows if r["cond"] == cond]
        return np.array([x if x is not None else np.nan for x in v], float)

    out.append(f"{'condition':<13}{'n':>4}{'MTLD':>16}{'convergence':>18}{'user-only conv':>18}")
    for c in CONDS:
        m, v, u = vals(c, "mtld"), vals(c, "conv"), vals(c, "conv_user")
        fm = lambda x: f"{np.nanmean(x):.4f} ({np.nanstd(x):.4f})" if not np.all(np.isnan(x)) else "—"
        out.append(f"{c:<13}{np.sum(~np.isnan(v)):>4}"
                   f"{np.nanmean(m):>9.1f} ({np.nanstd(m):.1f})"
                   f"{fm(v):>18}{fm(u):>18}")

    # PRIMARY: solo vs instructgpt
    out.append("\nPRIMARY CONFIRMATORY (solo vs instructgpt, Holm-corrected family of 2)")
    prompts = lambda c: [r["prompt"] for r in rows if r["cond"] == c]
    obs_c, p_c = perm_test_convergence(rows, "solo", "instructgpt")
    obs_m, p_m = perm_test_scores(vals("solo", "mtld"), prompts("solo"),
                                  vals("instructgpt", "mtld"), prompts("instructgpt"))
    adj = holm(np.array([p_c, p_m]))
    d_c = cohens_d(vals("solo", "conv"), vals("instructgpt", "conv"))
    d_m = cohens_d(vals("solo", "mtld"), vals("instructgpt", "mtld"))
    ci_c = boot_ci(vals("solo", "conv"), vals("instructgpt", "conv"))
    ci_m = boot_ci(vals("solo", "mtld"), vals("instructgpt", "mtld"))
    out.append(f"  Convergence: diff={obs_c:+.4f}, d={d_c:+.2f}, "
               f"95% CI [{ci_c[0]:+.4f}, {ci_c[1]:+.4f}], perm p={p_c:.4f}, Holm p={adj[0]:.4f}")
    out.append(f"  MTLD:        diff={obs_m:+.2f},  d={d_m:+.2f}, "
               f"95% CI [{ci_m[0]:+.2f}, {ci_m[1]:+.2f}], perm p={p_m:.4f}, Holm p={adj[1]:.4f}")
    dissoc = (adj[0] < .05) and (adj[1] >= .05) and (abs(d_m) < 0.3)
    out.append(f"  DECISION RULE (dissociation supported?): {'YES' if dissoc else 'NO'}")

    # SECONDARY
    out.append("\nSECONDARY / EXPLORATORY (no confirmatory claims)")
    obs_c2, p_c2 = perm_test_convergence(rows, "solo", "gpt3")
    obs_m2, p_m2 = perm_test_scores(vals("solo", "mtld"), prompts("solo"),
                                    vals("gpt3", "mtld"), prompts("gpt3"))
    out.append(f"  solo vs gpt3 — convergence: diff={obs_c2:+.4f}, "
               f"d={cohens_d(vals('solo','conv'), vals('gpt3','conv')):+.2f}, p={p_c2:.4f}")
    out.append(f"  solo vs gpt3 — MTLD:        diff={obs_m2:+.2f},  "
               f"d={cohens_d(vals('solo','mtld'), vals('gpt3','mtld')):+.2f}, p={p_m2:.4f}")
    for c in ["gpt3", "instructgpt"]:
        whole = vals(c, "conv"); user = vals(c, "conv_user")
        mask = ~np.isnan(whole) & ~np.isnan(user)
        w, pw = stats.wilcoxon(whole[mask], user[mask])
        out.append(f"  {c}: whole-essay vs user-only convergence "
                   f"{np.nanmean(whole):.4f} vs {np.nanmean(user):.4f}, "
                   f"Wilcoxon p={pw:.2e} (mechanism, exploratory)")
        d_us = cohens_d(vals("solo", "conv"), user)
        out.append(f"  {c}: user-only vs SOLO whole-essay convergence d={d_us:+.2f} (exploratory)")

    report = "\n".join(out)
    print("\n" + report)
    with open("study1_results.txt", "w") as f:
        f.write(report + "\n")
    pd.DataFrame([{k: r.get(k) for k in
                   ("cond", "prompt", "mtld", "conv", "conv_user")}
                  for r in rows]).to_csv("study1_per_essay.csv", index=False)
    print("\nSaved: study1_results.txt, study1_per_essay.csv")

if __name__ == "__main__":
    sys.exit(main())
