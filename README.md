# Zac-Arbib-Capstone
# Lexical Diversity, Semantic Convergence and Cognitive Residue in AI-Assisted Writing

Code and data for the LIS MASc Capstone 3 white paper (submitted 2 August 2026).
Two studies on a single measurement pipeline: Study 1 reanalyses the public
Padmakumar and He (2024) co-writing corpus; Study 2 generates and analyses
200 essays from a pinned frontier model on the same ten prompts.

## How to reproduce

Runs on Google Colab or any Python 3 environment:

```
pip install sentence-transformers scipy scikit-learn pandas matplotlib
unzip hai-diversity-main.zip
python study1_pipeline.py        # Study 1 confirmatory run
python study2_analysis.py        # Study 2 confirmatory run (uses study2_essays.jsonl)
python robustness_checks.py      # post hoc sensitivity analyses (optional; ~1 h)
```

Do NOT rerun `study2_generate.py`: it would call the OpenAI API and produce a
new corpus. The analysed corpus of record is `study2_essays.jsonl`, generated
once on 15 July 2026 under the pre-specified protocol; every raw API response
is preserved in `study2_raw.jsonl`.

An independent rerun on a fresh runtime on 20 July 2026 reproduced every
confirmatory statistic identically.

## Files

| File | Role |
|---|---|
| `study1_pipeline.py` | Study 1 analysis; header docstring is the locked plan (6 July 2026) |
| `study2_generate.py` | Study 2 generation protocol, locked 7 July 2026 (do not rerun) |
| `study2_analysis.py` | Study 2 analysis, amended version of record (punctuation normalisation) |
| `study2_analysis_PREAMENDMENT.py` | Pre-amendment copy retained for audit |
| `robustness_checks.py` | Post hoc sensitivity analyses (21 July 2026) |
| `Study1_run.ipynb`, `Study2_run.ipynb` | Colab runners used for the recorded runs |
| `hai-diversity-main.zip` | Padmakumar and He (2024) public corpus release, unmodified |
| `study2_essays.jsonl` | Generated corpus of record (200 essays) |
| `study2_raw.jsonl`, `study2_raw_pilot.jsonl` | Raw API responses (full run; pilot) |
| `study1_results.txt`, `study2_results.txt` | Confirmatory statistics as produced |
| `robustness_results.txt` | Sensitivity analyses output |
| `*_per_essay.csv`, `entropy_sensitivity.csv`, `fig_spectrum.png` | Per-essay measures, entropy sensitivity, spectrum figure |
| `requirements.txt` | Package versions from the analysis runtime |

## Key parameters

- Model: OpenAI GPT-5.5, pinned snapshot `[INSERT exact string from the model field of study2_essays.jsonl]`, reasoning effort low
- Seeds: 20260706 (Study 1), 20260707 (Study 2 generation and analysis), 20260721 (robustness)
- Statistics: stratified within-prompt permutation tests (10,000), bootstrap CIs (10,000), Holm correction as pre-specified

## Data provenance and ethics

Study 1 data are the publicly released, anonymised corpus of Padmakumar, V.
and He, H. (2024), "Does Writing with Language Models Reduce Content
Diversity?", included here unmodified as permitted by its public release;
please cite the original authors for the corpus. Study 2 data are
model-generated text. No participants were recruited for this project.
[LIS ethics status line to match the paper.]
