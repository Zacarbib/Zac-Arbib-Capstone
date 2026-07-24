"""
Study 2 generation script
=========================
Implements the pre-specified protocol locked 7 July 2026
(see Study2_PreSpecified_Plan.md). Do not modify conditions, personas,
templates, or exclusion rules without logging a deviation.

Usage (Colab or local):
  pip install openai
  python study2_generate.py --pilot   # 20 calls, format check only, NOT analysed
  python study2_generate.py           # full run, 200 calls

The script asks for your OpenAI API key at runtime (never hardcode it),
prints the 10 prompts and an estimate, and requires you to type 'yes'
before spending anything.

Outputs:
  study2_raw.jsonl        every API response, appended immediately (crash-safe)
  study2_essays.jsonl     the analysed corpus (cond, prompt title, slot, text)
  study2_generation_log.json   model snapshot, params, usage, regenerations
"""

import argparse, json, os, re, sys, time
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

MODEL = "gpt-5.5"          # pinned per plan section 2; snapshot recorded from responses
REASONING_EFFORT = "low"   # pre-specified, plan section 2
MAX_COMPLETION_TOKENS = 3000
SEED_BASE = 20260707
N_PER_CELL = 10
SLEEP_BETWEEN = 0.5
WORD_MIN, WORD_MAX = 150, 600

CANDIDATE_DIRS = ["processed_data", "hai-diversity-main/processed_data"]

PERSONAS = [
    "a retired secondary school teacher who values tradition and discipline",
    "a 19-year-old university student who spends most evenings gaming online",
    "an environmental scientist who weighs evidence carefully",
    "a small business owner focused on practical costs and benefits",
    "a stand-up comedian who approaches serious topics through humour",
    "a parent of three teenagers worried about screen time",
    "a professional athlete who believes in routine and sacrifice",
    "a public librarian who champions free access to information",
    "a software engineer sceptical of technology hype",
    "a poet who values ambiguity and personal experience",
]

BASE_INSTRUCTION = (
    "Write a short opinion essay of approximately 300 words in response to the "
    "prompt below. Write in plain prose, without headings or bullet points.\n\n"
    "Prompt:\n{prompt}"
)
PERSONA_SUFFIX = (
    "\n\nWrite the essay in the voice of {persona}: express the opinions, "
    "examples and reasoning this person would plausibly offer. Do not mention "
    "that you are adopting a persona."
)

def word_count(text):
    return len(re.findall(r"[A-Za-z']+", text))

def load_prompts():
    data_dir = next((d for d in CANDIDATE_DIRS if os.path.isdir(d)), None)
    if not data_dir:
        sys.exit("Cannot find processed_data/ - run next to or inside hai-diversity-main")
    prompts = {}
    with open(f"{data_dir}/essays_solo.jsonl") as f:
        for line in f:
            r = json.loads(line)
            prompts.setdefault(r["title"].strip(), r["prompt"].strip())
    if len(prompts) != 10:
        sys.exit(f"Expected 10 unique prompts, found {len(prompts)}")
    return prompts

def build_messages(prompt_text, persona=None):
    content = BASE_INSTRUCTION.format(prompt=prompt_text)
    if persona:
        content += PERSONA_SUFFIX.format(persona=persona)
    return [{"role": "user", "content": content}]

def call_api(client, messages, seed):
    kwargs = dict(model=MODEL, messages=messages, seed=seed,
                  max_completion_tokens=MAX_COMPLETION_TOKENS)
    try:
        return client.chat.completions.create(reasoning_effort=REASONING_EFFORT, **kwargs)
    except TypeError:
        # SDK/model without reasoning_effort support: log a deviation, proceed
        with open("study2_deviations.md", "a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}: reasoning_effort "
                    f"not accepted; call made without it.\n")
        return client.chat.completions.create(**kwargs)

def generate(client, messages, seed, raw_fh, meta):
    for attempt in range(3):
        try:
            resp = call_api(client, messages, seed)
            break
        except Exception as e:
            if attempt == 2:
                raise
            wait = 5 * (attempt + 1)
            print(f"    API error ({e}); retrying in {wait}s")
            time.sleep(wait)
    text = (resp.choices[0].message.content or "").strip()
    record = dict(meta, response_id=resp.id, model=resp.model,
                  system_fingerprint=getattr(resp, "system_fingerprint", None),
                  usage=resp.usage.model_dump() if resp.usage else None,
                  text=text, word_count=word_count(text),
                  utc=datetime.now(timezone.utc).isoformat())
    raw_fh.write(json.dumps(record) + "\n")
    raw_fh.flush()
    return record

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="1 essay per prompt per condition (20 calls), format check only")
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Run: pip install openai")

    key = os.environ.get("OPENAI_API_KEY") or getpass("Paste your OpenAI API key: ")
    client = OpenAI(api_key=key)

    prompts = load_prompts()
    n_slots = 1 if args.pilot else N_PER_CELL
    total_calls = len(prompts) * n_slots * 2
    mode = "PILOT (format check only - outputs are NOT analysed)" if args.pilot else "FULL RUN"

    print(f"\n=== Study 2 generation - {mode} ===")
    print(f"Model {MODEL}, reasoning_effort={REASONING_EFFORT}, "
          f"{total_calls} calls, ~300-word essays.")
    print("\nPrompts to be used verbatim from the corpus:")
    for i, t in enumerate(prompts, 1):
        print(f"  {i:2d}. {t}")
    if input("\nType 'yes' to proceed and spend API credit: ").strip().lower() != "yes":
        sys.exit("Aborted. Nothing generated, nothing spent.")

    raw_path = "study2_raw_pilot.jsonl" if args.pilot else "study2_raw.jsonl"
    essays, regens, excluded = [], [], []
    call_i = 0
    with open(raw_path, "a") as raw_fh:
        for cond in ["default", "persona"]:
            for title, ptext in prompts.items():
                for slot in range(n_slots):
                    call_i += 1
                    persona = PERSONAS[slot] if cond == "persona" else None
                    meta = dict(cond=cond, prompt_title=title, slot=slot,
                                persona=persona, pilot=args.pilot)
                    print(f"[{call_i}/{total_calls}] {cond} | {title[:45]} | slot {slot}")
                    rec = generate(client, build_messages(ptext, persona),
                                   SEED_BASE + call_i, raw_fh, meta)
                    if not (WORD_MIN <= rec["word_count"] <= WORD_MAX):
                        regens.append(dict(meta, first_wc=rec["word_count"]))
                        print(f"    out of range ({rec['word_count']} words) - regenerating once")
                        rec = generate(client, build_messages(ptext, persona),
                                       SEED_BASE + 100000 + call_i, raw_fh,
                                       dict(meta, regeneration=True))
                        if not (WORD_MIN <= rec["word_count"] <= WORD_MAX):
                            excluded.append(dict(meta, final_wc=rec["word_count"]))
                            print("    still out of range - EXCLUDED (logged)")
                            continue
                    essays.append({k: rec[k] for k in
                                   ("cond", "prompt_title", "slot", "persona",
                                    "text", "word_count", "model")})
                    time.sleep(SLEEP_BETWEEN)

    if not args.pilot:
        with open("study2_essays.jsonl", "w") as f:
            for e in essays:
                f.write(json.dumps(e) + "\n")

    log = dict(mode=mode, model_config=MODEL,
               reasoning_effort=REASONING_EFFORT, seed_base=SEED_BASE,
               calls=call_i, kept=len(essays), regenerations=regens,
               exclusions=excluded,
               finished_utc=datetime.now(timezone.utc).isoformat())
    log_path = "study2_generation_log_pilot.json" if args.pilot else "study2_generation_log.json"
    json.dump(log, open(log_path, "w"), indent=2)

    print(f"\nDone. Kept {len(essays)}/{call_i} generations; "
          f"{len(regens)} regenerated, {len(excluded)} excluded.")
    print(f"Saved: {raw_path}, {log_path}"
          + ("" if args.pilot else ", study2_essays.jsonl"))
    if args.pilot:
        print("\nPILOT RULE: check word counts and that outputs read as essays. "
              "Do NOT analyse pilot outputs for convergence or diversity.")

if __name__ == "__main__":
    main()
