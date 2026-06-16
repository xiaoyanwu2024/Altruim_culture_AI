# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: DG data collection — batch runner for all models on the Dictator Game
# =============================================================================

"""
Multi-model TPP data collection.
Models: Claude Haiku, GPT-4o-mini, Llama-3.3-70B, Gemini Flash, Mistral

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."
  export OPENAI_API_KEY="sk-..."
  export GROQ_API_KEY="gsk_..."
  export GOOGLE_API_KEY="AIza..."
  python3 collect_all_models.py
"""

import os, time, csv
from pathlib import Path
from datetime import datetime
import pandas as pd

# ─── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    "claude_haiku": {
        "provider":    "anthropic",
        "model_id":    "claude-haiku-4-5-20251001",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gpt4o_mini": {
        "provider":    "openai",
        "model_id":    "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "llama3_70b": {
        "provider":    "groq",
        "model_id":    "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "llama4_scout": {
        "provider":    "groq",
        "model_id":    "meta-llama/llama-4-scout-17b-16e-instruct",
        "api_key_env": "GROQ_API_KEY",
    },
    "mistral": {
        "provider":    "groq",
        "model_id":    "mistral-saba-24b",
        "api_key_env": "GROQ_API_KEY",
    },
    "gemini_flash": {
        "provider":    "google",
        "model_id":    "gemini-1.5-flash",
        "api_key_env": "GOOGLE_API_KEY",
    },
}

COUNTRIES  = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS    = ["male", "female"]
REPETITIONS = 20
TP_TOKENS  = 50

# ─── Prompts ──────────────────────────────────────────────────────────────────

def system_prompt(country, gender):
    return (
        f"You are a 23-year-old {gender} university student from {country}. "
        f"You were born in {country}, hold {country} nationality, and currently live there. "
        "You are not majoring in psychology or any related fields. "
        "It is October 2022. "
        "You are participating in an online behavioral economics experiment hosted on Prolific. "
        "You have voluntarily agreed to take part, and you have normal vision. "
        "You have no history of psychiatric or neurological illness. "
        "The base payment is £2. Tokens earned during the game are exchanged for pence at a rate of 2.75 tokens = 1 penny. "
        "At the end of the study, 10% of rounds are randomly selected to be implemented for real payment. "
        "Each round is equally likely to be selected, so treat every decision seriously.\n\n"

        "GAME INSTRUCTIONS\n"
        "Earlier in this experiment, you completed a single-round allocation game as the Allocator. "
        "In that game, there were two anonymous players: an Allocator and a Receiver. "
        "The Allocator had 100 tokens and decided how many to give to the Receiver (from 8 to 50 tokens); "
        "the Receiver had no choice and simply accepted. "
        "You played the role of Allocator and chose how many tokens to share with the Receiver. "
        "If you gave X tokens, your final tokens in that round were 100 - X.\n\n"
        "You will now play a multiple-round game. "
        "In each round, you will see how 100 tokens were divided between two other participants: an Allocator and a Receiver. "
        "The Allocator had the right to decide how to split 100 tokens between himself/herself and the Receiver. "
        "The Receiver could only accept the Allocator's decision. "
        "The allocators and receivers are real participants from a previous study — "
        "you will not be the Allocator or the Receiver in any round. "
        "Each round involves a different pair of participants.\n\n"
        "YOUR ROLE — THIRD PARTY\n"
        "You are the Third Party. You start each round with 50 tokens. "
        "After seeing the allocation outcome, you make decisions in two scenarios:\n\n"
        "Scenario 1 — REDUCE: You decide whether to reduce the Allocator's tokens at a cost to yourself.\n"
        "  - Yes: you spend X tokens (your tokens decrease by X); the Allocator loses X x multiplier tokens. "
        "The multiplier will be shown each round (either 1.5 or 3.0).\n"
        "  - No: nothing changes.\n"
        "  Note: this decision only affects you and the Allocator. The Receiver's tokens are unchanged.\n"
        "  Final tokens — Allocator: original - (cost x multiplier); Receiver: unchanged; You: 50 - cost.\n\n"
        "Scenario 2 — INCREASE: You decide whether to increase the Receiver's tokens at a cost to yourself.\n"
        "  - Yes: you spend X tokens (your tokens decrease by X); the Receiver gains X x multiplier tokens.\n"
        "  - No: nothing changes.\n"
        "  Note: this decision only affects you and the Receiver. The Allocator's tokens are unchanged.\n"
        "  Final tokens — Allocator: unchanged; Receiver: original + (cost x multiplier); You: 50 - cost.\n\n"
        "Answer every decision question with exactly one word: Yes or No."
    )

def user_prompt(row):
    x1     = int(row.x1)
    x2     = int(row.x2)
    cost   = int(row.cost)
    effect = int(cost * float(row.ratio))
    block  = row.block_type

    context = (
        "In this experiment you observe an allocation game between two anonymous participants:\n"
        "- The Allocator started with 100 tokens and decided how many to give to the Receiver.\n"
        "- The Receiver accepted whatever the Allocator decided to give.\n\n"
    )

    if block.strip("'").upper() == "REDUCE":
        scenario = (
            f"Round outcome:\n"
            f"  The Allocator gave {x2} tokens to the Receiver and kept {x1} tokens for themselves.\n"
            f"  Allocator now has: {x1} tokens\n"
            f"  Receiver now has:  {x2} tokens\n"
            f"  You have:          {TP_TOKENS} tokens (refreshed each round)\n\n"
            "This is the REDUCE scenario. You decide whether to reduce the Allocator's tokens.\n\n"
            f"If YES: Allocator ends with {x1-effect}, Receiver ends with {x2}, You end with {TP_TOKENS-cost}.\n"
            f"If NO:  Allocator ends with {x1}, Receiver ends with {x2}, You end with {TP_TOKENS}.\n\n"
            "Do you choose to intervene? Answer Yes or No."
        )
    else:
        scenario = (
            f"Round outcome:\n"
            f"  The Allocator gave {x2} tokens to the Receiver and kept {x1} tokens for themselves.\n"
            f"  Allocator now has: {x1} tokens\n"
            f"  Receiver now has:  {x2} tokens\n"
            f"  You have:          {TP_TOKENS} tokens (refreshed each round)\n\n"
            "This is the INCREASE scenario. You decide whether to increase the Receiver's tokens.\n\n"
            f"If YES: Allocator ends with {x1}, Receiver ends with {x2+effect}, You end with {TP_TOKENS-cost}.\n"
            f"If NO:  Allocator ends with {x1}, Receiver ends with {x2}, You end with {TP_TOKENS}.\n\n"
            "Do you choose to intervene? Answer Yes or No."
        )
    return context + scenario

# ─── Provider callers ─────────────────────────────────────────────────────────

def make_caller(provider, model_id, api_key):
    if provider == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=api_key)
        def call(sp, up):
            r = c.messages.create(
                model=model_id, max_tokens=10,
                system=sp,
                messages=[{"role": "user", "content": up}]
            )
            return r.content[0].text.strip()

    elif provider == "openai":
        from openai import OpenAI
        c = OpenAI(api_key=api_key)
        def call(sp, up):
            r = c.chat.completions.create(
                model=model_id, max_tokens=10,
                messages=[{"role":"system","content":sp},
                          {"role":"user","content":up}]
            )
            return r.choices[0].message.content.strip()

    elif provider == "groq":
        from groq import Groq
        c = Groq(api_key=api_key)
        def call(sp, up):
            r = c.chat.completions.create(
                model=model_id, max_tokens=10,
                messages=[{"role":"system","content":sp},
                          {"role":"user","content":up}]
            )
            return r.choices[0].message.content.strip()

    elif provider == "google":
        from google import genai
        c = genai.Client(api_key=api_key)
        def call(sp, up):
            r = c.models.generate_content(
                model=model_id,
                contents=sp + "\n\n" + up,
            )
            return r.text.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return call

def parse_response(text):
    t = text.lower()
    if "yes" in t: return "Yes", 1
    if "no"  in t: return "No",  0
    return text, None

# ─── Conditions ───────────────────────────────────────────────────────────────

def load_conditions():
    df = pd.read_excel("TPP_300Trials.xlsx")
    df["block_type"] = df["block_type"].str.strip("'")
    df["cond_id"] = range(len(df))
    return df

# ─── Run one model ────────────────────────────────────────────────────────────

def run_model(name, cfg, conditions, countries, genders, reps, out_dir):
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        print(f"  SKIP {name}: {cfg['api_key_env']} not set")
        return None

    call = make_caller(cfg["provider"], cfg["model_id"], api_key)

    out_path = out_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields   = ["model","country","gender","cond_id","block_type",
                "x1","x2","cost","ratio","rep","response","intervene"]
    total    = len(countries) * len(genders) * len(conditions) * reps
    done     = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for country in countries:
            for gender in genders:
                sp = system_prompt(country, gender)
                shuffled = conditions.sample(frac=1, random_state=None).reset_index(drop=True)

                for _, cond in shuffled.iterrows():
                    up = user_prompt(cond)
                    for rep in range(reps):
                        raw, resp, intervene = None, None, None
                        for attempt in range(4):
                            try:
                                raw  = call(sp, up)
                                resp, intervene = parse_response(raw)
                                break
                            except Exception as e:
                                wait = 2 ** attempt
                                print(f"    [{name}] retry {attempt+1}: {e} — wait {wait}s")
                                time.sleep(wait)

                        writer.writerow({
                            "model": name, "country": country, "gender": gender,
                            "cond_id": int(cond["cond_id"]), "block_type": cond["block_type"],
                            "x1": int(cond["x1"]), "x2": int(cond["x2"]),
                            "cost": int(cond["cost"]), "ratio": float(cond["ratio"]),
                            "rep": rep, "response": resp, "intervene": intervene,
                        })
                        f.flush()
                        done += 1

                pct = 100 * done / total
                print(f"  [{name}] {country} {gender} done — {done}/{total} ({pct:.0f}%)")

    print(f"  Saved → {out_path}\n")
    return out_path

# ─── Main ─────────────────────────────────────────────────────────────────────

def main(
    models_to_run=None,
    countries=COUNTRIES,
    genders=GENDERS,
    reps=REPETITIONS,
    out_dir="/Users/wuxiaoyan/Desktop/TPP_culture_AI/ai_study_data",
):
    conditions = load_conditions()
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    to_run = models_to_run or list(MODELS.keys())
    print(f"\nModels: {to_run}")
    print(f"Countries: {len(countries)} | Genders: {len(genders)} | "
          f"Conditions: {len(conditions)} | Reps: {reps}")
    print(f"Calls per model: {len(countries)*len(genders)*len(conditions)*reps:,}\n")

    for name in to_run:
        print(f"{'='*55}\n{name}  ({MODELS[name]['model_id']})\n{'='*55}")
        run_model(name, MODELS[name], conditions, countries, genders, reps, out)

    print("All done.")


if __name__ == "__main__":
    # ── Quick test: Claude only, 2 countries, 1 gender, 3 reps
    main(
        models_to_run=["claude_haiku"],
        countries=["Italy", "South Africa"],
        genders=["male"],
        reps=3,
    )
