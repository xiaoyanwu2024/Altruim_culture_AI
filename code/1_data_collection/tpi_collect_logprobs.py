# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: TPI data collection — logprobs elicitation for models supporting token log-probabilities
# =============================================================================

"""
Logprobs-based data collection for models that support it.
Extracts P(Yes) directly from token log-probabilities — one call per condition.

Supported: OpenAI (gpt-4o-mini), Groq (llama-3.3-70b, llama-4-scout)

P(Yes) = exp(logprob_yes) / (exp(logprob_yes) + exp(logprob_no))
If only one of Yes/No appears in top_logprobs, the other gets -100 (≈0).
"""

import os, csv, time, math
from pathlib import Path
from datetime import datetime
import pandas as pd

COUNTRIES = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS   = ["male", "female"]
TP_TOKENS = 50

MODELS = {
    "gpt4o_mini_logprobs": {
        "provider":    "openai",
        "model_id":    "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt4o_logprobs": {
        "provider":    "openai",
        "model_id":    "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
}


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


def extract_p_yes(top_logprobs):
    """
    Given a list of top_logprob objects, extract logprob for Yes and No tokens,
    then compute normalized P(Yes).
    Takes the MAX logprob across all yes/no variants to avoid dict-overwrite bugs
    when ' Yes' and 'Yes' both appear in top_logprobs.
    """
    lp_yes = -100.0
    lp_no  = -100.0
    for item in top_logprobs:
        t = item.token.strip().lower()
        if t == "yes":
            lp_yes = max(lp_yes, item.logprob)
        elif t == "no":
            lp_no  = max(lp_no,  item.logprob)

    p_yes_raw = math.exp(lp_yes)
    p_no_raw  = math.exp(lp_no)
    total     = p_yes_raw + p_no_raw

    if total < 1e-10:
        return None   # neither Yes nor No in top tokens
    return round(p_yes_raw / total, 6)


def make_openai_caller(model_id, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    def call(sp, up):
        r = client.chat.completions.create(
            model=model_id,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            messages=[{"role": "system", "content": sp},
                      {"role": "user",   "content": up}],
        )
        top_lp = r.choices[0].logprobs.content[0].top_logprobs
        return extract_p_yes(top_lp)
    return call


def make_groq_caller(model_id, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)
    def call(sp, up):
        r = client.chat.completions.create(
            model=model_id,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            messages=[{"role": "system", "content": sp},
                      {"role": "user",   "content": up}],
        )
        top_lp = r.choices[0].logprobs.content[0].top_logprobs
        return extract_p_yes(top_lp)
    return call


def load_conditions():
    df = pd.read_excel("TPP_300Trials.xlsx")
    df["block_type"] = df["block_type"].str.strip("'")
    df["cond_id"] = range(len(df))
    return df


def run_logprobs(model_name, cfg, conditions, out_dir):
    import os
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        print(f"  SKIP {model_name}: {cfg['api_key_env']} not set")
        return

    if cfg["provider"] == "openai":
        call = make_openai_caller(cfg["model_id"], api_key)
    elif cfg["provider"] == "groq":
        call = make_groq_caller(cfg["model_id"], api_key)
    else:
        raise ValueError(f"Logprobs not supported for {cfg['provider']}")

    out_path = out_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields   = ["model","country","gender","cond_id","block_type",
                "x1","x2","cost","ratio","p_yes"]
    total    = len(COUNTRIES) * len(GENDERS) * len(conditions)
    done     = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for country in COUNTRIES:
            for gender in GENDERS:
                sp = system_prompt(country, gender)
                for _, cond in conditions.iterrows():
                    up = user_prompt(cond)
                    p_yes = None
                    for attempt in range(4):
                        try:
                            p_yes = call(sp, up)
                            break
                        except Exception as e:
                            wait = 2 ** attempt
                            print(f"  [{model_name}] retry {attempt+1}: {e} — wait {wait}s")
                            time.sleep(wait)

                    writer.writerow({
                        "model": model_name, "country": country, "gender": gender,
                        "cond_id": int(cond["cond_id"]), "block_type": cond["block_type"],
                        "x1": int(cond["x1"]), "x2": int(cond["x2"]),
                        "cost": int(cond["cost"]), "ratio": float(cond["ratio"]),
                        "p_yes": p_yes,
                    })
                    f.flush()
                    done += 1

                pct = 100 * done / total
                print(f"  [{model_name}] {country} {gender} — {done}/{total} ({pct:.0f}%)")

    print(f"  [{model_name}] Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    import os
    conditions = load_conditions()
    out_dir    = Path("/Users/wuxiaoyan/Desktop/TPP_culture_AI/ai_study_data")
    out_dir.mkdir(exist_ok=True)

    for name, cfg in MODELS.items():
        print(f"\n{'='*55}\n{name}  ({cfg['model_id']})\n{'='*55}")
        run_logprobs(name, cfg, conditions, out_dir)
