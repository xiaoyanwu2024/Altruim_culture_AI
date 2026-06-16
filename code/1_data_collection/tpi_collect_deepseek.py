# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: TPI data collection — DeepSeek-V4-Flash (logprobs elicitation)
# =============================================================================

"""
DeepSeek V4-Flash Task 2 data collection.
Uses OpenAI-compatible API with logprobs (falls back to repeated sampling if unsupported).
8 countries × 2 genders × 300 conditions, parallel workers.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python3 collect_deepseek.py          # full run
    python3 collect_deepseek.py test     # test 5 rows only
"""

import os, sys, csv, math, time, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI

API_KEY    = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL   = "https://api.deepseek.com"
MODEL_ID   = "deepseek-chat"   # non-thinking mode of V4-Flash
TP_TOKENS  = 50
MAX_WORKERS = 30

COUNTRIES = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS   = ["male","female"]

DATA_DIR = Path("ai_study_data")

# ── Prompts ────────────────────────────────────────────────────────────────────

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
    x1 = int(row.x1); x2 = int(row.x2)
    cost = int(row.cost); effect = int(cost * float(row.ratio))
    block = str(row.block_type).strip("'").upper()
    ctx = (
        "In this experiment you observe an allocation game between two anonymous participants:\n"
        "- The Allocator started with 100 tokens and decided how many to give to the Receiver.\n"
        "- The Receiver accepted whatever the Allocator decided to give.\n\n"
    )
    if block == "REDUCE":
        sc = (
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
        sc = (
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
    return ctx + sc

# ── Logprobs extraction ────────────────────────────────────────────────────────

def extract_p_yes(top_logprobs):
    lp_yes = lp_no = -100.0
    for item in top_logprobs:
        t = item.token.strip().lower()
        if t == "yes":  lp_yes = max(lp_yes, item.logprob)
        elif t == "no": lp_no  = max(lp_no,  item.logprob)
    p_yes_raw = math.exp(lp_yes); p_no_raw = math.exp(lp_no)
    total = p_yes_raw + p_no_raw
    return round(p_yes_raw / total, 6) if total > 1e-10 else None

# ── Worker ─────────────────────────────────────────────────────────────────────

def call_one(client, country, gender, cond):
    sp = system_prompt(country, gender)
    up = user_prompt(cond)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL_ID,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
                messages=[
                    {"role": "system", "content": sp},
                    {"role": "user",   "content": up},
                ],
            )
            choice = r.choices[0]
            p_yes = None
            if choice.logprobs and choice.logprobs.content:
                p_yes = extract_p_yes(choice.logprobs.content[0].top_logprobs)
            # fallback: parse text
            if p_yes is None:
                text = choice.message.content.strip().lower()
                if text.startswith("yes"): p_yes = 1.0
                elif text.startswith("no"): p_yes = 0.0
            return p_yes
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [deepseek] retry {attempt+1}: {e} — wait {wait}s")
            time.sleep(wait)
    return None

def run_task(args):
    client, country, gender, cond = args
    p_yes = call_one(client, country, gender, cond)
    return {
        "model": "deepseek_v4flash",
        "country": country, "gender": gender,
        "cond_id": int(cond["cond_id"]),
        "block_type": str(cond["block_type"]).strip("'"),
        "x1": int(cond["x1"]), "x2": int(cond["x2"]),
        "cost": int(cond["cost"]), "ratio": float(cond["ratio"]),
        "p_yes": p_yes,
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def main(test=False):
    if not API_KEY:
        print("ERROR: set DEEPSEEK_API_KEY"); return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    conditions = pd.read_excel("TPP_300Trials.xlsx")
    conditions["block_type"] = conditions["block_type"].str.strip("'")
    conditions["cond_id"] = range(len(conditions))

    tasks = []
    for country in COUNTRIES:
        for gender in GENDERS:
            for _, cond in conditions.iterrows():
                tasks.append((client, country, gender, cond))

    if test:
        tasks = tasks[:5]
        print("TEST MODE: 5 rows only")

    total = len(tasks)
    out_path = DATA_DIR / f"deepseek_v4flash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields = ["model","country","gender","cond_id","block_type","x1","x2","cost","ratio","p_yes"]

    print(f"Model: {MODEL_ID}")
    print(f"Total: {total} requests | Workers: {MAX_WORKERS}")
    print(f"Output: {out_path}\n")

    done = 0
    lock = threading.Lock()
    start = time.time()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(run_task, t): t for t in tasks}
            for future in as_completed(futures):
                row = future.result()
                with lock:
                    writer.writerow(row)
                    f.flush()
                    done += 1
                    if done % 200 == 0 or (test and done % 1 == 0):
                        elapsed = (time.time() - start) / 60
                        rate = done / elapsed if elapsed > 0 else 0
                        remaining = (total - done) / rate if rate > 0 else 0
                        print(f"  {done}/{total} ({100*done/total:.1f}%) — "
                              f"{rate:.0f} rows/min — ~{remaining:.0f} min left")

    print(f"\nDone → {out_path}")

if __name__ == "__main__":
    main(test="test" in sys.argv)
