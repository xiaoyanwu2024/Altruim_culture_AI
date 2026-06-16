# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: TPI data collection — GPT-5.4-mini (direct probability elicitation)
# =============================================================================

"""
GPT-5.4-mini Task 2 — direct probability elicitation.
Model outputs 0-100 integer per trial.
8 countries x 2 genders x 300 conditions = 4800 requests.

Usage:
    export OPENAI_API_KEY="..."
    python3 collect_gpt54mini_prob.py          # full run
    python3 collect_gpt54mini_prob.py test     # test 5 rows only
"""

import os, sys, csv, re, time, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI

API_KEY    = os.environ.get("OPENAI_API_KEY", "")
MODEL_ID   = "gpt-5.4-mini"
TP_TOKENS  = 50
MAX_WORKERS = 60

COUNTRIES = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS   = ["male","female"]
DATA_DIR  = Path("ai_study_data")

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
        "For each decision, output a single integer from 0 to 100 representing the probability "
        "that you would choose to intervene (0 = definitely No, 100 = definitely Yes). "
        "Output the number only. Nothing else."
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
            "What is the probability (0-100) that you would choose to intervene? Output a single integer only."
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
            "What is the probability (0-100) that you would choose to intervene? Output a single integer only."
        )
    return ctx + sc

def parse_prob(text):
    nums = re.findall(r'\b(\d{1,3})\b', text.strip())
    for n in nums:
        v = int(n)
        if 0 <= v <= 100:
            return round(v / 100, 6)
    return None

def call_one(client, country, gender, cond):
    sp = system_prompt(country, gender)
    up = user_prompt(cond)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL_ID,
                max_completion_tokens=20,
                messages=[
                    {"role": "system", "content": sp},
                    {"role": "user",   "content": up},
                ],
            )
            raw = r.choices[0].message.content.strip()
            return parse_prob(raw)
        except Exception as e:
            err = str(e)
            wait = 30 if "429" in err or "rate" in err.lower() else 2 ** attempt
            print(f"  [gpt54mini] retry {attempt+1}: {err[:80]} — wait {wait}s")
            time.sleep(wait)
    return None

def run_task(args):
    client, country, gender, cond = args
    p_yes = call_one(client, country, gender, cond)
    return {
        "model": "gpt54mini_prob",
        "country": country, "gender": gender,
        "cond_id": int(cond["cond_id"]),
        "block_type": str(cond["block_type"]).strip("'"),
        "x1": int(cond["x1"]), "x2": int(cond["x2"]),
        "cost": int(cond["cost"]), "ratio": float(cond["ratio"]),
        "p_yes": p_yes,
    }

def main(test=False):
    if not API_KEY:
        print("ERROR: set OPENAI_API_KEY"); return

    client = OpenAI(api_key=API_KEY)
    conditions = pd.read_excel("TPP_300Trials.xlsx")
    conditions["block_type"] = conditions["block_type"].str.strip("'")
    conditions["cond_id"] = range(len(conditions))

    tasks = []
    for country in COUNTRIES:
        for gender in GENDERS:
            for _, cond in conditions.iterrows():
                tasks.append((client, country, gender, cond))

    if test:
        tasks = tasks[:10]
        print("TEST MODE: 10 rows only")

    total = len(tasks)
    out_path = DATA_DIR / f"gpt54mini_prob_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields = ["model","country","gender","cond_id","block_type","x1","x2","cost","ratio","p_yes"]

    print(f"Model: {MODEL_ID}  (direct probability elicitation)")
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
