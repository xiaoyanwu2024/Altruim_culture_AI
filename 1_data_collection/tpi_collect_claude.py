# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: TPI data collection — Claude Haiku (direct probability elicitation)
# =============================================================================

"""
Claude (Haiku) Task 2 data collection — repeated sampling, parallel execution.
8 countries × 2 genders × 300 conditions × REPS repetitions.
Uses ThreadPoolExecutor for parallel API calls.
"""

import os, csv, time, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_ID  = "claude-haiku-4-5-20251001"
MODEL_NAME = "claude_haiku"
REPS      = 5
MAX_WORKERS = 10   # parallel threads
TP_TOKENS = 50

COUNTRIES = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS   = ["male", "female"]

# ── Prompts (identical to collect_all_models.py) ───────────────────────────────

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
    block  = str(row.block_type).strip("'").upper()
    context = (
        "In this experiment you observe an allocation game between two anonymous participants:\n"
        "- The Allocator started with 100 tokens and decided how many to give to the Receiver.\n"
        "- The Receiver accepted whatever the Allocator decided to give.\n\n"
    )
    if block == "REDUCE":
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


def parse_response(text):
    t = text.strip().lower()
    if t.startswith("yes"): return "Yes", 1
    if t.startswith("no"):  return "No",  0
    if "yes" in t: return "Yes", 1
    if "no"  in t: return "No",  0
    return text, None


# ── Per-call worker ────────────────────────────────────────────────────────────

def call_once(client, sp, up):
    for attempt in range(4):
        try:
            r = client.messages.create(
                model=MODEL_ID, max_tokens=10,
                system=sp,
                messages=[{"role": "user", "content": up}]
            )
            return r.content[0].text.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [claude] retry {attempt+1}: {e} — wait {wait}s")
            time.sleep(wait)
    return None


def run_task(args):
    """Single unit of work: one (country, gender, cond_id, rep) tuple."""
    client, country, gender, cond, rep = args
    sp = system_prompt(country, gender)
    up = user_prompt(cond)
    raw = call_once(client, sp, up)
    resp, intervene = parse_response(raw) if raw else (None, None)
    return {
        "model": MODEL_NAME, "country": country, "gender": gender,
        "cond_id": int(cond["cond_id"]), "block_type": str(cond["block_type"]).strip("'"),
        "x1": int(cond["x1"]), "x2": int(cond["x2"]),
        "cost": int(cond["cost"]), "ratio": float(cond["ratio"]),
        "rep": rep, "response": resp, "intervene": intervene,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set"); return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    conditions = pd.read_excel("TPP_300Trials.xlsx")
    conditions["block_type"] = conditions["block_type"].str.strip("'")
    conditions["cond_id"] = range(len(conditions))

    out_dir  = Path("ai_study_data")
    out_path = out_dir / f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields   = ["model","country","gender","cond_id","block_type",
                "x1","x2","cost","ratio","rep","response","intervene"]

    total = len(COUNTRIES) * len(GENDERS) * len(conditions) * REPS
    print(f"Model: {MODEL_ID}")
    print(f"Total calls: {total:,}  ({len(COUNTRIES)} countries × {len(GENDERS)} genders × "
          f"{len(conditions)} conditions × {REPS} reps)")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Output: {out_path}\n")

    # Build all tasks
    tasks = []
    for country in COUNTRIES:
        for gender in GENDERS:
            for _, cond in conditions.iterrows():
                for rep in range(REPS):
                    tasks.append((client, country, gender, cond, rep))

    done  = 0
    lock  = threading.Lock()
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
                    if done % 500 == 0:
                        elapsed = (time.time() - start) / 60
                        rate = done / elapsed
                        remaining = (total - done) / rate
                        print(f"  {done}/{total} ({100*done/total:.1f}%) — "
                              f"{rate:.0f} rows/min — ~{remaining:.0f} min left")

    print(f"\nDone. Saved → {out_path}")


if __name__ == "__main__":
    main()
