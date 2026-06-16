# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: TPI data collection — Claude via Batch API
# =============================================================================

"""
Claude Haiku Task 2 via Anthropic Message Batches API.
Submits all 24000 requests in 3 batches (max 10000 each), parallel processing.

Usage:
    python3 collect_claude_batch.py submit
    python3 collect_claude_batch.py download
"""

import os, sys, json, time, csv
import pandas as pd
import anthropic
from pathlib import Path
from datetime import datetime

API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_ID   = "claude-haiku-4-5-20251001"
REPS       = 5
TP_TOKENS  = 50
BATCH_SIZE = 8000   # under 10000 limit
COUNTRIES  = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS    = ["male","female"]

DATA_DIR  = Path("ai_study_data")
BATCH_DIR = Path("batch_claude"); BATCH_DIR.mkdir(exist_ok=True)

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

# ── Submit ─────────────────────────────────────────────────────────────────────

def submit():
    client = anthropic.Anthropic(api_key=API_KEY)
    conditions = pd.read_excel("TPP_300Trials.xlsx")
    conditions["block_type"] = conditions["block_type"].str.strip("'")
    conditions["cond_id"] = range(len(conditions))

    # Build all requests
    all_requests = []
    meta = {}
    for country in COUNTRIES:
        for gender in GENDERS:
            sp = system_prompt(country, gender)
            for _, cond in conditions.iterrows():
                for rep in range(REPS):
                    cid = f"{country.replace(' ','_')}_{gender}_{int(cond['cond_id'])}_{rep}"
                    meta[cid] = {
                        "country": country, "gender": gender,
                        "cond_id": int(cond["cond_id"]),
                        "block_type": cond["block_type"],
                        "x1": int(cond["x1"]), "x2": int(cond["x2"]),
                        "cost": int(cond["cost"]), "ratio": float(cond["ratio"]),
                        "rep": rep,
                    }
                    all_requests.append({
                        "custom_id": cid,
                        "params": {
                            "model": MODEL_ID,
                            "max_tokens": 10,
                            "system": sp,
                            "messages": [{"role": "user", "content": user_prompt(cond)}],
                        }
                    })

    with open(BATCH_DIR / "meta.json", "w") as f:
        json.dump(meta, f)

    print(f"Total requests: {len(all_requests)}")

    # Split into batches of BATCH_SIZE
    chunks = [all_requests[i:i+BATCH_SIZE] for i in range(0, len(all_requests), BATCH_SIZE)]
    batch_ids = []

    for i, chunk in enumerate(chunks):
        print(f"Submitting batch {i+1}/{len(chunks)} ({len(chunk)} requests)...")
        batch = client.messages.batches.create(requests=chunk)
        print(f"  Batch ID: {batch.id} | Status: {batch.processing_status}")
        batch_ids.append(batch.id)

    with open(BATCH_DIR / "batch_ids.json", "w") as f:
        json.dump(batch_ids, f)

    print(f"\nAll {len(chunks)} batches submitted. IDs saved to batch_claude/batch_ids.json")
    print("Run 'python3 collect_claude_batch.py download' to poll and retrieve results.")

# ── Download ───────────────────────────────────────────────────────────────────

def download():
    client = anthropic.Anthropic(api_key=API_KEY)
    batch_ids = json.load(open(BATCH_DIR / "batch_ids.json"))
    meta      = json.load(open(BATCH_DIR / "meta.json"))

    all_results = {}

    for batch_id in batch_ids:
        print(f"\nPolling {batch_id}...")
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            c = batch.request_counts
            print(f"  {batch.processing_status} | processing={c.processing} ended={c.succeeded+c.errored}")
            if batch.processing_status == "ended":
                break
            print("  Waiting 60s...")
            time.sleep(60)

        for result in client.messages.batches.results(batch_id):
            cid = result.custom_id
            if result.result.type == "succeeded":
                text = result.result.message.content[0].text.strip().lower()
                if text.startswith("yes") or ("yes" in text and "no" not in text):
                    intervene = 1
                elif text.startswith("no") or "no" in text:
                    intervene = 0
                else:
                    intervene = None
                all_results[cid] = {"response": result.result.message.content[0].text.strip(),
                                    "intervene": intervene}
            else:
                all_results[cid] = {"response": None, "intervene": None}

    out_path = DATA_DIR / f"claude_haiku_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields   = ["model","country","gender","cond_id","block_type","x1","x2","cost","ratio","rep","response","intervene"]
    ok = fail = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cid, m in meta.items():
            res = all_results.get(cid, {"response": None, "intervene": None})
            writer.writerow({**{"model":"claude_haiku"}, **m, **res})
            if res["intervene"] is not None: ok += 1
            else: fail += 1

    print(f"\nParsed: {ok} OK, {fail} failed → {out_path}")

if __name__ == "__main__":
    if not API_KEY:
        print("Set ANTHROPIC_API_KEY"); sys.exit(1)
    mode = sys.argv[1] if len(sys.argv) > 1 else "submit"
    if mode == "submit":     submit()
    elif mode == "download": download()
    else: print("Usage: python3 collect_claude_batch.py [submit|download]")
