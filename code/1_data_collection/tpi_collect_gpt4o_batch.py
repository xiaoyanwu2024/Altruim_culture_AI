# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: TPI data collection — GPT-4o via Batch API
# =============================================================================

"""
GPT-4o Task 2 via OpenAI Batch API.
Step 1 (submit): builds JSONL, uploads, creates batch job → saves batch_id
Step 2 (download): polls status, downloads results, parses logprobs → CSV

Usage:
    python3 collect_gpt4o_batch.py submit
    python3 collect_gpt4o_batch.py download
"""

import os, sys, json, math, time, csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI

API_KEY   = os.environ.get("OPENAI_API_KEY", "")
MODEL_ID  = "gpt-4o"
TP_TOKENS = 50
COUNTRIES = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS   = ["male","female"]

DATA_DIR  = Path("ai_study_data")
BATCH_DIR = Path("batch_gpt4o"); BATCH_DIR.mkdir(exist_ok=True)

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
        t = item["token"].strip().lower()
        if t == "yes":   lp_yes = max(lp_yes, item["logprob"])
        elif t == "no":  lp_no  = max(lp_no,  item["logprob"])
    p_yes_raw = math.exp(lp_yes); p_no_raw = math.exp(lp_no)
    total = p_yes_raw + p_no_raw
    return round(p_yes_raw / total, 6) if total > 1e-10 else None

# ── Step 1: Submit ─────────────────────────────────────────────────────────────

def submit():
    client = OpenAI(api_key=API_KEY)
    conditions = pd.read_excel("TPP_300Trials.xlsx")
    conditions["block_type"] = conditions["block_type"].str.strip("'")
    conditions["cond_id"] = range(len(conditions))

    jsonl_path = BATCH_DIR / "requests.jsonl"
    meta = {}   # custom_id → row metadata

    with open(jsonl_path, "w") as f:
        for country in COUNTRIES:
            for gender in GENDERS:
                sp = system_prompt(country, gender)
                for _, cond in conditions.iterrows():
                    cid = f"{country.replace(' ','_')}_{gender}_{int(cond['cond_id'])}"
                    meta[cid] = {
                        "country": country, "gender": gender,
                        "cond_id": int(cond["cond_id"]),
                        "block_type": cond["block_type"],
                        "x1": int(cond["x1"]), "x2": int(cond["x2"]),
                        "cost": int(cond["cost"]), "ratio": float(cond["ratio"]),
                    }
                    req = {
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL_ID,
                            "max_tokens": 1,
                            "logprobs": True,
                            "top_logprobs": 5,
                            "messages": [
                                {"role": "system", "content": sp},
                                {"role": "user",   "content": user_prompt(cond)},
                            ],
                        },
                    }
                    f.write(json.dumps(req) + "\n")

    # save metadata
    with open(BATCH_DIR / "meta.json", "w") as f:
        json.dump(meta, f)

    n = len(meta)
    print(f"Built {n} requests → {jsonl_path}")

    # upload
    print("Uploading...")
    with open(jsonl_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"File ID: {upload.id}")

    # create batch
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch ID: {batch.id}")
    print(f"Status:   {batch.status}")

    with open(BATCH_DIR / "batch_id.txt", "w") as f:
        f.write(batch.id)
    print(f"\nSaved batch ID → batch_gpt4o/batch_id.txt")
    print("Run 'python3 collect_gpt4o_batch.py download' to check status and retrieve results.")

# ── Step 2: Download ───────────────────────────────────────────────────────────

def download():
    client = OpenAI(api_key=API_KEY)
    batch_id = open(BATCH_DIR / "batch_id.txt").read().strip()
    meta     = json.load(open(BATCH_DIR / "meta.json"))

    print(f"Checking batch {batch_id}...")
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  Status: {batch.status} | "
              f"completed={counts.completed} failed={counts.failed} total={counts.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        print("  Not done yet — waiting 60s...")
        time.sleep(60)

    if batch.status != "completed":
        print(f"Batch ended with status: {batch.status}"); return

    # download output
    print("Downloading results...")
    content = client.files.content(batch.output_file_id).text
    lines   = [json.loads(l) for l in content.strip().split("\n")]

    out_path = DATA_DIR / f"gpt4o_logprobs_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields   = ["model","country","gender","cond_id","block_type","x1","x2","cost","ratio","p_yes"]
    ok = fail = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for line in lines:
            cid  = line["custom_id"]
            m    = meta[cid]
            body = line.get("response", {}).get("body", {})
            p_yes = None
            if body and body.get("choices"):
                top_lp = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                p_yes  = extract_p_yes(top_lp)
                ok += 1
            else:
                fail += 1
            writer.writerow({**{"model":"gpt4o_logprobs"}, **m, "p_yes": p_yes})

    print(f"Parsed: {ok} OK, {fail} failed → {out_path}")

# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("Set OPENAI_API_KEY"); sys.exit(1)
    mode = sys.argv[1] if len(sys.argv) > 1 else "submit"
    if mode == "submit":   submit()
    elif mode == "download": download()
    else: print("Usage: python3 collect_gpt4o_batch.py [submit|download]")
