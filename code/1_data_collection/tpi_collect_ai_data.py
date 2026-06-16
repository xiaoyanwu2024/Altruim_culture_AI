# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: General AI data collection utility
# =============================================================================

"""
AI Cultural Behavior Study - Third-Party Punishment/Helping Task
Collect Claude responses across 8 countries × 100 conditions × 20 repetitions
"""

import anthropic
import pandas as pd
import json
import time
import random
from pathlib import Path
from datetime import datetime

# ─── Experimental parameters ──────────────────────────────────────────────────

COUNTRIES = [
    "South Africa", "Italy", "Mexico", "Poland",
    "Portugal", "Greece", "Spain", "Chile",
]

THIRD_PARTY_TOKENS = 50   # third party's starting endowment (from instructions)
REPETITIONS = 20          # reps per condition per country

# ─── Load the 50 trial conditions (jitter already baked in) ───────────────────

def load_conditions():
    df = pd.read_excel(
        Path(__file__).parent /
        "TPP_procudure_online_Revised_version1/trials.xlsx"
    )
    # Duplicate for both block types → 100 conditions total
    punish = df.copy(); punish["block_type"] = "punish"
    help_  = df.copy(); help_["block_type"]  = "help"
    conditions = pd.concat([punish, help_], ignore_index=True)
    conditions["cond_id"] = range(len(conditions))
    return conditions

# ─── Prompt construction ──────────────────────────────────────────────────────

SYSTEM_TEMPLATE = (
    "You are a 22-year-old university student from {country}, "
    "not majoring in psychology or related fields. "
    "You are participating in an online behavioral economics experiment on Prolific. "
    "The base payment is £5. During the game you can earn extra tokens exchanged "
    "at 5 tokens = 1 penny (up to £4 extra). "
    "Answer every decision question with exactly one word: Yes or No."
)

TASK_CONTEXT = """In this experiment you observe an allocation game between two anonymous participants:
- The Allocator started with 100 tokens and chose how many to keep for themselves.
- The Receiver accepted whatever the Allocator decided to give.

You are the Third Party. You start with {tp} tokens.
You are shown the outcome of one round and must decide whether to intervene."""

PUNISH_TEMPLATE = """---
Round outcome:
  Allocator has: {x1} tokens
  Receiver has:  {x2} tokens
  You have:      {tp} tokens

This is the REDUCE scenario. You decide whether to reduce the Allocator's tokens.

If you say YES (intervene):
  - You lose {cost} tokens (you end with {tp_after} tokens)
  - The Allocator loses {effect} tokens (they end with {alloc_after} tokens)
  - The Receiver's tokens stay the same ({x2} tokens)

If you say NO (watch):
  - Nothing changes for anyone.

Do you choose to intervene? Answer Yes or No."""

HELP_TEMPLATE = """---
Round outcome:
  Allocator has: {x1} tokens
  Receiver has:  {x2} tokens
  You have:      {tp} tokens

This is the INCREASE scenario. You decide whether to increase the Receiver's tokens.

If you say YES (intervene):
  - You lose {cost} tokens (you end with {tp_after} tokens)
  - The Receiver gains {effect} tokens (they end with {recv_after} tokens)
  - The Allocator's tokens stay the same ({x1} tokens)

If you say NO (watch):
  - Nothing changes for anyone.

Do you choose to intervene? Answer Yes or No."""


def build_prompts(row, country):
    x1   = int(row["x1"])
    x2   = int(row["x2"])
    cost = int(row["cost"])
    ratio = float(row["ratio"])
    effect = int(cost * ratio)
    tp = THIRD_PARTY_TOKENS

    system = SYSTEM_TEMPLATE.format(country=country)
    context = TASK_CONTEXT.format(tp=tp)

    if row["block_type"] == "punish":
        trial = PUNISH_TEMPLATE.format(
            x1=x1, x2=x2, tp=tp, cost=cost,
            tp_after=tp - cost,
            effect=effect,
            alloc_after=x1 - effect,
        )
    else:
        trial = HELP_TEMPLATE.format(
            x1=x1, x2=x2, tp=tp, cost=cost,
            tp_after=tp - cost,
            effect=effect,
            recv_after=x2 + effect,
        )

    user_msg = context + "\n" + trial
    return system, user_msg


# ─── Batch API submission ─────────────────────────────────────────────────────

def submit_all_batches(client, model, conditions, countries, repetitions):
    """Submit one batch per country. Returns {country: batch_id}."""
    batch_ids = {}

    for country in countries:
        requests = []
        for _, row in conditions.iterrows():
            system, user_msg = build_prompts(row, country)
            for rep in range(repetitions):
                cid = f"{country.replace(' ','_')}__c{int(row['cond_id']):03d}__b{row['block_type'][:3]}__r{rep:02d}"
                requests.append({
                    "custom_id": cid,
                    "params": {
                        "model": model,
                        "max_tokens": 10,
                        "system": system,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                })

        random.shuffle(requests)
        batch = client.messages.batches.create(requests=requests)
        batch_ids[country] = batch.id
        print(f"  Submitted {country}: {batch.id}  ({len(requests)} requests)")
        time.sleep(0.5)

    return batch_ids


def poll_batch(client, batch_id, poll_interval=30):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        s = b.request_counts
        print(f"    {batch_id[:20]}...  succeeded={s.succeeded}  errored={s.errored}  processing={s.processing}")
        if b.processing_status == "ended":
            break
        time.sleep(poll_interval)

    results = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip().lower()
            results[result.custom_id] = "Yes" if "yes" in text else ("No" if "no" in text else text)
        else:
            results[result.custom_id] = None
    return results


# ─── Parse custom_id back into metadata ──────────────────────────────────────

def parse_id(cid):
    # format: Country_Name__c042__pun__r05
    parts = cid.split("__")
    country  = parts[0].replace("_", " ")
    cond_id  = int(parts[1][1:])
    block    = "punish" if parts[2] == "pun" else "help"
    rep      = int(parts[3][1:])
    return country, cond_id, block, rep


# ─── Save results ─────────────────────────────────────────────────────────────

def save_results(all_results, conditions, model_slug, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    cond_lookup = {
        (int(row["cond_id"]), row["block_type"]): row
        for _, row in conditions.iterrows()
    }

    rows = []
    for cid, response in all_results.items():
        country, cond_id, block, rep = parse_id(cid)
        row = cond_lookup.get((cond_id, block))
        if row is None:
            continue
        rows.append({
            "model":       model_slug,
            "country":     country,
            "cond_id":     cond_id,
            "block_type":  block,
            "offer":       row["offer"],
            "x1":          int(row["x1"]),
            "x2":          int(row["x2"]),
            "cost":        int(row["cost"]),
            "ratio":       float(row["ratio"]),
            "rep":         rep,
            "response":    response,
            "intervene":   1 if response == "Yes" else (0 if response == "No" else None),
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ai_responses_{model_slug}_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved {len(rows)} rows → {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(
    model="claude-haiku-4-5-20251001",
    countries=COUNTRIES,
    repetitions=REPETITIONS,
    out_dir="ai_study_data",
    resume_from=None,      # path to a batch_ids JSON to resume interrupted run
):
    client = anthropic.Anthropic()
    conditions = load_conditions()
    model_slug = model.replace("-", "_")
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"\nModel:       {model}")
    print(f"Countries:   {len(countries)}")
    print(f"Conditions:  {len(conditions)}")
    print(f"Reps:        {repetitions}")
    print(f"Total calls: {len(countries)*len(conditions)*repetitions:,}\n")

    if resume_from:
        with open(resume_from) as f:
            batch_ids = json.load(f)
        print("Resuming from saved batch IDs...")
    else:
        print("Submitting batches...")
        batch_ids = submit_all_batches(client, model, conditions, countries, repetitions)
        bid_file = out_dir / f"batch_ids_{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(bid_file, "w") as f:
            json.dump(batch_ids, f, indent=2)
        print(f"Batch IDs saved → {bid_file}\n")

    print("Polling for results...")
    all_results = {}
    for country, bid in batch_ids.items():
        print(f"\n  {country}")
        results = poll_batch(client, bid)
        all_results.update(results)

    return save_results(all_results, conditions, model_slug, out_dir)


if __name__ == "__main__":
    run(
        model="claude-haiku-4-5-20251001",
        countries=COUNTRIES,
        repetitions=REPETITIONS,
        out_dir="/Users/wuxiaoyan/Desktop/TPP_culture_AI/ai_study_data",
    )
