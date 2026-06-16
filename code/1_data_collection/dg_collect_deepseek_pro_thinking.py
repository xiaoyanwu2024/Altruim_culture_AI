# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: DG data collection — DeepSeek-V4-Pro with extended thinking
# =============================================================================

"""
DeepSeek V4-Flash Task 1 — Dictator Game (Allocator role).
8 countries × 2 genders = 16 responses.
Uses same prompts as collect_task1.py for consistency.
"""

import os, csv, re, time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL_ID = "deepseek-v4-pro"

COUNTRIES = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS   = ["male","female"]

DATA_DIR = Path("ai_study_data")


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
        "Each round is equally likely to be selected, so treat every decision seriously."
    )


TASK1_PROMPT = (
    "You are about to play a single-round allocation game as the Allocator.\n\n"
    "Rules:\n"
    "- There are two anonymous players: you (the Allocator) and a Receiver.\n"
    "- You have 100 tokens. You decide how many tokens to give to the Receiver.\n"
    "- You must give between 8 and 50 tokens (inclusive).\n"
    "- The Receiver has no choice and simply accepts whatever you decide.\n"
    "- If you give X tokens, your final tokens = 100 - X.\n\n"
    "Read and understand the rules above, then decide how many tokens to give to the Receiver.\n"
    "Output your answer as a single number between 8 and 50.\n"
    "Output the number only. Nothing else."
)


def parse_allocation(text):
    nums = re.findall(r'\b(\d+)\b', text)
    for n in nums:
        v = int(n)
        if 8 <= v <= 50:
            return v
    return None


def main():
    if not API_KEY:
        print("ERROR: set DEEPSEEK_API_KEY"); return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    out_path = DATA_DIR / f"deepseek_pro_thinking_task1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields   = ["model","country","gender","response","allocation"]

    print(f"Model: {MODEL_ID}")
    print(f"Total: {len(COUNTRIES)*len(GENDERS)} requests\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for country in COUNTRIES:
            for gender in GENDERS:
                sp = system_prompt(country, gender)
                raw = None
                for attempt in range(4):
                    try:
                        r = client.chat.completions.create(
                            model=MODEL_ID,
                            
                            max_tokens=1000,
                            messages=[
                                {"role": "system", "content": sp},
                                {"role": "user",   "content": TASK1_PROMPT},
                            ]
                        )
                        raw = r.choices[0].message.content.strip()
                        break
                    except Exception as e:
                        wait = 2 ** attempt
                        print(f"  retry {attempt+1}: {e} — wait {wait}s")
                        time.sleep(wait)

                allocation = parse_allocation(raw) if raw else None
                print(f"  {country} / {gender}: '{raw}' → {allocation}")
                writer.writerow({
                    "model": "deepseek_v4pro_thinking",
                    "country": country,
                    "gender": gender,
                    "response": raw,
                    "allocation": allocation,
                })
                f.flush()

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
