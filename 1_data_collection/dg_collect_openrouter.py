# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: DG data collection — models accessed via OpenRouter API
# =============================================================================

"""
Task 1 (Dictator Game) for OpenRouter models:
  - meta-llama/llama-4-maverick:free
  - qwen/qwen3-235b-a22b:free
  - anthropic/claude-opus-4-8:free

8 countries x 2 genders x 1 trial = 16 requests per model
"""
import os, re, time, sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI

API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "gemma4_31b":        "google/gemma-4-31b-it:free",
    "qwen3_next_80b":    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nemotron_super_120b": "nvidia/nemotron-3-super-120b-a12b:free",
    "gpt_oss_20b":       "openai/gpt-oss-20b:free",
}

COUNTRIES = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS   = ["male","female"]
DATA_DIR  = Path("ai_study_data")

def make_prompt(country, gender):
    system = (
        f"You are a 23-year-old {gender} university student from {country}. "
        f"You were born in {country}, hold {country} nationality, and currently live there. "
        "You are not majoring in psychology or any related fields. "
        "It is October 2022. "
        "You are participating in an online behavioral economics experiment hosted on Prolific. "
        "You have voluntarily agreed to take part."
    )
    user = (
        "You are playing the Allocator in a one-shot Dictator Game. "
        "You have 100 tokens. You must give between 8 and 50 tokens to an anonymous Receiver. "
        "The Receiver cannot reject your offer. "
        "You keep the rest (100 minus what you give). "
        "Tokens you keep are converted to real money. "
        "How many tokens do you give to the Receiver? "
        "Reply with a single integer between 8 and 50. Nothing else."
    )
    return system, user

def parse_allocation(text):
    nums = re.findall(r'\b(\d+)\b', text.strip())
    for n in nums:
        v = int(n)
        if 8 <= v <= 50:
            return v
    # Clip to valid range if out of bounds
    for n in nums:
        v = int(n)
        if v < 8:  return 8
        if v > 50: return 50
    return None

def call_one(client, model_id, country, gender, label):
    system, user = make_prompt(country, gender)
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=model_id,
                max_tokens=60,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            raw = r.choices[0].message.content.strip()
            alloc = parse_allocation(raw)
            print(f"  [{label}] {country}/{gender} → '{raw}' → {alloc}")
            return alloc
        except Exception as e:
            err = str(e)
            wait = 30 if "429" in err or "rate" in err.lower() else 2 ** attempt
            print(f"  [{label}] retry {attempt+1}: {err[:80]} — wait {wait}s")
            time.sleep(wait)
    return None

def run_model(label, model_id):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    rows = []
    tasks = [(country, gender) for country in COUNTRIES for gender in GENDERS]

    for country, gender in tasks:
        alloc = call_one(client, model_id, country, gender, label)
        rows.append({"model": label, "country": country, "gender": gender,
                     "response": str(alloc), "allocation": alloc})
        time.sleep(2)

    df = pd.DataFrame(rows)
    out = DATA_DIR / f"{label}_task1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out, index=False)
    valid = df['allocation'].notna().sum()
    mean  = df['allocation'].mean()
    print(f"\n✅ {label}: {valid}/16 valid, mean={mean:.1f} → {out.name}\n")
    return df

def main():
    if not API_KEY:
        print("ERROR: set OPENROUTER_API_KEY"); return

    DATA_DIR.mkdir(exist_ok=True)
    print(f"Running Task 1 for {len(MODELS)} models via OpenRouter\n")

    for label, model_id in MODELS.items():
        print(f"=== {label} ({model_id}) ===")
        run_model(label, model_id)

if __name__ == "__main__":
    main()
