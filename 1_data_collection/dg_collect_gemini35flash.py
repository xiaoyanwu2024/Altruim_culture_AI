# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: DG data collection — Gemini-3.5-Flash
# =============================================================================

"""
Gemini 3.5 Flash Task 1 — Dictator Game (Allocator role).
8 countries × 2 genders = 16 responses.
"""

import os, csv, re, time
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types

API_KEY  = os.environ.get("GEMINI_API_KEY", "")
MODEL_ID = "gemini-3.5-flash"

COUNTRIES = ["South Africa","Italy","Mexico","Poland","Portugal","Greece","Spain","Chile"]
GENDERS   = ["male","female"]

DATA_DIR = Path("ai_study_data")

SAFETY_OFF = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]


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
        print("ERROR: set GEMINI_API_KEY"); return

    client = genai.Client(api_key=API_KEY)
    out_path = DATA_DIR / f"gemini35flash_task1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields   = ["model","country","gender","response","allocation"]

    print(f"Model: {MODEL_ID}")
    print(f"Total: {len(COUNTRIES)*len(GENDERS)} requests\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for country in COUNTRIES:
            for gender in GENDERS:
                sp = system_prompt(country, gender)
                full_prompt = sp + "\n\n" + TASK1_PROMPT
                allocation = None
                for attempt in range(8):
                    try:
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=full_prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=20,
                                temperature=1.0,
                                safety_settings=SAFETY_OFF,
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                            )
                        )
                        raw = response.text.strip() if response.text else None
                        if raw:
                            allocation = parse_allocation(raw)
                            if allocation is not None:
                                print(f"  {country} / {gender}: '{raw}' → {allocation}")
                                writer.writerow({
                                    "model": "gemini_3.5_flash",
                                    "country": country,
                                    "gender": gender,
                                    "response": raw,
                                    "allocation": allocation,
                                })
                                f.flush()
                                break
                    except Exception as e:
                        err = str(e)
                        wait = max(15, 2 ** attempt) if ("429" in err or "quota" in err.lower()) else min(2 ** attempt, 30)
                        print(f"  [{country}/{gender}] attempt {attempt+1}: {err[:80]} — wait {wait}s")
                        time.sleep(wait)
                if allocation is None:
                    print(f"  ⚠ {country}/{gender}: failed after 8 attempts")
                time.sleep(0.5)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
