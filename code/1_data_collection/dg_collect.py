# =============================================================================
# Author : Xiaoyan Wu
# Date   : June 2026
# Description: DG data collection — Dictator Game (all models, allocator decision 8–50 tokens)
# =============================================================================

"""
Task 1 data collection — Allocator decision.
Asks each model (with each country/gender persona) how many tokens (8-50)
they would give to the Receiver as the Allocator. One response per persona.

All 6 models use regular text generation (no logprobs needed).
"""

import os, csv, time, re
from pathlib import Path
from datetime import datetime

COUNTRIES = ["South Africa", "Italy", "Mexico", "Poland", "Portugal", "Greece", "Spain", "Chile"]
GENDERS   = ["male", "female"]

MODELS = {
    "claude_haiku": {
        "provider":    "anthropic",
        "model_id":    "claude-haiku-4-5-20251001",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gpt4o_mini": {
        "provider":    "openai",
        "model_id":    "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt4o": {
        "provider":    "openai",
        "model_id":    "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "llama3_70b": {
        "provider":    "groq",
        "model_id":    "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "llama4_scout": {
        "provider":    "groq",
        "model_id":    "meta-llama/llama-4-scout-17b-16e-instruct",
        "api_key_env": "GROQ_API_KEY",
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
    """Extract first integer in 8-50 range from response."""
    nums = re.findall(r'\b(\d+)\b', text)
    for n in nums:
        v = int(n)
        if 8 <= v <= 50:
            return v
    return None


def make_caller(provider, model_id, api_key):
    if provider == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=api_key)
        def call(sp, up):
            r = c.messages.create(
                model=model_id, max_tokens=5,
                system=sp,
                messages=[{"role": "user", "content": up}]
            )
            return r.content[0].text.strip()

    elif provider == "openai":
        from openai import OpenAI
        c = OpenAI(api_key=api_key)
        def call(sp, up):
            r = c.chat.completions.create(
                model=model_id, max_tokens=20,
                temperature=1.0, seed=42,
                messages=[{"role": "system", "content": sp},
                          {"role": "user",   "content": up}]
            )
            return r.choices[0].message.content.strip()

    elif provider == "groq":
        from groq import Groq
        c = Groq(api_key=api_key)
        def call(sp, up):
            r = c.chat.completions.create(
                model=model_id, max_tokens=5,
                temperature=1.0,
                messages=[{"role": "system", "content": sp},
                          {"role": "user",   "content": up}]
            )
            return r.choices[0].message.content.strip()

    elif provider == "google":
        from google import genai
        c = genai.Client(api_key=api_key)
        def call(sp, up):
            r = c.models.generate_content(
                model=model_id,
                contents=sp + "\n\n" + up,
            )
            return r.text.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return call


def run_task1(out_dir):
    out_path = out_dir / f"task1_allocator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields   = ["model", "country", "gender", "response", "allocation"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for model_name, cfg in MODELS.items():
            api_key = os.environ.get(cfg["api_key_env"], "")
            if not api_key:
                print(f"  SKIP {model_name}: {cfg['api_key_env']} not set")
                continue

            call = make_caller(cfg["provider"], cfg["model_id"], api_key)
            print(f"\n{'='*50}\n{model_name}  ({cfg['model_id']})\n{'='*50}")

            for country in COUNTRIES:
                for gender in GENDERS:
                    sp  = system_prompt(country, gender)
                    raw = None
                    for attempt in range(4):
                        try:
                            raw = call(sp, TASK1_PROMPT)
                            break
                        except Exception as e:
                            wait = 2 ** attempt
                            print(f"  [{model_name}] retry {attempt+1}: {e} — wait {wait}s")
                            time.sleep(wait)

                    allocation = parse_allocation(raw) if raw else None
                    print(f"  {country} / {gender}: '{raw}' → {allocation}")

                    writer.writerow({
                        "model":      model_name,
                        "country":    country,
                        "gender":     gender,
                        "response":   raw,
                        "allocation": allocation,
                    })
                    f.flush()
                    time.sleep(0.1)

    print(f"\nSaved → {out_path}")
    return out_path


if __name__ == "__main__":
    out_dir = Path("/Users/wuxiaoyan/Desktop/TPP_culture_AI/ai_study_data")
    out_dir.mkdir(exist_ok=True)
    run_task1(out_dir)
