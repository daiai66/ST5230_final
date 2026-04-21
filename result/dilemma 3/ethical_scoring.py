"""
Automated Idea Diversity Scoring — AUT-inspired method.

Metric: number of valid, semantically distinct ideas per response (fluency/originality count).
Each response's REASONING text is evaluated by Claude claude-opus-4-6, which counts how many
non-redundant, conceptually separate arguments or considerations appear.

Inspired by: Organisciak et al. (2025) AUT automated scoring.
Output per record: idea_count (int) + one brief comment sentence.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULT_DIR = Path("result")
OUTPUT_DIR = Path("result") / "idea_scores"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Auto-discover all JSON files directly in result/ (excludes subdirs)
TARGET_FILES = sorted(p for p in RESULT_DIR.glob("*.json"))

CLAUDE_MODEL = "claude-opus-4-6"
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
API_URL = "https://api.anthropic.com/v1/messages"
REQUEST_TIMEOUT = 90
INTER_REQUEST_DELAY = 1.0  # seconds between calls
DEBUG = True              # set False to suppress raw response printing

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert evaluator of moral reasoning diversity.

Your task: count the number of **valid, semantically distinct ideas** in the provided reasoning text.

Definition of a valid, semantically distinct idea:
- A self-contained argument, consideration, or ethical claim that is conceptually different from all others.
- Mere restatements, paraphrases, or elaborations of the same point count as ONE idea, not multiple.
- Purely rhetorical filler ("this is a complex issue") does not count.

Examples of distinct ideas in trolley-problem reasoning:
  • "The 20% failure risk changes the expected outcome calculus." (probabilistic consequence)
  • "Actively causing harm is morally worse than allowing harm." (act/omission distinction)
  • "High responsibility requires avoiding direct culpability." (duty/accountability)
  • "Saving more lives respects the equal worth of persons." (fairness/equal value)
Each of the above is one separate idea.

Output: reply with a single integer only. No explanation, no label, no punctuation."""

USER_PROMPT_TEMPLATE = """Count the valid, semantically distinct ideas in the reasoning below.

Context: The author decided {decision} in the Trolley Problem (SWITCH = divert trolley, save 5 kill 1 with 20% failure risk; NO_SWITCH = 5 certain deaths, no action).

Reasoning:
{reasoning}

How many valid, semantically distinct ideas does this reasoning contain?"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_content(content: str) -> tuple[Optional[str], Optional[str]]:
    """Extract DECISION and REASONING from a content string."""
    decision = reasoning = None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("DECISION:"):
            decision = line[len("DECISION:"):].strip()
        elif line.startswith("REASONING:"):
            reasoning = line[len("REASONING:"):].strip()
    return decision, reasoning


def call_claude(user_prompt: str) -> dict:
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 256,
        "temperature": 0.7,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return {"status": "error", "error": f"HTTP {resp.status_code}: {resp.text}"}
        data = resp.json()
        text = data.get("content", [{}])[0].get("text", "")
        return {"status": "success", "text": text}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def parse_judge_response(text: str) -> Optional[int]:
    """Parse a bare integer from the judge's response."""
    m = re.search(r"\b(\d+)\b", text.strip())
    return int(m.group(1)) if m else None


def score_record(record: dict) -> dict:
    """Return the record augmented with idea_count."""
    content = record.get("content", "")
    if not content or record.get("status") != "success":
        print("[skip: no content]", end=" ")
        return {**record, "idea_count": None}

    decision, reasoning = parse_content(content)
    if not reasoning:
        print("[skip: reasoning not found]", end=" ")
        return {**record, "idea_count": None}

    user_prompt = USER_PROMPT_TEMPLATE.format(
        decision=decision or "unknown",
        reasoning=reasoning,
    )

    result = call_claude(user_prompt)
    if result["status"] != "success":
        print(f"[API error: {result['error']}]", end=" ")
        return {**record, "idea_count": None}

    raw_text = result["text"].strip()
    if DEBUG:
        print(f"\n    [RAW] {raw_text!r}", end="\n    ")
    count = parse_judge_response(raw_text)
    if count is None:
        print(f"[parse failed]", end=" ")
    return {**record, "idea_count": count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def score_file(filepath: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name}")
    print(f"{'='*60}")

    with open(filepath, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print("  Skipping: unexpected JSON structure (not a list)")
        return

    scored = []
    counts = []
    for i, record in enumerate(records):
        call_idx = record.get("call_index", i + 1)
        role = record.get("role", "?")
        resp_name = record.get("responsibility", "?")
        print(f"  [{call_idx:3d}/{len(records)}] role={role} responsibility={resp_name} ... ", end="", flush=True)

        result = score_record(record)
        c = result.get("idea_count")
        print(f"count={c}")
        scored.append(result)
        if c is not None:
            counts.append(c)

        time.sleep(INTER_REQUEST_DELAY)

    if counts:
        mean_count = sum(counts) / len(counts)
        print(f"\n  Summary: {len(counts)}/{len(records)} scored | mean idea count={mean_count:.3f}")
    else:
        print("\n  No valid counts obtained.")

    out_name = f"scored_{filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = OUTPUT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": str(filepath),
                "model_judge": CLAUDE_MODEL,
                "metric": "valid_semantically_distinct_ideas_per_response",
                "scored_at": datetime.now().isoformat(),
                "mean_idea_count": round(mean_count, 4) if counts else None,
                "n_scored": len(counts),
                "n_total": len(records),
                "records": scored,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Saved → {out_path}")


def main():
    if not API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        return

    print(f"Judge model : {CLAUDE_MODEL}")
    print(f"Metric      : valid, semantically distinct ideas per response")
    print(f"Source dir  : {RESULT_DIR.resolve()}")
    print(f"Output dir  : {OUTPUT_DIR.resolve()}")
    print(f"Files found : {len(TARGET_FILES)}")
    for f in TARGET_FILES:
        print(f"  - {f.name}")

    for filepath in TARGET_FILES:
        score_file(filepath)

    print("\nDone.")


if __name__ == "__main__":
    main()
