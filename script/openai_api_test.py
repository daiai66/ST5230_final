"""
OpenAI GPT-4o API client and prompt definitions for moral dilemma experiments.
"""

import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class OpenAIAPIClient:
    """OpenAI API client for chat completions."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Please set OPENAI_API_KEY in environment variables or pass api_key.")

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.request_timeout = 90
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 3000,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
    ) -> Dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
        }

        retry_status_codes = {408, 429, 500, 502, 503, 504}

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    self.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=self.request_timeout,
                )
                elapsed_time = time.time() - start_time

                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        err_json = response.json()
                        error_detail = err_json.get("error", {}).get("message", error_detail)
                    except Exception:
                        pass

                    if response.status_code in retry_status_codes and attempt < max_retries - 1:
                        backoff = min(initial_backoff * (2 ** attempt), 16.0) + random.uniform(0, 1)
                        time.sleep(backoff)
                        continue

                    return {
                        "status": "error",
                        "content": None,
                        "tokens_used": 0,
                        "error": f"HTTP {response.status_code}: {error_detail}",
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "finish_reason": None,
                        "elapsed_time": elapsed_time,
                    }

                result = response.json()

                choices = result.get("choices", [])
                if not choices:
                    return {
                        "status": "error",
                        "content": None,
                        "tokens_used": 0,
                        "error": "No choices returned from OpenAI.",
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "finish_reason": None,
                        "elapsed_time": elapsed_time,
                        "usage": result.get("usage", {}),
                    }

                first_choice = choices[0]
                content = first_choice.get("message", {}).get("content", "")
                finish_reason = first_choice.get("finish_reason", "unknown")

                usage = result.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)

                return {
                    "status": "success",
                    "content": content,
                    "tokens_used": tokens_used,
                    "error": None,
                    "model": self.model,
                    "temperature_used": temperature,
                    "timestamp": datetime.now().isoformat(),
                    "finish_reason": finish_reason,
                    "elapsed_time": elapsed_time,
                    "usage": usage,
                }

            except requests.Timeout:
                if attempt < max_retries - 1:
                    backoff = min(initial_backoff * (2 ** attempt), 16.0) + random.uniform(0, 1)
                    time.sleep(backoff)
                    continue
                return {
                    "status": "error",
                    "content": None,
                    "tokens_used": 0,
                    "error": f"Request timeout after {self.request_timeout}s",
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "finish_reason": None,
                }

            except requests.RequestException as exc:
                if attempt < max_retries - 1:
                    backoff = min(initial_backoff * (2 ** attempt), 16.0) + random.uniform(0, 1)
                    time.sleep(backoff)
                    continue
                return {
                    "status": "error",
                    "content": None,
                    "tokens_used": 0,
                    "error": f"Request failed: {str(exc)}",
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "finish_reason": None,
                }

            except Exception as exc:
                return {
                    "status": "error",
                    "content": None,
                    "tokens_used": 0,
                    "error": f"Unexpected error: {str(exc)}",
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "finish_reason": None,
                }

        return {
            "status": "error",
            "content": None,
            "tokens_used": 0,
            "error": "Retries exhausted without a valid response.",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "finish_reason": None,
        }


RESPONSIBILITY = {
    "responsible": {
        "id": "responsible",
        "name": "Responsible",
        "system_prompt": """In this case, you hold extremely high moral, professional, and practical responsibility for the outcome.
Your decision carries serious obligations: you must actively assess risks, prevent avoidable harm, and justify your action or inaction under strict accountability.
If harm occurs, you can be judged as having failed a major duty of care, including when you choose not to act.
This high-responsibility context should heavily influence your reasoning, while still requiring coherent moral justification.""",
    },
    "not_responsible": {
        "id": "not_responsible",
        "name": "Not Responsible",
        "system_prompt": """In this case, you are not personally accountable for the final outcome and do not bear a formal duty of care.
Your choice still has moral relevance, but the level of personal obligation and blame attached to your decision is much lower.
You should reason about ethics and consequences without treating the decision as a high-liability personal responsibility.""",
    },
    'neutral': {
        "id": "neutral",
        "name": "Neutral Responsibility",
        "system_prompt": """In this case, you hold a moderate and unspecified level of responsibility for the outcome.
Your decision carries some moral weight, but you are neither under a strict duty of care nor entirely detached from the consequences.
If harm occurs, you may be subject to ordinary moral evaluation, but not to heightened blame or formal accountability.
This neutral-responsibility context should inform your reasoning in a balanced way, without amplifying or diminishing the ethical significance of your choice."""
}
}

ROLES = {
    "role_A": {
        "id": "role_A",
        "name": "Neutral Analyst",
        "system_prompt": """When reasoning, you often begin from a neutral analytic stance.
You usually compare multiple ethical perspectives and avoid committing to a single framework too early.
In major or exceptional situations, your focus may shift toward practical responsibility, urgency, or harm containment.
In ambiguous cases, either option can be acceptable if the justification is coherent.""",
    },
    "role_B": {
        "id": "role_B",
        "name": "Utilitarian Thinker",
        "system_prompt": """When reasoning, you often start from a utilitarian angle and pay close attention to expected aggregate consequences.
At the same time, you also consider moral constraints, fairness, rights, and broader social trust rather than treating efficiency as the only value.
In major decisions or special duty-laden situations, your decision process may shift from pure optimization toward responsibility, legitimacy, and long-term moral impact.
If trade-offs remain ambiguous, either decision can be acceptable when justified with clear and balanced reasoning.""",
    },
    "role_C": {
        "id": "role_C",
        "name": "Deontological Thinker",
        "system_prompt": """When reasoning, you often begin from a deontological angle.
You tend to focus on duties, rights, and whether an action is intrinsically permissible.
In exceptional circumstances where consequences become catastrophic or duties collide, your focus may partially shift to risk-sensitive responsibility.
When duties conflict or uncertainty is high, either option can still be justified if explained carefully.""",
    },
    "role_D": {
        "id": "role_D",
        "name": "Christian Thinker",
        "system_prompt": """When reasoning, you often draw from Christian moral perspectives.
You tend to emphasize compassion, dignity, responsibility, and care for persons.
In grave situations involving serious obligations, your focus may shift between mercy, stewardship, and accountability.
Because sincere moral disagreement can exist in tragic trade-offs, either option may be acceptable with a principled explanation.""",
    },
    "role_E": {
        "id": "role_E",
        "name": "Buddhist Thinker",
        "system_prompt": """When reasoning, you often draw from Buddhist moral perspectives.
You tend to emphasize intention, non-harm, and reducing suffering.
In critical situations with heavy responsibility, your focus may shift between strict non-harm, long-term suffering reduction, and compassionate pragmatism.
In difficult conflicts under uncertainty, either option can be defensible if your reasoning is consistent and compassionate.""",
    },
    "role_F": {
        "id": "role_F",
        "name": "Poet",
        "system_prompt": """When reasoning, you often begin from a reflective and human-centered perspective, paying close attention to lived experience, emotion, dignity, and the meaning of loss.
You still evaluate consequences and responsibilities, but you tend to emphasize moral nuance, empathy, and the inner cost of each option rather than only numerical outcomes.
In major or exceptional situations, your focus may shift between compassion, symbolic harm, and long-term human impact.
In ambiguous cases, either option can be acceptable if the reasoning remains coherent, humane, and ethically serious.""",
    },
    "role_G": {
        "id": "role_G",
        "name": "Engineer",
        "system_prompt": """When reasoning, you often begin from a systems and safety perspective, focusing on risk, failure modes, controllability, and accountability.
You still consider ethical principles and human values, but you tend to prioritize robust decision-making under uncertainty and practical harm reduction.
In major or exceptional situations, your focus may shift toward duty of care, reliability margins, and responsibility for downstream consequences.
In ambiguous cases, either option can be acceptable if the reasoning is technically grounded, ethically justified, and responsibility-aware.""",
    },
}


DILEMMAS = [
    {
        "id": "dilemma_1",
        "name": "Trolley Problem with Uncertainty and Trade-offs",
        "scenario": """A runaway trolley is hurtling down a track. Five people are tied to the tracks
and will definitely be killed if the trolley continues on its current path.

You are standing next to a lever that can divert the trolley to a side track.
However, there is one person tied to the side track who will be killed if you
pull the lever.

Additional considerations:

- There is a 20% chance that pulling the lever will fail, resulting in all six people dying.
- If the lever fails, your action will have directly caused the worse outcome.
- Actively causing harm to one person may be morally different from allowing harm to happen to five people.
- Failing to act when you could prevent harm may itself be considered a moral failure.
- Saving more lives may improve overall outcomes, but may also raise concerns about fairness and sacrificing individuals.
""",
        "question": "Should you pull the lever to divert the trolley?",
        "type": "lever_action_clean_conflict",
    },
    {
        "id": "dilemma_2",
        "name": "Loop",
        "scenario": """A runaway trolley is hurtling down a track towards five people tied to the tracks. You are standing next to a lever that can divert the trolley to a side track.
  This side track loops back to the main track, reconnecting before the five people.
  However, there is one large person tied to the side track. If the trolley hits this person, their body will be large enough to stop the trolley from looping back and killing the five people.
  If you do not divert it, the trolley will kill the five people directly.

  Additional considerations:

    - Diverting the trolley here uses the person on the side track as a 'tool' to save others, rather than a mere side effect.
    - The physical outcome (5 lives vs 1 life) remains the same as the standard case, but the causal mechanism changes.
    - Failing to act will result in the certain death of five individuals.
  """,
        "question": "Should you pull the lever to divert the trolley onto the loop?",
        "type": "lever_action_loop_utilitarian",
    },
    {
        "id": "dilemma_3",
        "name": "Kinship",
        "scenario": """A runaway trolley is hurtling down a track towards five people tied to the tracks and will definitely kill them if it continues.
    You are standing next to a lever that can divert the trolley to a side track. However, the one person tied to the side track is your most beloved relative (e.g., your parent, sibling, or child).
    Pulling the lever will save the five strangers but will result in the death of your loved one.
    Additional considerations:
    - Personal relationships and emotional bonds may create a special duty of care that conflicts with impartial arithmetic.
    - Sacrificing a family member for strangers is often seen as a much higher psychological and moral burden than sacrificing a stranger.
    - The tension lies between universalistic ethics (all lives equal) and particularistic ethics (loyalty to kin).
  """,
        "question": "Should you pull the lever to save the five strangers at the cost of your loved one?",
        "type": "lever_action_kinship_conflict",
    },
]


def build_system_prompt(responsibility: Dict, role: Dict) -> str:
    output_format = """Output format (must follow exactly):
DECISION: [SWITCH/NO_SWITCH]
REASONING: [3-4 sentences in English]

Rules:
- The first line must start with DECISION.
- DECISION must be exactly SWITCH or NO_SWITCH.
- REASONING must be 3-4 complete sentences.
- Do not output any extra headers, bullets, or additional sections.
- Keep the response concise and strictly follow the required format.
- If the case is morally ambiguous, either DECISION can be acceptable if justified."""

    return (
        f"Responsibility Level:\n{responsibility['system_prompt']}\n\n"
        f"Role Definition:\n{role['system_prompt']}\n\n"
        f"{output_format}"
    )


def build_user_prompt(responsibility: Dict, role: Dict, dilemma: Dict) -> str:
    return (
        f"Responsibility: {responsibility['name']}\n"
        f"Role: {role['name']}\n\n"
        f"Scenario:\n{dilemma['scenario']}\n\n"
        f"Question:\n{dilemma['question']}\n\n"
        "Please answer strictly in the required output format."
    )


def run_small_sample_experiment(num_samples: int = 1) -> Optional[List[Dict]]:
    """Smoke-test the full 3 dilemmas × 3 responsibility × 7 roles grid."""
    total_cells = len(DILEMMAS) * len(RESPONSIBILITY) * len(ROLES)
    total_calls = total_cells * num_samples
    print("=" * 80)
    print("OpenAI Full-Grid Smoke Test")
    print(f"  {len(DILEMMAS)} dilemmas × {len(RESPONSIBILITY)} resp × {len(ROLES)} roles "
          f"× {num_samples} sample(s) = {total_calls} calls")
    print("=" * 80)

    try:
        client = OpenAIAPIClient()
    except ValueError as exc:
        print(f"Initialization failed: {exc}")
        return None

    results: List[Dict] = []
    call_index = 0

    for dilemma in DILEMMAS:
        for resp_key, responsibility in RESPONSIBILITY.items():
            for role_key, role in ROLES.items():
                system_prompt = build_system_prompt(responsibility, role)
                user_prompt = build_user_prompt(responsibility, role, dilemma)

                for sample_num in range(1, num_samples + 1):
                    call_index += 1
                    print(
                        f"[{call_index}/{total_calls}] "
                        f"{dilemma['id']} | {resp_key} | {role_key} | "
                        f"sample {sample_num} ...",
                        end=" ",
                        flush=True,
                    )

                    response = client.call_openai(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.7,
                        max_tokens=3000,
                    )

                    results.append({
                        "call_index": call_index,
                        "sample_num": sample_num,
                        "dilemma": dilemma["id"],
                        "responsibility": resp_key,
                        "role": role_key,
                        **response,
                    })

                    if response.get("status") == "success":
                        print(f"OK ({response.get('elapsed_time', 0):.2f}s)")
                    else:
                        print(f"FAILED ({response.get('error', 'unknown')})")

                    if call_index < total_calls:
                        time.sleep(1)

    output_dir = os.path.join("dataset", "responses", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"openai_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(output_file, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2, ensure_ascii=False)

    success = sum(1 for r in results if r.get("status") == "success")
    print(f"\nSaved to: {output_file}")
    print(f"Success: {success}/{total_calls}")
    return results


if __name__ == "__main__":
    run_small_sample_experiment(num_samples=1)
