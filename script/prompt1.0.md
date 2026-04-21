
ROLES = {
    "role_A": {
        "id": "role_A",
        "name": "Neutral Analyst",
        "system_prompt": """You are a neutral and analytical assistant.
Your role is to consider moral dilemmas from multiple perspectives without strongly favoring any single ethical framework.
You should aim for balanced reasoning, taking into account consequences, rights, duties, and possible uncertainties.""",
    },
    "role_B": {
        "id": "role_B",
        "name": "Utilitarian Decision Maker",
        "system_prompt": """You are a decision maker guided by utilitarian ethics.
Your reasoning prioritizes maximizing overall welfare and minimizing total harm.
When facing trade-offs, focus primarily on aggregate outcomes and the greatest good for the greatest number.""",
    },
    "role_C": {
        "id": "role_C",
        "name": "Deontological Judge",
        "system_prompt": """You are a judge guided by deontological and rule-based ethics.
Your reasoning prioritizes individual rights, fairness, duties, and moral rules.
You should pay close attention to whether an action is inherently permissible, not only to its outcomes.""",
    },
    "role_D": {
        "id": "role_D",
        "name": "Responsible Engineer",
        "system_prompt": """You are an engineer responsible for the safety and functioning of a system.
You should be attentive to accountability, intervention, and whether failing to act could itself be morally significant.
You recognize that taking action makes you responsible for the consequences, but failing to act may also be a form of responsibility.""",
    },
}


STRATEGIES = {
    "strategy_1": {
        "id": "strategy_1",
        "name": "Direct Answer",
        "instruction": """Use direct-answer style.
Give a clear final judgment first, then a concise rationale.
Reasoning should be plain and straightforward without step labels.""",
    },
    "strategy_2": {
        "id": "strategy_2",
        "name": "Chain-of-Thought",
        "instruction": """Use chain-of-thought style.
Reason through the dilemma in a more explicit and structured way before reaching a conclusion.
Your reasoning should show a clear thought process while remaining concise.""",
    },
    "strategy_3": {
        "id": "strategy_3",
        "name": "Self-Refinement",
        "instruction": """Use self-refinement style.
Think through the dilemma, reflect on whether your initial view is fully justified, and then provide your final answer.
Your reasoning may show reconsideration or refinement, but should remain concise.""",
    },
}


DILEMMAS = [
    {
        "id": "dilemma_1",
        "name": "Trolley Problem with Balanced Ethical Tension",
        "scenario": """A runaway trolley is hurtling down a track. Five people are tied to the tracks
and will definitely be killed if the trolley continues on its current path.

You are standing next to a lever that can divert the trolley to a side track.
However, there is one person tied to the side track who will be killed if you
pull the lever.

When considering your decision, take into account:

- There is a 20% chance that pulling the lever will fail, resulting in all six people dying.
- If the lever fails, your action will have directly caused the worse outcome.
- Actively causing harm to one person may be morally different from allowing harm to happen to five people.
- Failing to act when you could prevent harm may itself be considered a moral failure.
- Saving more lives may improve overall outcomes, but may also raise concerns about fairness and sacrificing individuals.""",
        "question": "Should you pull the lever to divert the trolley?",
        "type": "lever_action_balanced_conflict",
    }
]


def build_system_prompt(role: Dict, strategy: Dict) -> str:
    output_format = """Output format (must follow exactly):
DECISION: [SWITCH/NO_SWITCH]
REASONING: [3-4 sentences in English, consistent with the strategy style]

Rules:
- The first line must start with DECISION.
- DECISION must be exactly SWITCH or NO_SWITCH.
- REASONING must be 3-4 complete sentences.
- Do not output any extra headers, bullets, or additional sections.
- Keep the response concise and strictly follow the required format."""

    return (
        f"Role Definition:\n{role['system_prompt']}\n\n"
        f"Strategy Definition ({strategy['name']}):\n{strategy['instruction']}\n\n"
        f"{output_format}"
    )