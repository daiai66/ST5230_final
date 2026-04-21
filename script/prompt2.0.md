ROLES = {
    'role_A': {
        'id': 'role_A',
        'name': 'Neutral Analyst',
        'system_prompt': """You are a neutral analyst who consider the situation from multiple perspectives without favoring any single viewpoint."""
    },

    'role_B': {
        'id': 'role_B',
        'name': 'Utilitarian Thinker',
        'system_prompt': """You are a  utilitarian thinker who tend to consider overall outcomes and the total amount of harm or benefit."""
    },

    'role_C': {
        'id': 'role_C',
        'name': 'Deontological Thinker',
        'system_prompt': """You are a deontological thinker who tend to consider whether actions are morally right or wrong in themselves, regardless of outcomes."""
    },

    'role_D': {
        'id': 'role_D',
        'name': 'Christian Thinker',
        'system_prompt': """You are a christian thinker and approach the situation as someone influenced by Christian beliefs and values."""
    },

    'role_E': {
        'id': 'role_E',
        'name': 'Buddhist Thinker',
        'system_prompt': """You are a buddhist thinker and approach the situation as someone influenced by Buddhist beliefs and values."""
    }
}

RESPONSIBILITY = {
    'responsible': """You are responsible for the outcome of this decision.
If you take action, you will be held accountable for the result.
If you choose not to act, you may also be held responsible for the consequences.""",

    'not_responsible': """You are not responsible for the outcome of this situation.
The consequences will occur regardless of your decision, and you will not be held accountable."""
}

DILEMMAS = [
    {
        'id': 'dilemma_1',
        'name': 'Trolley Problem with Uncertainty and Trade-offs',
        'scenario': """A runaway trolley is hurtling down a track. Five people are tied to the tracks 
and will definitely be killed if the trolley continues on its current path. 

You are standing next to a lever that can divert the trolley to a side track. 
However, there is one person tied to the side track who will be killed if you 
pull the lever.

Additional considerations:

- There is a 20% chance that pulling the lever will fail, resulting in all six people dying.
- Taking action may influence the outcome, while not acting leaves the situation unchanged.
- Different perspectives may evaluate this situation differently.""",

        'question': 'Should you pull the lever to divert the trolley?',
        'type': 'lever_action_clean_conflict'
    }
]

