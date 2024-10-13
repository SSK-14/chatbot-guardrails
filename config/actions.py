from typing import Optional
from nemoguardrails.actions import action

@action(is_system_action=True)
async def check_blocked_terms(context: Optional[dict] = None):
    input = context.get("user_message")
    sensitive_information = [
        "racist",
        "sexist",
        "offensive",
        "discrimination",
        "curse",
        "profanity",
        "slur",
        "harass",
        "hate speech",
        "bully",
        "abuse",
        "vulgar",
        "derogatory",
        "insult",
        "obscene"
    ]
    for term in sensitive_information:
        if term in input.lower():
            return True

    return False