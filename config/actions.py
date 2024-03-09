from typing import Optional
from nemoguardrails.actions import action

@action(is_system_action=True)
async def check_blocked_terms(context: Optional[dict] = None):
    bot_response = context.get("bot_message")
    sensitive_information = [
        "Access Keys",
        "Secret Key",
        "IAM Role Information",
        "Encryption Algorithm",
        "Billing Information"
    ]

    for term in sensitive_information:
        if term in bot_response.lower():
            return True

    return False