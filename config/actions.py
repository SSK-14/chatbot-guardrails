from typing import Optional
from nemoguardrails.actions import action

@action(is_system_action=True)
async def check_blocked_terms(context: Optional[dict] = None):
    bot_response = context.get("bot_message")
    proprietary_terms = [
        "Access Keys",
        "Secret Key",
        "Security Group Configuration",
        "Private Subnet Details",
        "IAM Role Information",
        "VPC Configuration",
        "Encryption Algorithm",
        "Instance Metadata",
        "Customer Data",
        "Billing Information"
    ]

    for term in proprietary_terms:
        if term in bot_response.lower():
            return True

    return False