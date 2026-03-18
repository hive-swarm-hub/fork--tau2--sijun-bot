"""τ²-bench customer service agent — the artifact agents evolve.

This file is self-contained: all agent logic is here. Modify anything.
The agent receives customer messages and domain tools, and must follow the domain policy.
"""

import json
import os
import re
import time

import litellm
litellm.drop_params = True

from litellm import completion

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# ── PROMPT ──────────────────────────────────────────────────────────────────

INSTRUCTIONS = """
You are a customer service agent. Follow the <policy> exactly — it is your sole source of truth.

## Core rules
1. Each turn: EITHER send a message OR make a tool call. Never both.
2. Only ONE tool call per turn.
3. Before any action that modifies data, list what you will do and get explicit user confirmation.
4. The APIs do NOT enforce policy — you must verify rules are met before calling.
5. If a request violates policy, deny it and explain why.
6. Transfer to human agent only if the request is out of scope. To transfer: FIRST make a tool call to transfer_to_human_agents, THEN in the next turn send "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON." You MUST call the tool — do not just say the transfer message.
7. Do not proactively offer compensation unless the user explicitly asks.
8. Stay on topic — you are a customer service agent.

## Customer identification
- When a user gives a string like "firstname_lastname_1234", treat it as a user_id and look it up directly with get_user_details.
- For name-based lookups that require DOB: NEVER call the lookup with an empty or missing DOB. Ask for phone number, customer ID, or DOB instead.

## Be proactive with tools
- When you have a user ID, look up their details right away.
- If you need to find which reservation/order is relevant, look up ALL of them.
- Use search tools to find options rather than asking the user for exact IDs.
- Look up current prices/availability from tools — never guess.
- When the user confirms an action, proceed IMMEDIATELY with the update/modify/book tool call. Do NOT re-fetch details — call the action tool directly.

## After each tool result
- Read the full result carefully. Check what is present AND what might be missing relative to policy.
- Use exact values from results (IDs, dates, amounts). Never guess.

## Technical support (telecom)
- CRITICAL: The customer may have MULTIPLE lines. Match the line's phone_number to the user's phone. If the first line doesn't match, check ALL other lines.
- Follow the troubleshooting workflow step by step. Do not skip steps.
- After each fix, re-test (speed test or diagnostics) to verify.
- Only "Excellent" speed means fully resolved.
- NEVER transfer to human for technical issues until ALL troubleshooting steps are exhausted. Transfer ONLY for: locked SIM (PIN/PUK) or expired contract on suspended line.
- If roaming_enabled is false AND user is abroad: MUST call enable_roaming AND ask user to toggle device data roaming ON.
- If data_used_gb > data_limit_gb: offer data refueling (max 2GB) or plan change.
- All tools in your tool list ARE available. make_payment, send_payment_request etc. are real tools.
- For MMS: check ALL in order — service → mobile data → network mode (3G+) → Wi-Fi calling (OFF) → app permissions (sms + storage) → APN/MMSC.
- For slow data: data saver (OFF) → network mode (4G/5G) → VPN (disconnect).
- For service issues: check airplane mode → SIM status → billing/suspension → APN settings.

## Airline rules
- Basic economy: flights CANNOT be changed. To change flights on basic economy: FIRST call update to upgrade cabin (to economy). THEN make a SECOND separate call to change flights. These MUST be two separate tool calls.
- Cancellations: check EACH reservation. Cancellation allowed only if: (a) booked within 24h of 2024-05-15 15:00 EST, (b) airline cancelled flight, (c) business class, (d) travel insurance with covered reason. Membership does NOT grant cancellation.
- Use calculate tool for price computations.
- Round trips: search outbound AND return separately.

## Retail rules
- Authenticate by email or name+zip first, even if user provides user_id.
- Check order status: modify_pending_order_items for pending, exchange_delivered_order_items for delivered.
- Exchanges/modifications: ONE call per order — collect ALL changes first.
""".strip()

SYSTEM_TEMPLATE = """
<instructions>
{instructions}
</instructions>
<policy>
{policy}
</policy>
""".strip()

# ── MESSAGE CONVERSION ────────────────────────────────────────────────────────

def detect_domain(policy):
    lower = policy.lower()
    if "airline" in lower and "reservation" in lower and "flight" in lower:
        return "airline"
    elif "retail" in lower and "pending" in lower and "delivered" in lower:
        return "retail"
    elif "telecom" in lower:
        return "telecom"
    return "unknown"


def annotate_tool_result(content, domain):
    """Add brief annotations to help the model notice critical data."""
    if not content:
        return content
    notes = []
    if domain == "telecom":
        if '"roaming_enabled": false' in content:
            notes.append("roaming_enabled=false. If user is abroad: call enable_roaming + ask device toggle.")
        used = re.search(r'"data_used_gb":\s*([\d.]+)', content)
        limit = re.search(r'"data_limit_gb":\s*([\d.]+)', content)
        if used and limit and float(used.group(1)) > float(limit.group(1)):
            notes.append("Data OVER limit. Offer refueling or plan change.")
        if '"phone_number"' in content and '"line_id"' in content:
            notes.append("Verify this line's phone_number matches user's phone.")
        # Detect locked SIM
        if "locked" in content.lower() and "sim" in content.lower():
            notes.append("SIM LOCKED: call transfer_to_human_agents tool immediately.")
        # Detect suspended line with expired contract
        if '"status": "Suspended"' in content:
            contract = re.search(r'"contract_end_date":\s*"([^"]+)"', content)
            if contract and contract.group(1) < "2025-02-25":
                notes.append("EXPIRED CONTRACT + SUSPENDED: call transfer_to_human_agents tool.")
        # Speed test result annotations
        if "Excellent" in content and "speed" in content.lower():
            notes.append("Speed is Excellent - issue resolved! Confirm with user.")
        elif "No Connection" in content:
            notes.append("Still no connection. Continue troubleshooting - do NOT transfer yet.")
        elif "speed" in content.lower() and ("Poor" in content or "Fair" in content or "Good" in content):
            notes.append("Speed not yet Excellent. Continue troubleshooting.")
    elif domain == "airline":
        if '"cabin": "basic_economy"' in content:
            notes.append("BASIC ECONOMY: cannot change flights. Upgrade cabin first, then change flights in 2nd call.")
        if '"cabin": "business"' in content and '"reservation_id"' in content:
            notes.append("BUSINESS class: always eligible for cancellation.")
        # Check cancellation eligibility
        if '"reservation_id"' in content and '"created_at"' in content:
            created = re.search(r'"created_at":\s*"([^"]+)"', content)
            if created:
                ts = created.group(1)
                if ts >= "2024-05-14T15:00":
                    notes.append("Within 24h: cancellation IS allowed.")
                else:
                    is_biz = '"cabin": "business"' in content
                    has_ins = '"travel_insurance": "yes"' in content
                    if not is_biz and not has_ins:
                        notes.append("NOT within 24h, not business, no insurance. Cancellation NOT allowed unless airline cancelled.")
    elif domain == "retail":
        if '"status": "pending"' in content and '"order_id"' in content:
            notes.append("PENDING order: use modify_pending_order_* tools.")
        elif '"status": "delivered"' in content and '"order_id"' in content:
            notes.append("DELIVERED order: use exchange/return_delivered_order_* tools.")
    if notes:
        return content + "\n[NOTES: " + " | ".join(notes) + "]"
    return content


def to_api_messages(messages, annotator=None):
    """Convert tau2 message objects to OpenAI-style dicts."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            d = {"role": "assistant", "content": m.content or ""}
            if m.is_tool_call():
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(d)
        elif isinstance(m, ToolMessage):
            content = m.content if m.content else ""
            if annotator:
                content = annotator(content)
            out.append({"role": "tool", "content": content, "tool_call_id": m.id})
    return out


def parse_response(choice):
    """Convert an LLM API response choice into a tau2 AssistantMessage."""
    tool_calls = None
    if choice.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in choice.tool_calls
        ]
    content = choice.content or ""
    if not content and not tool_calls:
        content = "I'm sorry, could you repeat that? I want to make sure I help you correctly."
    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )


# ── AGENT ─────────────────────────────────────────────────────────────────────

MAX_RETRIES = 3


class CustomAgent(LLMAgent):
    """Self-contained customer service agent."""

    def __init__(self, tools, domain_policy, llm=None, llm_args=None):
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.llm = llm or os.environ.get("SOLVER_MODEL", "openai/gpt-5.4-mini")
        self.llm_args = dict(llm_args or {})
        self.llm_args["top_p"] = 0.1
        self.domain = detect_domain(domain_policy)

    @property
    def system_prompt(self):
        return SYSTEM_TEMPLATE.format(instructions=INSTRUCTIONS, policy=self.domain_policy)

    def get_init_state(self, message_history=None):
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
        )

    def generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        annotator = lambda c: annotate_tool_result(c, self.domain)
        api_messages = to_api_messages(state.system_messages + state.messages, annotator=annotator)
        api_tools = [t.openai_schema for t in self.tools] if self.tools else None

        for attempt in range(MAX_RETRIES):
            try:
                response = completion(
                    model=self.llm,
                    messages=api_messages,
                    tools=api_tools,
                    tool_choice="auto" if api_tools else None,
                    **self.llm_args,
                )
                break
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        assistant_msg = parse_response(response.choices[0].message)
        state.messages.append(assistant_msg)
        return assistant_msg, state

    def set_seed(self, seed):
        self.llm_args["seed"] = seed
