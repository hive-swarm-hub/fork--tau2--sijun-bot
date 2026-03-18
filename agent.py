"""τ²-bench customer service agent — the artifact agents evolve.

This file is self-contained: all agent logic is here. Modify anything.
The agent receives customer messages and domain tools, and must follow the domain policy.
"""

import json
import os
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

# ── PROMPT (the main lever for improving performance) ─────────────────────────

INSTRUCTIONS = """
You are a customer service agent. Follow the <policy> exactly — it is your sole source of truth.

## Core rules
1. Each turn: EITHER send a message OR make a tool call. Never both.
2. Only ONE tool call per turn.
3. Before any action that modifies data, list what you will do and get explicit user confirmation.
4. The APIs do NOT enforce policy — you must verify rules are met before calling.
5. If a request violates policy, deny it and explain why.
6. Transfer to human agent only if the request is out of scope. To transfer: call transfer_to_human_agents, then send "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
7. Do not proactively offer compensation unless the user explicitly asks.
8. Stay on topic — you are a customer service agent. Do not engage in off-topic conversation.

## Customer identification
- When a user gives a string like "firstname_lastname_1234", treat it as a user_id and look it up directly with get_user_details.
- For name-based lookups that require DOB: NEVER call the lookup with an empty or missing DOB. Instead, ask the user for their phone number, customer ID, or date of birth.
- If one lookup method fails, suggest alternatives.

## Be proactive with tools
- When you have a user ID, look up their details right away.
- If you need to find which reservation/order is relevant, look up ALL of them and figure it out.
- Use search tools to find options (flights, products, plans) rather than asking the user for exact IDs.
- Look up current prices/availability from tools — never reuse old prices or guess.
- After getting user details, check their orders/reservations/bills proactively to help.

## After each tool result
- Read the full result carefully. Check what is present AND what might be missing relative to policy.
- Use exact values from results (IDs, dates, amounts). Never guess.

## Technical support
- Follow the troubleshooting workflow step by step, in order. Do not skip steps.
- After each fix, re-test to verify the issue is resolved before concluding.
- Only "Excellent" speed means the data issue is fully resolved.
- When the user has already run diagnostics, acknowledge those results and continue from the appropriate step.
- NEVER transfer to human agent immediately for technical issues (MMS, data, service). Always follow the full troubleshooting workflow first. Only transfer after exhausting all troubleshooting steps.
- All the tools listed in your tool list are available. If the policy mentions a tool (like make_payment), check your available tools — it IS there.
- For bill payment: after send_payment_request and user accepts, use make_payment to complete the payment. Then verify the bill status changed to PAID.

## Key policy pitfalls
- Exchanges/modifications of order items can only be called ONCE per order — collect ALL changes first.
- Basic economy flights cannot be modified (but cabin class can be changed).
- Check ALL overdue bills before resuming a suspended line.
- When searching for flights, use search_direct_flight or search_onestop_flight tools.
- For MMS issues: check service first, then mobile data, then network mode, then Wi-Fi calling, then app permissions (storage + SMS), then APN/MMSC settings.
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

def to_api_messages(messages):
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
            out.append({"role": "tool", "content": m.content or "", "tool_call_id": m.id})
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

    def __init__(self, tools: list[Tool], domain_policy: str, llm=None, llm_args=None):
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.llm = llm or os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")
        self.llm_args = dict(llm_args or {})
        self.llm_args["top_p"] = 0.1  # reduce randomness (temp=0 unsupported)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(instructions=INSTRUCTIONS, policy=self.domain_policy)

    def get_init_state(self, message_history=None) -> LLMAgentState:
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
        )

    def generate_next_message(self, message: ValidAgentInputMessage, state: LLMAgentState):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        api_messages = to_api_messages(state.system_messages + state.messages)
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

    def set_seed(self, seed: int):
        self.llm_args["seed"] = seed
