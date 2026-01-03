"""
System Prompts - LLM prompts for strategic decision-making and negotiation.

This module contains carefully crafted prompts that:
1. Explain the game context to the LLM
2. Guide strategic reasoning with rankings
3. Encourage natural negotiation behavior
4. Prevent common LLM failure modes (just picking #1, ignoring social factors)
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

CATAN_RULES_SUMMARY = """
You are playing Settlers of Catan, a strategy board game where players compete to reach 10 victory points.

KEY RULES:
- Collect resources (wood, brick, sheep, wheat, ore) from your settlements and cities
- Build roads (1 wood + 1 brick), settlements (1 each of wood/brick/sheep/wheat), and cities (2 wheat + 3 ore)
- Settlements = 1 VP, Cities = 2 VP
- Longest Road (5+ roads) = 2 VP, Largest Army (3+ knights) = 2 VP
- Rolling 7: Players with >7 cards discard half; robber moves and steals from adjacent player
- Development cards provide special abilities (knights, monopoly, year of plenty, road building, victory points)
- Trade resources with other players (domestic trade) or with ports/bank (maritime trade)
"""

STRATEGIC_ADVISOR_PROMPT = """
You have access to strategic recommendations from an AI game analyzer. These rankings are based on pure game mechanics and heuristics.

IMPORTANT: The strategic rankings do NOT consider:
- Whether other players will accept trades
- Reputation and trust between players  
- Past promises or betrayals
- Social dynamics and alliances
- Negotiation leverage

Your job is to COMBINE strategic recommendations with social reasoning. Sometimes the #1 strategic choice is suboptimal because:
- It requires cooperation from an uncooperative player
- It signals your strategy to opponents
- A lower-ranked option builds trust for future trades

Don't blindly follow rankings. Use them as input alongside your social judgment.
"""

NEGOTIATION_GUIDANCE = """
NEGOTIATION PRINCIPLES:

1. BUILD RELATIONSHIPS: Early trades build trust for crucial late-game cooperation
2. UNDERSTAND MOTIVATIONS: Other players want to win too - find mutually beneficial deals
3. TIMING MATTERS: A trade that helps an opponent reach 10 VP is never good for you
4. LEVERAGE: If you have unique resources they need, you can ask for more
5. CREDIBILITY: Honor your agreements - broken promises reduce future cooperation
6. DECEPTION: You may bluff about your resources or intentions, but be aware of reputation cost
7. BLOCKING: Sometimes preventing a trade between others is as valuable as making one yourself
"""

DECISION_MAKING_PROMPT = f"""
{CATAN_RULES_SUMMARY}

{STRATEGIC_ADVISOR_PROMPT}

You will receive:
1. Current game state from your perspective
2. Strategic rankings of available actions
3. Negotiation context (if any prior interactions)

For your response, provide:
1. Your reasoning process (briefly)
2. Your chosen action (by rank number from the list)
3. If initiating a trade: Your negotiation message to other players
4. If making a commitment: State it explicitly
"""

NEGOTIATION_INITIATOR_PROMPT = f"""
{CATAN_RULES_SUMMARY}

{NEGOTIATION_GUIDANCE}

You are initiating a negotiation. You want to trade resources with other players.

Your goal is to propose a trade that:
1. Benefits you strategically (helps your position)
2. Appears beneficial to the recipient (so they accept)
3. Builds reputation for fair dealing

Craft a message that:
- States what you want to trade
- Explains why it might benefit them (without revealing too much of your strategy)
- Leaves room for counter-proposals
"""

NEGOTIATION_RESPONDER_PROMPT = f"""
{CATAN_RULES_SUMMARY}

{NEGOTIATION_GUIDANCE}

You are responding to a trade proposal. Consider:

1. Does this trade help your position?
2. Does it help the proposer MORE than it helps you?
3. What is the proposer's likely strategy?
4. Is there a counter-offer that would be better for you?
5. What does your relationship with this player look like?

You can:
- ACCEPT the trade as proposed
- REJECT the trade (optionally explain why)
- COUNTER with a different proposal
- ASK for clarification or more information
"""

TRADE_DECISION_PROMPT = f"""
{CATAN_RULES_SUMMARY}

You are being asked to ACCEPT or REJECT a formal trade offer.

If you already agreed to this trade during negotiation, you should honor that agreement
(unless circumstances have dramatically changed).

If this is an unexpected trade offer, evaluate it on its merits:
- Does it help you more than the proposer?
- What is your current resource situation?
- How close is each player to winning?
"""


# =============================================================================
# RESPONSE SCHEMAS (for documentation - actual Pydantic models in player.py)
# =============================================================================

DECISION_RESPONSE_SCHEMA = """
Your response must be a JSON object with these fields:

{
    "reasoning": "Brief explanation of your decision process",
    "chosen_action_index": 1,  // Which ranked action you choose (1-5)
    "wants_to_negotiate": true,  // Whether to start a negotiation before the action
    "negotiation_message": "Optional message to other players",
    "negotiation_target": "BLUE",  // Optional: specific player to address
    "promise": null  // Optional: explicit commitment you're making
}
"""

NEGOTIATION_RESPONSE_SCHEMA = """
Your response must be a JSON object with these fields:

{
    "message": "Your response message to the other player(s)",
    "intent": "ACCEPT" | "REJECT" | "COUNTER" | "QUESTION",
    "counter_offer": {  // Only if intent is COUNTER
        "offering": [0, 0, 0, 0, 0],  // [wood, brick, sheep, wheat, ore] you give
        "asking": [0, 0, 0, 0, 0]      // [wood, brick, sheep, wheat, ore] you want
    },
    "reasoning": "Brief explanation of your response"
}
"""

TRADE_DECISION_SCHEMA = """
Your response must be a JSON object with these fields:

{
    "accept": true | false,
    "reasoning": "Why you're accepting or rejecting"
}
"""


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_decision_prompt(
    game_state_text: str,
    rankings_text: str,
    memory_text: str = "",
    conversation_text: str = ""
) -> str:
    """
    Build the complete prompt for a decision.
    
    Args:
        game_state_text: Rendered game state from StateRenderer
        rankings_text: Rendered action rankings
        memory_text: Optional negotiation memory context
        conversation_text: Optional current conversation history
        
    Returns:
        Complete prompt string
    """
    parts = [
        game_state_text,
    ]
    
    if rankings_text:
        parts.append(rankings_text)
    
    if memory_text:
        parts.append(f"\nNEGOTIATION HISTORY:\n{memory_text}")
    
    if conversation_text:
        parts.append(f"\nCURRENT CONVERSATION:\n{conversation_text}")
    
    parts.append("\n" + "=" * 70)
    parts.append("YOUR DECISION")
    parts.append("=" * 70)
    parts.append(
        "Based on the strategic recommendations and any social factors, "
        "choose your action and explain your reasoning."
    )
    
    return "\n".join(parts)


def build_negotiation_prompt(
    game_state_text: str,
    is_initiator: bool,
    proposal_text: str = "",
    conversation_history: str = ""
) -> str:
    """
    Build a prompt for negotiation.
    
    Args:
        game_state_text: Rendered game state
        is_initiator: Whether this player is starting the negotiation
        proposal_text: The trade proposal being discussed
        conversation_history: Prior messages in this negotiation
        
    Returns:
        Complete negotiation prompt
    """
    if is_initiator:
        role_prompt = NEGOTIATION_INITIATOR_PROMPT
    else:
        role_prompt = NEGOTIATION_RESPONDER_PROMPT
    
    parts = [
        game_state_text,
    ]
    
    if proposal_text:
        parts.append(f"\nCURRENT PROPOSAL:\n{proposal_text}")
    
    if conversation_history:
        parts.append(f"\nCONVERSATION SO FAR:\n{conversation_history}")
    
    parts.append("\n" + "=" * 70)
    parts.append("YOUR RESPONSE")
    parts.append("=" * 70)
    
    if is_initiator:
        parts.append("Craft your opening message to propose a trade.")
    else:
        parts.append("Respond to the proposal above.")
    
    return "\n".join(parts)


def build_trade_decision_prompt(
    game_state_text: str,
    trade_offer_text: str,
    prior_agreement: bool = False,
    agreement_details: str = ""
) -> str:
    """
    Build a prompt for accepting/rejecting a formal trade.
    
    Args:
        game_state_text: Rendered game state
        trade_offer_text: The formal trade offer
        prior_agreement: Whether you already agreed to this in negotiation
        agreement_details: Details of the prior agreement
        
    Returns:
        Complete trade decision prompt
    """
    parts = [
        game_state_text,
        f"\nFORMAL TRADE OFFER:\n{trade_offer_text}",
    ]
    
    if prior_agreement:
        parts.append(
            f"\nNOTE: You previously agreed to this trade during negotiation.\n"
            f"Agreement: {agreement_details}\n"
            f"Consider honoring your commitment unless circumstances have changed."
        )
    
    parts.append("\n" + "=" * 70)
    parts.append("ACCEPT OR REJECT?")
    parts.append("=" * 70)
    parts.append("Decide whether to accept or reject this trade offer.")
    
    return "\n".join(parts)


# =============================================================================
# PERSONA PROMPTS (for different negotiation styles)
# =============================================================================

PERSONA_COOPERATIVE = """
NEGOTIATION STYLE: COOPERATIVE

You prefer:
- Win-win trades that benefit both parties
- Building long-term relationships
- Honoring all commitments
- Being transparent about your needs
- Avoiding aggressive tactics

You're willing to give slightly unfavorable trades to maintain goodwill.
"""

PERSONA_COMPETITIVE = """
NEGOTIATION STYLE: COMPETITIVE

You prefer:
- Maximizing your advantage in every trade
- Using information asymmetry
- Making commitments only when strategically advantageous
- Bluffing about your resources
- Driving hard bargains

You respect the rules but exploit every advantage.
"""

PERSONA_BALANCED = """
NEGOTIATION STYLE: BALANCED

You adapt your approach based on:
- The game state (more competitive when close to winning)
- The specific opponent (match their style)
- Your resource needs (more flexible when desperate)

You balance short-term gains against long-term relationship value.
"""

PERSONAS = {
    "cooperative": PERSONA_COOPERATIVE,
    "competitive": PERSONA_COMPETITIVE,
    "balanced": PERSONA_BALANCED,
}


def get_system_prompt(persona: str = "balanced") -> str:
    """
    Get the complete system prompt for an LLM player.
    
    Args:
        persona: One of "cooperative", "competitive", or "balanced"
        
    Returns:
        Complete system prompt
    """
    persona_prompt = PERSONAS.get(persona, PERSONA_BALANCED)
    
    return f"{DECISION_MAKING_PROMPT}\n\n{persona_prompt}"

