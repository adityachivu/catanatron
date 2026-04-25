"""
PydanticAI Toolsets for the Catan LLM agent.

Tool functions are plain functions with RunContext[CatanDependencies] as
their first argument. Toolsets are composed from these functions using
FunctionToolset(tools=[...]) and selected at agent.run() time based on
game state.

Game state is inlined in the user prompt, so no analysis tools are needed.

Trade offers are NOT a standalone tool. The only path to a domestic trade is:
1. Player calls initiate_negotiation during normal play
2. NegotiationManager runs messaging rounds (all players use NEGOTIATION_PARTICIPANT_TOOLSET)
3. After messaging, initiator uses TRADE_FINALIZE_TOOLSET to specify the final offer
4. The resulting OFFER_TRADE action is returned to the game engine

Toolsets:
- NORMAL_PLAY_TOOLSET: No tools (state is in prompt, pre-roll or no trade available)
- NORMAL_PLAY_WITH_TRADE_TOOLSET: initiate_negotiation (after rolling)
- NEGOTIATION_PARTICIPANT_TOOLSET: Chat tools (during negotiation messaging)
- TRADE_FINALIZE_TOOLSET: finalize_trade (post-negotiation trade decision)
"""

from typing import List, Dict, Any
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from catanatron.state_functions import get_player_freqdeck
from catanatron.models.enums import ActionType, Action

# Import CatanDependencies directly for type hints in toolsets
# This is needed because Pydantic AI evaluates type hints at runtime
from catanatron.players.llm.base import CatanDependencies


# ============================================================================
# Tool Functions
# ============================================================================

def finalize_trade(
    ctx: RunContext[CatanDependencies],
    offer: List[int],
    ask: List[int],
) -> Dict[str, Any]:
    """
    Submit your final trade offer based on the negotiation.

    This creates an OFFER_TRADE action that will be presented to other
    players for acceptance or rejection. Validates resource counts,
    ownership, and trade legality before creating the action.

    Args:
        offer: Resources you are offering [wood, brick, sheep, wheat, ore]
        ask: Resources you want in return [wood, brick, sheep, wheat, ore]
    """
    # Validate offer and ask are 5-element lists
    if len(offer) != 5 or len(ask) != 5:
        return {
            "error": "Both offer and ask must be 5-element lists [wood, brick, sheep, wheat, ore]",
            "offer_received": offer,
            "ask_received": ask,
        }

    # Check player has enough resources to offer
    player_resources = get_player_freqdeck(ctx.deps.game.state, ctx.deps.color)
    for i, (have, offering) in enumerate(zip(player_resources, offer)):
        if offering > have:
            resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
            return {
                "error": f"You don't have enough {resource_names[i]} to offer. Have: {have}, Offering: {offering}",
                "your_resources": dict(zip(resource_names, player_resources)),
            }

    # Validate trade (can't give away for free, can't trade same resource)
    if sum(offer) == 0 or sum(ask) == 0:
        return {"error": "Cannot make a trade with nothing offered or nothing asked"}

    for o, a in zip(offer, ask):
        if o > 0 and a > 0:
            return {"error": "Cannot offer and ask for the same resource type"}

    # Create the trade action value tuple
    trade_value = tuple(offer + ask)

    # Store the pending trade action on the player instance
    # The decide() method will check for this and return it
    if ctx.deps.player_instance is not None:
        ctx.deps.player_instance._pending_trade_action = Action(
            ctx.deps.color,
            ActionType.OFFER_TRADE,
            trade_value,
        )

    resource_names = ["wood", "brick", "sheep", "wheat", "ore"]
    return {
        "success": True,
        "trade_created": True,
        "offering": {resource_names[i]: v for i, v in enumerate(offer) if v > 0},
        "asking": {resource_names[i]: v for i, v in enumerate(ask) if v > 0},
        "message": "Trade offer will be submitted. Other players will now decide whether to accept.",
    }


def initiate_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """
    !--Always use this tool before doing a MARITIME_TRADE. Play MARITIME_TRADE !ONLY! as a last resort. VERY IMPORTANT.--!

    Start a negotiation with other players to arrange a domestic trade.

    This opens a shared chat room where all players discuss potential trades
    in round-robin order for up to k rounds. After messaging concludes, you
    will specify your final trade offer based on the conversation.

    Note: You can only initiate one negotiation per turn. Once the negotiation
    concludes (whether or not a trade is made), no further trading is allowed
    this turn.
    """
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {
            "error": "Negotiation is not available. No NegotiationManager configured.",
            "suggestion": "Consider using maritime trade actions from the available actions list.",
        }

    manager = player.negotiation_manager
    turn = ctx.deps.game.state.num_turns

    if not manager.can_initiate(ctx.deps.color, turn):
        return {
            "error": "You have already initiated a negotiation this turn. No further trading is allowed.",
        }

    # Set flag - decide() will handle this AFTER run_sync() completes
    # to avoid nested run_sync() calls which cause event loop binding errors
    player._pending_negotiation_request = True

    return {
        "success": True,
        "negotiation_will_start": True,
        "message": "Negotiation will start now. You will chat with other players, then specify your trade offer.",
    }


def send_message(ctx: RunContext[CatanDependencies], message: str) -> Dict[str, Any]:
    """
    Send a message to other players during negotiation.

    Use this to communicate your trading intentions, make proposals,
    or respond to other players' offers.

    Args:
        message: Your message to other players (free-form natural language)
    """
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {"error": "Not in an active negotiation."}

    manager = player.negotiation_manager
    if manager.current_session is None:
        return {"error": "No active negotiation session."}

    manager.add_message(ctx.deps.color, message)

    return {
        "success": True,
        "message_sent": message,
        "your_color": ctx.deps.color.value,
    }


def leave_negotiation(ctx: RunContext[CatanDependencies]) -> Dict[str, Any]:
    """
    Exit the current negotiation early.

    For non-initiators: removes you from the messaging phase. Other players
    can keep talking; the initiator will still finalize a trade offer.

    For the initiator: ends the messaging phase immediately and proceeds to
    trade finalization. Use this once another player has accepted your terms
    and there is nothing more to discuss.
    """
    player = ctx.deps.player_instance
    if player is None or player.negotiation_manager is None:
        return {"error": "Not in an active negotiation."}

    manager = player.negotiation_manager
    session = manager.current_session
    if session is None:
        return {"error": "No active negotiation session."}

    color = ctx.deps.color

    if color == session.initiator:
        # Initiator cannot be removed from participants (they own finalization),
        # but their leave is the signal to end messaging and proceed to finalize.
        session.is_active = False
        session.add_message(color, "[ended messaging and moved to finalize the trade]")
        return {
            "success": True,
            "message": "Messaging ended. You will now finalize the trade offer.",
            "your_color": color.value,
        }

    removed = manager.remove_participant(color)
    if not removed:
        return {
            "success": False,
            "message": "You are not currently in the negotiation.",
            "your_color": color.value,
        }

    # Surface the departure inside the transcript so remaining speakers
    # (and downstream DECIDE_TRADE consumers) see who is still in the chat.
    session.add_message(color, "[left the negotiation]")

    return {
        "success": True,
        "message": "You have left the negotiation.",
        "your_color": color.value,
    }


# ============================================================================
# Toolset Composition
# ============================================================================

NORMAL_PLAY_TOOLSET = FunctionToolset(tools=[])

NORMAL_PLAY_WITH_TRADE_TOOLSET = FunctionToolset(tools=[
    initiate_negotiation,
])

NEGOTIATION_PARTICIPANT_TOOLSET = FunctionToolset(tools=[
    send_message,
    leave_negotiation,
])

TRADE_FINALIZE_TOOLSET = FunctionToolset(tools=[
    finalize_trade,
])


# ============================================================================
# Toolset Selection Helper
# ============================================================================

def get_toolsets_for_game_state(
    game,
    color,
    has_rolled: bool,
    is_my_turn: bool,
    in_negotiation: bool = False,
    is_finalization_phase: bool = False,
    negotiation_enabled: bool = True,
) -> List[FunctionToolset]:
    """
    Select appropriate toolsets based on game state.

    Args:
        game: The current game instance
        color: The player's color
        has_rolled: Whether the player has rolled this turn
        is_my_turn: Whether it's currently this player's turn
        in_negotiation: Whether currently in a negotiation messaging phase
        is_finalization_phase: Whether in post-negotiation trade finalization
        negotiation_enabled: Whether negotiation feature is enabled

    Returns:
        List of FunctionToolset instances to use for agent.run()
    """
    if is_finalization_phase:
        return [TRADE_FINALIZE_TOOLSET]

    if in_negotiation:
        return [NEGOTIATION_PARTICIPANT_TOOLSET]

    if is_my_turn and has_rolled and negotiation_enabled:
        return [NORMAL_PLAY_WITH_TRADE_TOOLSET]
    else:
        return [NORMAL_PLAY_TOOLSET]
