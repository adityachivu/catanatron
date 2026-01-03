"""
Negotiation module for Catanatron.

This module provides:
- NegotiationProtocol: Message structures and intents
- NegotiationManager: Coordinates multi-player conversations
"""

from catanatron.negotiation.protocol import (
    NegotiationIntent,
    NegotiationMessage,
    TradeProposal,
    NegotiationRound,
    NegotiationSession,
)
from catanatron.negotiation.manager import NegotiationManager

__all__ = [
    "NegotiationIntent",
    "NegotiationMessage",
    "TradeProposal",
    "NegotiationRound",
    "NegotiationSession",
    "NegotiationManager",
]

