"""
Example: LLM Players with Natural Language Negotiation

This example demonstrates how to use LLM-powered players that can:
1. Make strategic decisions using AI rankings
2. Engage in natural language negotiation before trades
3. Track reputation and honor agreements

Requirements:
    pip install catanatron[llm]
    
    For OpenAI: set OPENAI_API_KEY environment variable
    For Anthropic: set ANTHROPIC_API_KEY environment variable
    For Ollama: run ollama locally with your preferred model
"""

import asyncio
from catanatron import Game
from catanatron.models.player import Color, RandomPlayer

# Import LLM player components
from catanatron.players.llm import (
    create_llm_player,
    create_mock_player,
    StrategicAdvisor,
    StateRenderer,
    LLMConfig,
)
from catanatron.negotiation import (
    NegotiationManager,
    TradeProposal,
    NegotiationSession,
)


def example_mock_player_game():
    """
    Example 1: Mock LLM Players (no API calls needed)
    
    This demonstrates the system without actual LLM calls.
    Mock players use the strategic advisor to pick actions.
    """
    print("=" * 60)
    print("Example 1: Mock LLM Player Game")
    print("=" * 60)
    
    # Create mock players that use strategic rankings
    players = [
        create_mock_player(Color.RED, strategy="top_ranked"),
        create_mock_player(Color.BLUE, strategy="top_ranked"),
    ]
    
    # Create and play game
    game = Game(players, seed=42)
    
    print(f"Starting game with {len(players)} mock LLM players...")
    
    # Play for a limited number of turns
    max_turns = 100
    for turn in range(max_turns):
        if game.winning_color() is not None:
            break
        game.play_tick()
        
        if turn % 20 == 0:
            print(f"  Turn {game.state.num_turns}: Game in progress...")
    
    winner = game.winning_color()
    if winner:
        print(f"\nGame finished! Winner: {winner.value}")
    else:
        print(f"\nGame in progress after {game.state.num_turns} turns")
    
    # Print decision logs from players
    for player in players:
        if hasattr(player, 'decision_log') and player.decision_log:
            print(f"\n{player.color.value}'s decisions:")
            for log in player.decision_log[:3]:  # First 3 decisions
                print(f"  Turn {log['turn']}: Chose #{log['chosen_index']} - {log['reasoning'][:50]}...")


def example_strategic_advisor():
    """
    Example 2: Strategic Advisor (standalone)
    
    Shows how the advisor ranks actions without LLM calls.
    """
    print("\n" + "=" * 60)
    print("Example 2: Strategic Advisor")
    print("=" * 60)
    
    # Create a game
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
    ]
    game = Game(players, seed=42)
    
    # Play past initial placement
    for _ in range(20):
        game.play_tick()
    
    # Use advisor to rank actions
    advisor = StrategicAdvisor()
    rankings = advisor.rank_actions(game, game.playable_actions, top_n=5)
    
    print(f"\nCurrent player: {game.state.current_color().value}")
    print(f"Available actions: {len(game.playable_actions)}")
    print("\nTop 5 recommended actions:")
    
    for ranking in rankings:
        print(f"  #{ranking.rank}: {ranking.action.action_type.value}")
        print(f"       Value: {ranking.normalized_value:.2f}")
        print(f"       Explanation: {ranking.explanation}")


def example_state_renderer():
    """
    Example 3: State Renderer
    
    Shows how game state is rendered for LLM consumption.
    """
    print("\n" + "=" * 60)
    print("Example 3: State Renderer")
    print("=" * 60)
    
    # Create a game
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
    ]
    game = Game(players, seed=42)
    
    # Play some turns
    for _ in range(30):
        game.play_tick()
    
    # Render state
    renderer = StateRenderer()
    advisor = StrategicAdvisor()
    
    color = game.state.current_color()
    rankings = advisor.rank_actions(game, game.playable_actions, top_n=3)
    
    context = renderer.render_full_context(
        game,
        color,
        rankings=rankings,
        memory_context="(No prior negotiations)",
    )
    
    print(f"\nRendered context for {color.value}:")
    print("-" * 40)
    # Print first 2000 characters
    print(context[:2000])
    if len(context) > 2000:
        print(f"... ({len(context) - 2000} more characters)")


def example_negotiation_protocol():
    """
    Example 4: Negotiation Protocol
    
    Shows how negotiation messages and sessions work.
    """
    print("\n" + "=" * 60)
    print("Example 4: Negotiation Protocol")
    print("=" * 60)
    
    from catanatron.negotiation.protocol import (
        NegotiationMessage,
        NegotiationIntent,
        create_proposal_message,
        create_accept_message,
    )
    
    # Create a trade proposal
    proposal = TradeProposal(
        offering=(2, 0, 0, 0, 0),  # 2 wood
        asking=(0, 0, 0, 1, 0),     # 1 wheat
    )
    
    print(f"Trade proposal: {proposal.format_readable()}")
    print(f"Valid: {proposal.is_valid()}")
    print(f"Trade tuple: {proposal.to_trade_tuple()}")
    
    # Create a negotiation session
    session = NegotiationSession(
        session_id="demo123",
        initiator=Color.RED,
        participants=[Color.RED, Color.BLUE, Color.ORANGE],
        start_turn=10,
    )
    
    # Add messages
    session.add_message(create_proposal_message(
        sender=Color.RED,
        proposal=proposal,
        message="I have extra wood. Anyone need some? 2 wood for 1 wheat!",
    ))
    
    session.add_message(NegotiationMessage(
        sender=Color.BLUE,
        content="That's a bit steep. How about 2 wood for 2 wheat?",
        intent=NegotiationIntent.COUNTER,
    ))
    
    session.add_message(create_accept_message(
        sender=Color.RED,
        message="Deal! I'll take it.",
        proposer=Color.BLUE,
    ))
    
    print(f"\nNegotiation session: {session.session_id}")
    print(f"Participants: {[p.value for p in session.participants]}")
    print(f"Messages: {len(session.get_all_messages())}")
    print("\nConversation log:")
    print(session.get_conversation_log())


def example_real_llm_player():
    """
    Example 5: Real LLM Player (requires API key)
    
    This example requires either:
    - OPENAI_API_KEY environment variable set, OR
    - ANTHROPIC_API_KEY environment variable set, OR
    - GOOGLE_API_KEY environment variable set, OR
    - Ollama running locally
    
    Uncomment to use:
    """
    print("\n" + "=" * 60)
    print("Example 5: Real LLM Player")
    print("=" * 60)
    
    print("""
To use real LLM players, uncomment the code below and set up your API keys:

# For OpenAI:
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

player = create_llm_player(
    Color.RED,
    provider="openai",
    model="gpt-4",
    persona="balanced"
)

# For Anthropic:
player = create_llm_player(
    Color.RED,
    provider="anthropic", 
    model="claude-3-sonnet-20240229",
    persona="cooperative"
)

# For Google Gemini:
os.environ["GOOGLE_API_KEY"] = "your-key-here"

player = create_llm_player(
    Color.RED,
    provider="gemini",
    model="gemini-2.5-pro",  # or "gemini-2.5-flash" for faster/cheaper
    persona="balanced"
)

# For Ollama (local):
player = create_llm_player(
    Color.RED,
    provider="ollama",
    model="llama3",
    persona="competitive"
)
""")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CATANATRON LLM NEGOTIATION EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_mock_player_game()
    example_strategic_advisor()
    example_state_renderer()
    example_negotiation_protocol()
    example_real_llm_player()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

