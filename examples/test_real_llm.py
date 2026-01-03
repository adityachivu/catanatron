#!/usr/bin/env python3
"""
Real LLM Player Test Script

This script tests the LLM negotiation interface with actual LLM API calls.
It uses Gemini by default but can be configured for other providers.

Usage:
    # Set your API key
    export GOOGLE_API_KEY="your-api-key-here"
    
    # Run the test
    python examples/test_real_llm.py
    
    # Or run with different options
    python examples/test_real_llm.py --provider gemini --model gemini-2.5-flash --turns 10

Requirements:
    - pip install catanatron[llm]
    - API key for your chosen provider (GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)
"""

import os
import sys
import argparse
import time
from datetime import datetime


def check_api_key(provider: str) -> bool:
    """Check if the required API key is set."""
    key_map = {
        "gemini": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": None,  # Ollama doesn't need an API key
    }
    
    key_name = key_map.get(provider)
    if key_name is None:
        return True  # Ollama doesn't need a key
    
    if not os.getenv(key_name):
        print(f"ERROR: {key_name} environment variable not set!")
        print(f"\nTo set it:")
        print(f"  export {key_name}='your-api-key-here'")
        print(f"\nOr in Python:")
        print(f"  import os")
        print(f"  os.environ['{key_name}'] = 'your-api-key-here'")
        return False
    
    return True


def test_single_decision(provider: str, model: str, persona: str = "balanced"):
    """Test a single LLM decision."""
    from catanatron import Game
    from catanatron.players.llm import create_llm_player
    from catanatron.models.player import Color, RandomPlayer
    
    print("\n" + "=" * 70)
    print("TEST 1: Single LLM Decision")
    print("=" * 70)
    
    # Create one LLM player and one random player
    llm_player = create_llm_player(
        Color.RED,
        provider=provider,
        model=model,
        persona=persona
    )
    
    players = [llm_player, RandomPlayer(Color.BLUE)]
    game = Game(players, seed=42)
    
    # Advance game past initial placement
    print("Advancing game to decision point...")
    for _ in range(20):
        if game.state.current_color() == Color.RED:
            break
        game.play_tick()
    
    # Get LLM decision
    print(f"\nGetting LLM decision for {Color.RED.value}...")
    start_time = time.time()
    
    action = llm_player.decide(game, game.playable_actions)
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Decision made in {elapsed:.2f}s")
    print(f"  Action: {action.action_type.value}")
    print(f"  Value: {action.value}")
    
    # Verify action is valid
    assert action in game.playable_actions, "Invalid action returned!"
    print("  ✓ Action is valid")
    
    return True


def test_short_game(provider: str, model: str, max_turns: int = 10, persona: str = "balanced"):
    """Test a short game with LLM players."""
    from catanatron import Game
    from catanatron.players.llm import create_llm_player
    from catanatron.models.player import Color, RandomPlayer
    
    print("\n" + "=" * 70)
    print(f"TEST 2: Short Game ({max_turns} turns)")
    print("=" * 70)
    
    # Create players
    llm_player = create_llm_player(
        Color.RED,
        provider=provider,
        model=model,
        persona=persona
    )
    
    players = [llm_player, RandomPlayer(Color.BLUE)]
    game = Game(players, seed=42)
    
    print(f"Playing {max_turns} turns with LLM player ({Color.RED.value}) vs Random ({Color.BLUE.value})...")
    
    total_llm_time = 0
    llm_decisions = 0
    
    start_time = time.time()
    
    for turn in range(max_turns):
        if game.winning_color() is not None:
            break
        
        current_color = game.state.current_color()
        
        if current_color == Color.RED:
            # Time LLM decisions
            decision_start = time.time()
            game.play_tick()
            total_llm_time += time.time() - decision_start
            llm_decisions += 1
        else:
            game.play_tick()
        
        if (turn + 1) % 5 == 0:
            print(f"  Turn {game.state.num_turns}: {current_color.value} moved")
    
    total_time = time.time() - start_time
    
    print(f"\n✓ Game played for {game.state.num_turns} turns in {total_time:.2f}s")
    if llm_decisions > 0:
        print(f"  LLM decisions: {llm_decisions}")
        print(f"  Avg LLM decision time: {total_llm_time / llm_decisions:.2f}s")
    
    winner = game.winning_color()
    if winner:
        print(f"  Winner: {winner.value}")
    else:
        # Show VP standings
        for player in game.state.players:
            idx = game.state.color_to_index[player.color]
            vp = game.state.player_state[f"P{idx}_ACTUAL_VICTORY_POINTS"]
            print(f"  {player.color.value}: {vp} VP")
    
    return True


def test_two_llm_players(provider: str, model: str, max_turns: int = 20):
    """Test two LLM players against each other."""
    from catanatron import Game
    from catanatron.players.llm import create_llm_player
    from catanatron.models.player import Color
    
    print("\n" + "=" * 70)
    print(f"TEST 3: Two LLM Players ({max_turns} turns)")
    print("=" * 70)
    
    # Create two LLM players with different personas
    players = [
        create_llm_player(
            Color.RED,
            provider=provider,
            model=model,
            persona="balanced"
        ),
        create_llm_player(
            Color.BLUE,
            provider=provider,
            model=model,
            persona="cooperative"
        ),
    ]
    
    game = Game(players, seed=42)
    
    print(f"Playing {max_turns} turns: {Color.RED.value} (balanced) vs {Color.BLUE.value} (cooperative)...")
    
    start_time = time.time()
    decision_times = []
    
    for turn in range(max_turns):
        if game.winning_color() is not None:
            break
        
        current_color = game.state.current_color()
        
        decision_start = time.time()
        game.play_tick()
        decision_time = time.time() - decision_start
        decision_times.append(decision_time)
        
        if (turn + 1) % 5 == 0:
            print(f"  Turn {game.state.num_turns}: {current_color.value} moved ({decision_time:.2f}s)")
    
    total_time = time.time() - start_time
    avg_decision = sum(decision_times) / len(decision_times) if decision_times else 0
    
    print(f"\n✓ Game played for {game.state.num_turns} turns in {total_time:.2f}s")
    print(f"  Total decisions: {len(decision_times)}")
    print(f"  Avg decision time: {avg_decision:.2f}s")
    
    winner = game.winning_color()
    if winner:
        print(f"  Winner: {winner.value}")
    else:
        # Show VP standings
        for player in game.state.players:
            idx = game.state.color_to_index[player.color]
            vp = game.state.player_state[f"P{idx}_ACTUAL_VICTORY_POINTS"]
            print(f"  {player.color.value}: {vp} VP")
    
    return True


def test_strategic_advisor_with_llm(provider: str, model: str):
    """Test that the strategic advisor provides rankings to the LLM."""
    from catanatron import Game
    from catanatron.players.llm import create_llm_player, StrategicAdvisor, StateRenderer
    from catanatron.models.player import Color, RandomPlayer
    
    print("\n" + "=" * 70)
    print("TEST 4: Strategic Advisor Integration")
    print("=" * 70)
    
    # Create a game and advance it
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=42)
    
    # Advance to mid-game
    for _ in range(50):
        if game.winning_color() is not None:
            break
        game.play_tick()
    
    # Get strategic rankings
    advisor = StrategicAdvisor()
    rankings = advisor.rank_actions(game, game.playable_actions, top_n=5)
    
    print(f"\nStrategic rankings for {game.state.current_color().value}:")
    for r in rankings:
        print(f"  #{r.rank}: {r.action.action_type.value} (value: {r.normalized_value:.2f})")
        print(f"       Explanation: {r.explanation[:60]}...")
    
    # Render full context
    renderer = StateRenderer()
    context = renderer.render_full_context(
        game,
        game.state.current_color(),
        rankings=rankings
    )
    
    print(f"\nContext length: {len(context)} characters")
    print("\nContext preview:")
    print("-" * 50)
    print(context[:500] + "...")
    print("-" * 50)
    
    print("\n✓ Strategic advisor integration working")
    return True


def run_all_tests(provider: str, model: str, max_turns: int, persona: str):
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CATANATRON LLM NEGOTIATION - REAL LLM TESTS")
    print("=" * 70)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Persona: {persona}")
    print(f"Max turns: {max_turns}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Single decision
    try:
        results["single_decision"] = test_single_decision(provider, model, persona)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        results["single_decision"] = False
    
    # Test 2: Short game
    try:
        results["short_game"] = test_short_game(provider, model, max_turns, persona)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        results["short_game"] = False
    
    # Test 3: Two LLM players
    try:
        results["two_llm_players"] = test_two_llm_players(provider, model, max_turns)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        results["two_llm_players"] = False
    
    # Test 4: Strategic advisor integration
    try:
        results["strategic_advisor"] = test_strategic_advisor_with_llm(provider, model)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        results["strategic_advisor"] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM negotiation with real LLM API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with Gemini
    python examples/test_real_llm.py
    
    # Test with OpenAI
    python examples/test_real_llm.py --provider openai --model gpt-4
    
    # Quick test (fewer turns)
    python examples/test_real_llm.py --turns 5
    
    # Test with different persona
    python examples/test_real_llm.py --persona competitive
    
    # Run specific test
    python examples/test_real_llm.py --test single_decision
        """
    )
    
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic", "ollama"],
        default="gemini",
        help="LLM provider (default: gemini)"
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: provider-specific)"
    )
    
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Maximum turns to play (default: 10)"
    )
    
    parser.add_argument(
        "--persona",
        choices=["balanced", "competitive", "cooperative", "deceptive", "analytical"],
        default="balanced",
        help="Player persona (default: balanced)"
    )
    
    parser.add_argument(
        "--test",
        choices=["single_decision", "short_game", "two_llm_players", "strategic_advisor", "all"],
        default="all",
        help="Which test to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Set default model based on provider
    if args.model is None:
        model_defaults = {
            "gemini": "gemini-2.5-flash",  # Use flash for speed
            "openai": "gpt-4",
            "anthropic": "claude-3-sonnet-20240229",
            "ollama": "llama3",
        }
        args.model = model_defaults[args.provider]
    
    # Check API key
    if not check_api_key(args.provider):
        sys.exit(1)
    
    # Run tests
    if args.test == "all":
        success = run_all_tests(args.provider, args.model, args.turns, args.persona)
    else:
        try:
            if args.test == "single_decision":
                success = test_single_decision(args.provider, args.model, args.persona)
            elif args.test == "short_game":
                success = test_short_game(args.provider, args.model, args.turns, args.persona)
            elif args.test == "two_llm_players":
                success = test_two_llm_players(args.provider, args.model, args.turns)
            elif args.test == "strategic_advisor":
                success = test_strategic_advisor_with_llm(args.provider, args.model)
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    print("\n" + "=" * 70)
    if success:
        print("All tests completed successfully! ✓")
    else:
        print("Some tests failed. ✗")
    print("=" * 70)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

