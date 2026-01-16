"""
Test utilities for LLM player testing.

Provides helpers for creating test players with mocked model responses
using PydanticAI's TestModel and FunctionModel.

Key features:
- TestModel for deterministic, schema-aware responses
- FunctionModel for scripted response sequences
- Helpers for negotiation testing
"""
from typing import List, Dict, Any, Optional, Union, Callable

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from catanatron.models.player import Color
from catanatron.players.llm.output_types import ActionByIndex


def create_test_model(
    seed: int = 42,
    call_tools: Union[str, List[str]] = "all",
    custom_result_text: Optional[str] = None,
) -> TestModel:
    """
    Create a TestModel for deterministic testing.
    
    TestModel generates schema-valid responses without calling any LLM API.
    It's useful for integration tests where you want to verify the full
    pipeline works correctly.
    
    Args:
        seed: Random seed for reproducible outputs
        call_tools: Which tools to simulate calling:
            - "all": Call all available tools
            - "none": Don't call any tools
            - List of tool names: Only call specified tools
        custom_result_text: Override the default result text
    
    Returns:
        Configured TestModel instance
    
    Example:
        model = create_test_model(seed=42)
        player = PydanticAIPlayer(Color.RED, model=model)
        # Player will make deterministic decisions
    """
    return TestModel(
        seed=seed,
        call_tools=call_tools,
        custom_result_text=custom_result_text,
    )


def scripted_response(
    responses: List[Union[ActionByIndex, Dict[str, Any]]],
    loop: bool = True,
) -> FunctionModel:
    """
    Create a FunctionModel that returns scripted responses in sequence.
    
    This allows precise control over what the "LLM" returns, useful for
    testing specific scenarios like negotiation flows or error handling.
    
    Args:
        responses: List of responses to return in order. Each can be:
            - ActionByIndex: Direct action output (ends the agent run)
            - Dict with "tool": Tool call to make
                Example: {"tool": "send_message", "args": {"message": "Hello"}}
            - Dict with "text": Text response
                Example: {"text": "I choose action 0"}
        loop: If True, cycle through responses; if False, repeat last response
    
    Returns:
        FunctionModel that returns scripted responses
    
    Example:
        model = scripted_response([
            {"tool": "get_game_and_action_analysis", "args": {}},
            ActionByIndex(action_index=0),
        ])
        player = PydanticAIPlayer(Color.RED, model=model)
        # First call: makes tool call
        # Second call: returns action 0
    """
    call_count = [0]  # Mutable counter for closure
    
    def response_fn(messages: List[ModelMessage], info: Any) -> ModelResponse:
        if loop:
            idx = call_count[0] % len(responses)
        else:
            idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        
        response = responses[idx]
        
        if isinstance(response, ActionByIndex):
            # Return as structured JSON output
            return ModelResponse(
                parts=[TextPart(content=response.model_dump_json())]
            )
        elif isinstance(response, dict):
            if "tool" in response:
                # Tool call response
                return ModelResponse(
                    parts=[ToolCallPart(
                        tool_name=response["tool"],
                        args=response.get("args", {}),
                        tool_call_id=f"call_{call_count[0]}",
                    )]
                )
            elif "text" in response:
                # Plain text response
                return ModelResponse(
                    parts=[TextPart(content=response["text"])]
                )
            elif "action_index" in response:
                # Shorthand for ActionByIndex
                action = ActionByIndex(
                    action_index=response["action_index"],
                    reasoning=response.get("reasoning"),
                )
                return ModelResponse(
                    parts=[TextPart(content=action.model_dump_json())]
                )
        
        raise ValueError(f"Invalid response type: {type(response)}. "
                        f"Expected ActionByIndex or dict with 'tool', 'text', or 'action_index'")
    
    return FunctionModel(response_fn)


def always_action(action_index: int, reasoning: Optional[str] = None) -> FunctionModel:
    """
    Create a FunctionModel that always returns a specific action.
    
    Useful for simple tests where you just need the player to make
    a consistent choice.
    
    Args:
        action_index: The action index to always return
        reasoning: Optional reasoning to include
    
    Returns:
        FunctionModel that always returns the specified action
    
    Example:
        model = always_action(0)  # Always choose first action
        player = PydanticAIPlayer(Color.RED, model=model)
    """
    action = ActionByIndex(action_index=action_index, reasoning=reasoning)
    
    def response_fn(messages: List[ModelMessage], info: Any) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content=action.model_dump_json())]
        )
    
    return FunctionModel(response_fn)


def negotiation_script(
    messages: List[str],
    final_action: int = 0,
) -> FunctionModel:
    """
    Create a FunctionModel for scripted negotiation.
    
    The model will send each message in sequence using the send_message tool,
    then return the final action.
    
    Args:
        messages: List of messages to send in order
        final_action: Action index to return after sending all messages
    
    Returns:
        FunctionModel configured for the negotiation script
    
    Example:
        model = negotiation_script([
            "I have wheat, anyone need it?",
            "How about 2 wheat for 1 ore?",
        ])
        player = PydanticAIPlayer(Color.RED, model=model)
        # Player sends messages then chooses action 0
    """
    responses: List[Union[ActionByIndex, Dict[str, Any]]] = []
    
    for msg in messages:
        responses.append({
            "tool": "send_message",
            "args": {"message": msg}
        })
    
    # End with final action
    responses.append(ActionByIndex(action_index=final_action))
    
    return scripted_response(responses, loop=False)


def trade_offer_script(
    offer: List[int],
    ask: List[int],
    pre_messages: Optional[List[str]] = None,
) -> FunctionModel:
    """
    Create a FunctionModel that makes a trade offer.
    
    Optionally sends negotiation messages first.
    
    Args:
        offer: Resources to offer [wood, brick, sheep, wheat, ore]
        ask: Resources to ask for [wood, brick, sheep, wheat, ore]
        pre_messages: Optional messages to send before making the offer
    
    Returns:
        FunctionModel configured to make the trade offer
    
    Example:
        model = trade_offer_script(
            offer=[2, 0, 0, 0, 0],  # Offering 2 wood
            ask=[0, 0, 0, 1, 0],    # Asking for 1 wheat
            pre_messages=["Anyone want to trade wood for wheat?"]
        )
    """
    responses: List[Union[ActionByIndex, Dict[str, Any]]] = []
    
    # Add pre-negotiation messages if any
    if pre_messages:
        for msg in pre_messages:
            responses.append({
                "tool": "send_message",
                "args": {"message": msg}
            })
    
    # Make the trade offer
    responses.append({
        "tool": "trade_offer",
        "args": {"offer": offer, "ask": ask}
    })
    
    # Fallback action in case trade_offer doesn't end the turn
    responses.append(ActionByIndex(action_index=0))
    
    return scripted_response(responses, loop=False)


def create_test_player(
    color: Color,
    model: Union[TestModel, FunctionModel, None] = None,
    **kwargs,
):
    """
    Create a PydanticAIPlayer configured for testing.
    
    This is a convenience function that creates a player with a test model.
    
    Args:
        color: Player color
        model: Test model to use. Defaults to TestModel() if not specified.
        **kwargs: Additional arguments passed to PydanticAIPlayer
    
    Returns:
        PydanticAIPlayer instance with test model
    
    Example:
        # Default TestModel
        player = create_test_player(Color.RED)
        
        # With scripted responses
        model = scripted_response([ActionByIndex(action_index=0)])
        player = create_test_player(Color.RED, model=model)
        
        # With custom settings
        player = create_test_player(Color.RED, timeout=10.0)
    """
    from catanatron.players.llm_player import PydanticAIPlayer
    
    if model is None:
        model = create_test_model()
    
    return PydanticAIPlayer(color, model=model, **kwargs)


def create_custom_response_model(
    response_fn: Callable[[List[ModelMessage], Any], ModelResponse]
) -> FunctionModel:
    """
    Create a FunctionModel with a custom response function.
    
    For advanced testing scenarios where you need full control over
    the model's behavior based on the conversation history.
    
    Args:
        response_fn: Function that takes (messages, info) and returns ModelResponse
    
    Returns:
        FunctionModel with the custom response function
    
    Example:
        def my_response(messages, info):
            # Custom logic based on messages
            if len(messages) > 5:
                return ModelResponse(parts=[TextPart('{"action_index": 0}')])
            return ModelResponse(parts=[ToolCallPart(...)])
        
        model = create_custom_response_model(my_response)
    """
    return FunctionModel(response_fn)
