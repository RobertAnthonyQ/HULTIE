from typing import Union, Any, Literal
from langchain_core.messages import AnyMessage
from pydantic import BaseModel

from utils.general_utils import print_colored

def tools_edge_planb(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
):
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.

    """


    tool_node_name = "planb_tools"
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return tool_node_name
    return "__end__"

def tools_end_planb(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
):
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.

    """

    tool_ends = ["send_imagen_presentandose"],

    if isinstance(state, list):
        ai_message = state[-2]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-2]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-2]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        name= ai_message.tool_calls[-1]["name"]
        if ai_message.tool_calls[-1]["name"] in tool_ends:
            print_colored(f"Termino", 33)
            return "__end__"
    return "hultie_planb"
