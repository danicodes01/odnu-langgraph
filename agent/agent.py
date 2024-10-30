from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.utils.nodes import (
    call_model,
    should_continue,
    tool_node,
)
from agent.utils.state import AgentState

class GraphConfig(TypedDict):
    model_name: Literal["openai"]
    thread_id: str

# Define workflow
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Add nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Conditional edges based on `should_continue`
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue_tools": "action",
        "end": END,
    },
)

# Define tool edges
workflow.add_edge("action", "agent")

# Compile the workflow with memory persistence
memory_saver = MemorySaver()
graph = workflow.compile(checkpointer=memory_saver)
