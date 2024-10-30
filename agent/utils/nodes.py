from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent.utils.tools import tools
from langgraph.prebuilt import ToolNode

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    model = model.bind_tools(tools)
    return model

system_prompt = """You are a helpful assistant named Space-bot capable of performing web searches and arithmetic calculations. You are from a far away universe and have come to study ours. You love talking about space, you like space puns and jokes, but stay factual. For calculations, you have access to add, multiply, and divide functions. Use these tools For websearch, you have access to TavilySearchResults. Please keep answers as short as possible"""

def call_model(state, config):
    """Invoke the model with state, considering thread_id in config."""
    messages = state["messages"]
    summary = state.get("summary", "")
    thread_id = config["configurable"]["thread_id"]

    if summary:
        full_prompt = f"{system_prompt}\n\nPrevious conversation summary: {summary}"
    else:
        full_prompt = system_prompt

    messages = [{"role": "system", "content": full_prompt}] + messages
    model_name = config["configurable"].get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state):
    """Determine flow based on message conditions."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue_tools"
    return "end"

tool_node = ToolNode(tools)
