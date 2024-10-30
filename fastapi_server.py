from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel
from typing import Dict, AsyncGenerator
import uuid
import json
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from agent.agent import graph  # Import the pre-defined agent graph
from agent.utils.state import AgentState  # Import the AgentState class

app = FastAPI()

# In-memory store for conversation states
memory_store: Dict[str, AgentState] = {}

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

class QuestionRequest(BaseModel):
    question: str
    thread_id: str = None  # Optional, generated if not provided

def serialize_message(message):
    """Convert AIMessage or HumanMessage to a JSON-serializable format."""
    if isinstance(message, AIMessage):
        return {"type": "ai", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"type": "user", "content": message.content}
    return {"type": "unknown", "content": str(message)}

async def event_stream(state: dict, config: dict) -> AsyncGenerator[str, None]:
    """Yields each output as an SSE message for the EventSource client, filtering out any tool metadata."""
    try:
        # Stream each response from the graph
        for output in graph.stream(state, config):
            # Log the output to help with debugging
            print("Output from graph:", output)

            # Check for 'agent' output and process assistant messages
            if "agent" in output:
                ai_messages = output["agent"].get("messages", [])
                serialized_output = [
                    serialize_message(msg) for msg in ai_messages if isinstance(msg, AIMessage)
                ]
                content = serialized_output[0]["content"] if serialized_output else ""

                # Log the content to track what’s being processed
                print("Serialized content to stream:", content)

                # Send content in smaller chunks for streaming effect
                chunk_size = 20
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size]
                    yield f"data: {json.dumps({'message': chunk})}\n\n"
                    await asyncio.sleep(0.1)  # Adjust for pacing

        yield "data: [DONE]\n\n"  # Signal end of stream
    except Exception as e:
        print("Error in event_stream:", e)  # Log the error
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/generate")
async def generate_route(request: QuestionRequest):
    # Generate or retrieve the thread_id
    thread_id = request.thread_id or str(uuid.uuid4())
    print(f"Received thread_id: {thread_id}")

    # Retrieve or initialize the state for this thread_id
    state = memory_store.get(thread_id, AgentState(messages=[], summary=""))
    state["messages"].append(HumanMessage(content=request.question))  # Add the user question to conversation

    # Configuration with the required 'thread_id'
    config = {"configurable": {"thread_id": thread_id}}

    # Store the updated state
    memory_store[thread_id] = state

    # Return a StreamingResponse with the event stream
    return StreamingResponse(
        event_stream(state, config),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)









# # File: /Users/danielknowles/Dev/langchain/mylang/fastapi_server.py

# import asyncio
# from fastapi import FastAPI, Request, Response
# from fastapi.middleware.cors import CORSMiddleware
# from sse_starlette.sse import EventSourceResponse
# import json
# from langchain_core.messages import HumanMessage
# from typing import Dict
# import uuid
# from agent.agent import graph
# from agent.utils.state import AgentState

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# memory_store: Dict[str, AgentState] = {}

# @app.post("/generate")
# async def chat_with_memory(request: Request):
#     data = await request.json()
#     prompt = data.get("prompt", "")
#     # Retrieve or generate thread_id
#     thread_id = data.get("thread_id", str(uuid.uuid4())) 
#     print(f"Received thread_id: {thread_id}")  # Log the thread_id

#     # Retrieve or initialize the state for this thread_id
#     state = memory_store.get(thread_id, AgentState(messages=[], summary=""))
#     state["messages"].append(HumanMessage(content=prompt))  # Add the prompt to the conversation messages

#     # Process the state through the LangGraph workflow
#     response = graph.invoke(state)
#     memory_store[thread_id] = response
#     output_messages = response.get("messages", [])
#     response_text = output_messages[-1].content if output_messages else "No response generated."

#     return {"response": response_text, "thread_id": thread_id}

# @app.post("/stream_chat")
# async def stream_chat(request: Request):
#     data = await request.json()
#     prompt = data.get("prompt", "")
#     thread_id = data.get("thread_id", str(uuid.uuid4()))

#     state = memory_store.get(thread_id, AgentState(messages=[], summary=""))
#     state["messages"].append(HumanMessage(content=prompt))

#     async def event_generator():
#         seen_messages = set()  # Track unique messages to avoid repetitions
#         try:
#             response = graph.invoke(state)
#             for message in response["messages"]:
#                 if message.id not in seen_messages:
#                     yield f"data: {json.dumps({'type': 'message', 'content': message.content})}\n\n"
#                     seen_messages.add(message.id)  # Mark message as seen
#             yield "data: [DONE]\n\n"
#             memory_store[thread_id] = response  # Save updated state
#         except Exception as e:
#             print(f"Error streaming chat response: {e}")
#             yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"

#     return EventSourceResponse(event_generator())







# THIS WORKS

# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse, RedirectResponse
# from pydantic import BaseModel
# from langchain_core.messages import HumanMessage, AIMessage
# from agent.utils.state import AgentState
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import StateGraph, MessagesState, START
# from agent.utils.nodes import call_model
# import uuid
# from typing import AsyncGenerator
# import json
# import asyncio

# app = FastAPI()

# # Initialize MemorySaver for thread-level persistence
# memory_saver = MemorySaver()

# # Define the graph builder and nodes
# builder = StateGraph(MessagesState)
# builder.add_node("call_model", call_model)
# builder.add_edge(START, "call_model")

# # Compile the graph with the memory saver
# graph = builder.compile(checkpointer=memory_saver)

# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")

# class QuestionRequest(BaseModel):
#     question: str
#     thread_id: str = None  # Optional, generated if not provided

# def serialize_message(message):
#     """Convert AIMessage or HumanMessage to a JSON-serializable format."""
#     if isinstance(message, AIMessage):
#         return {"type": "ai", "content": message.content}
#     elif isinstance(message, HumanMessage):
#         return {"type": "user", "content": message.content}
#     return {"type": "unknown", "content": str(message)}

# async def event_stream(state: dict, config: dict) -> AsyncGenerator[str, None]:
#     """Yields each output as an SSE message for the EventSource client, in smaller chunks."""
#     try:
#         # Stream each response from the graph
#         for output in graph.stream(state, config):
#             if "call_model" in output:
#                 ai_messages = output["call_model"].get("messages", [])
#                 serialized_output = [serialize_message(msg) for msg in ai_messages]
#                 content = serialized_output[0]["content"] if serialized_output else ""
                
#                 # Send content in smaller chunks
#                 chunk_size = 20  # You can adjust this size as needed for chunking effect
#                 for i in range(0, len(content), chunk_size):
#                     chunk = content[i:i+chunk_size]
#                     yield f"data: {json.dumps({'message': chunk})}\n\n"
#                     await asyncio.sleep(0.1)  # Adjust for pacing

#         yield "data: [DONE]\n\n"  # Signal end of stream
#     except Exception as e:
#         yield f"data: {json.dumps({'error': str(e)})}\n\n"

# @app.post("/generate")
# async def generate_route(request: QuestionRequest):
#     # Generate or retrieve the thread_id
#     thread_id = request.thread_id or str(uuid.uuid4())
#     print(f"Received thread_id: {thread_id}")

#     # Prepare the initial message with the user's question
#     input_message = {"type": "user", "content": request.question}

#     # Configuration with thread_id to ensure persistence within the same conversation
#     config = {"configurable": {"thread_id": thread_id}}

#     # Return a StreamingResponse with the event stream
#     return StreamingResponse(
#         event_stream({"messages": [input_message]}, config), 
#         media_type="text/event-stream"
#     )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)
