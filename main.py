import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import AsyncGenerator, Dict, Any, List
from finqalab_agent.agent import graph
from finqalab_agent.utils.nodes import memory

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Finqalab"

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class DeleteThreadsRequest(BaseModel):
    thread_ids: List[str]

async def agent_streamer(thread_id: str, message: str) -> AsyncGenerator[str, None]:

    config = {"configurable": {"agent_config": {"configurable": {"thread_id": thread_id},"recursion_limit": 6}},"recursion_limit": 6}
    
    try:
        async for event in graph.astream(
            {"messages": [{"role": "user", "content": message}]},
            config = config,
            stream_mode = "values"
        ):
            latest_message = event["messages"][-1]
        yield latest_message.content

    except Exception as e:
        yield f"Error occurred: {str(e)}"

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    return EventSourceResponse(
        agent_streamer(request.thread_id, request.message),
        media_type = "text/event-stream"
    )

@app.post("/delete-threads")
async def delete_threads(request: DeleteThreadsRequest):
    deleted = []
    not_found = []
    for thread_id in request.thread_ids:
        if thread_id in memory.storage:
            del memory.storage[thread_id]
            deleted.append(thread_id)
        else:
            not_found.append(thread_id)
    return {"deleted": deleted, "not_found": not_found}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port = 8000)