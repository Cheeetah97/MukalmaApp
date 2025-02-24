from langgraph.graph import MessagesState

class GraphState(MessagesState):
    rewritten_query: str
    user_intent: str
    relevance_score: int