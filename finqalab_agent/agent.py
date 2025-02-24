from typing import TypedDict, Dict, Any
from langgraph.pregel import RetryPolicy
from langgraph.graph import END, StateGraph
from finqalab_agent.utils.state import GraphState
from finqalab_agent.utils.nodes import memory, rewrite_query_node, intent_detection_node, greeting_node, relevance_check_node, irrelevant_queries_node, retriever_node

# class GraphConfig(TypedDict):
#     agent_config: Dict[str, Any]

# workflow = StateGraph(GraphState, config_schema = GraphConfig)

# workflow.add_node("Retrieval", retrieval_node, retry = RetryPolicy(max_attempts = 2))
# workflow.add_node('Translation', translation_node)

# workflow.set_entry_point("Retrieval")

# workflow.add_edge("Retrieval", "Translation")
# workflow.add_edge("Translation", END)

# graph = workflow.compile()

workflow = StateGraph(GraphState)

workflow.add_node("query_rewriter", rewrite_query_node, retry = RetryPolicy(max_attempts = 2))
workflow.add_node("intent_detector", intent_detection_node, retry = RetryPolicy(max_attempts = 2))
workflow.add_node("greeter", greeting_node)
workflow.add_node("relevance_checker", relevance_check_node, retry = RetryPolicy(max_attempts = 2))
workflow.add_node("irrelevant_queries", irrelevant_queries_node)
workflow.add_node("retriever", retriever_node, retry = RetryPolicy(max_attempts = 2))

workflow.set_entry_point("query_rewriter")

graph = workflow.compile(checkpointer = memory)