from typing import TypedDict, Dict, Any
from langgraph.graph import END, MessagesState, StateGraph
from finqalab_agent.utils.nodes import retrieval_node, translation_node

# Define the config
class GraphConfig(TypedDict):
    agent_config: Dict[str, Any]

workflow = StateGraph(MessagesState, config_schema = GraphConfig)

workflow.add_node("Retrieval", retrieval_node)
workflow.add_node('Translation', translation_node)

workflow.set_entry_point("Retrieval")

workflow.add_edge("Retrieval", "Translation")
workflow.add_edge("Translation", END)

graph = workflow.compile()