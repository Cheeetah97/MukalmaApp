import os
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from finqalab_agent.utils.load_once import _get_model
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from finqalab_agent.utils.tools import information_retriever_tool
from langgraph.prebuilt.chat_agent_executor import AgentState

memory = MemorySaver()

def retrieval_node(state, config):

    # Define Retrieval Agent
    def modify_state_messages(state):

        # Keep last 2 Conversations
        human_indices = []
        for i in range(len(state["messages"]) - 1, -1, -1):
            if state["messages"][i].type == "human":
                human_indices.append(i)
            if len(human_indices) >= 3:
                break
        human_indices = human_indices[::-1]

        if len(human_indices) > 2:
            start, end = human_indices[1], human_indices[2]
            state["messages"] = ([msg for msg in state["messages"][start:end] if msg.type == "human" or (msg.type == "ai" and not msg.tool_calls)] + state["messages"][end:])

        return [("system","""You are a dedicated Customer Support Agent for Finqalab (Your Identity) with no prior knowledge of the company. You have a tool to access a knowledge base to answer customer queries.
        1. Analyze the Customer's query and Translate it in English if needed.
        2. For greetings or inquiries about your functionality, respond professionally without referencing external knowledge. Do not disclose internal processes, tool names, or the existence of any tools. The only thing you can disclose in your identity.
        3. For all other queries, its mandatory to use the `information_retriever_tool` to fetch relevant information from the knowledge base. Do not rely on prior knowledge to generate responses.
        4. If no relevant information is found, politely redirect the customer to Finqalab's support via Email (support@finqalab.com.pk).
        5. If a customer submits the same question again immediately, provide a direct answer from your previous response instead of initiating a new search.
        6. Answer in your own words without referencing or mentioning the retrieved text explicitly.""")] + state["messages"]

    retrieval_agent = create_react_agent(model = _get_model('openai', temp = 0), 
                                         tools = [information_retriever_tool],
                                         prompt = modify_state_messages,
                                         checkpointer = memory)
    result = retrieval_agent.invoke(state, config = config.get('configurable', {}).get("agent_config",{}))
    
    return {"messages": [result['messages'][-1]]}


def translation_node(state):

    class TranslationOutput(BaseModel):
        """Structured output for translation"""
        output: str = Field(description = "Finalized Output")

    last_human = next((msg for msg in reversed(state["messages"]) if msg.type == "human"), None)
    last_ai = next((msg for msg in reversed(state["messages"]) if msg.type == "ai"), None)

    # Define Translation Chain
    routing_prompt = PromptTemplate(
        input_variables=["query", "response"],
        template= """You are a language translator. You have been given two pieces of text: User Query and System Response. 
        
        1. If the User Query is in English (contains only English words), output the System Response as is.
        2. If the User Query is in Romanized Urdu (contains Urdu words written in Roman script), translate the System Response into Romanized Urdu without losing any information.

        Always strictly follow these two conditions. Do not output anything other than your finalized output.

        Example 1:
        User Query: I dont want to pay zakaat. What should I do?
        System Response: To avoid zakat deduction, you must submit a declaration on stamp paper. Finqalab can assist with the paperwork for a PKR 500 fee.
        Finalized Output: To avoid zakat deduction, you must submit a declaration on stamp paper. Finqalab can assist with the paperwork for a PKR 500 fee.

        Example 2:
        User Query: Finqalab App ke liye signup kesay karein?
        System Response: To sign up for the Finqalab app, open the app and select the "Sign Up" option. Enter the required information and click "Continue."
        Finalized Output: Finqalab app ka sign-up karne ke liye, app ko kholain aur "Sign Up" option select karein. Zaroori maloomat darj karein aur "Continue" par click karein.

        Now process the following input carefully. Ensure that the Finalized Output is completely free of Hindi, and strictly avoid repeating the User Query:

        User Query: {query}
        System Response: {response}
        Finalized Output:
        """
    )

    structured_llm = _get_model('google', temp = 0).with_structured_output(TranslationOutput)
    structured_response = structured_llm.invoke(routing_prompt.format(query = last_human.content, response = last_ai.content))
    
    return {"messages": [AIMessage(structured_response.output)]}