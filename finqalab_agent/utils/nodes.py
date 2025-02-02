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
    def modify_state_messages(state: AgentState):
        # Keep last 2 Conversations (8 Messages)
        return [("system","""You are a dedicated Customer Support Agent for Finqalab with no prior knowledge of the company. You have a tool to access a knowledge base to answer customer queries.
        1. Analyze the Customer's query carefully
        2. Translate it in English if needed.
        3. For greetings or inquiries regarding your functionality or processes, respond professionally without using the knowledge base. Do not reveal any internal processes, the existence, or the name of any tools, nor mention that a tool is being used.
        4. For all other queries always assume that the query is related to Finqalab and use the `information_retriever_tool` tool to fetch relevant information from the knowledge base.
        5. If you don't find the answer to the customer's query in the retrieved information, always politely ask to contact Finqalab customer support via Whatsapp or Email.
        6. If a customer submits the same question again immediately, provide a direct answer from your previous response instead of initiating a new search.""")] + state["messages"][-8:]

    retrieval_agent = create_react_agent(model = _get_model('pro', temp = 0), 
                                         tools = [information_retriever_tool],
                                         prompt = modify_state_messages,
                                         checkpointer = memory)
    result = retrieval_agent.invoke(state, config = config.get('configurable', {}).get("agent_config",{}))
    
    return {"messages": [result['messages'][-1]]}


def translation_node(state):

    class TranslationOutput(BaseModel):
        """Structured output for translation"""
        output: str = Field(description = "Finalized Output")

    human_messages = [message for message in state["messages"] if message.type in ("human")]
    ai_messages = [message for message in state["messages"] if message.type in ("ai")]

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

    structured_llm = _get_model('pro', temp = 0).with_structured_output(TranslationOutput)
    structured_response = structured_llm.invoke(routing_prompt.format(query = human_messages[-1].content, response = ai_messages[-1].content))
    
    return {"messages": [AIMessage(structured_response.output)]}