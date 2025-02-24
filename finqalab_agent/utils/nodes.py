import os
import pytz
from langgraph.graph import END
from datetime import datetime, UTC
from langgraph.types import Command
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from finqalab_agent.utils.load_once import _get_model
from langchain_core.messages import AIMessage
from typing import List, Annotated, Dict, Any, Literal
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from finqalab_agent.utils.load_once import _get_bm25ret
from finqalab_agent.utils.tools import information_retriever_tool, human_assistance_tool

memory = MemorySaver()

def rewrite_query_node(state) -> Command[Literal["intent_detector","retriever"]]:

    # State Modification (Keeping last 3 Conversations)
    human_indices = []
    for i in range(len(state["messages"]) - 1, -1, -1):
        if state["messages"][i].type == "human":
            human_indices.append(i)
        if len(human_indices) >= 3:
            break
    human_indices = human_indices[::-1]

    if len(human_indices) >= 3:
        start, end = human_indices[0], human_indices[2]
        state["messages"] = [msg for msg in state["messages"][start:end + 1]]
    
    def modify_state_messages(state):
    
        return [("system","""You are a highly skilled AI assistant specialized in rewriting customer queries.
        Follow these steps:
        1. If the customer query is clear and self-contained (e.g., greeting, simple question), return it as it is.
        2. If the customer query is ambiguous and requires historical context to understand, rewrite it.
        3. You must always return either the exact customer query or the rewritten customer query. You are not allowed to output anything else.""")] + state["messages"]

    agent_config = {"recursion_limit": 6}
    rewriter_agent = create_react_agent(model = _get_model('openai', temp = 0),
                                        tools = [],
                                        prompt = modify_state_messages,
                                        checkpointer = memory)
    result = rewriter_agent.invoke(state)

    try:
        rewritten_query = result["messages"][-1].content
        return Command(goto = "intent_detector", update = {"rewritten_query": rewritten_query})

    except Exception as e:
        print("Error generating structured output:", e)
        return Command(goto = "retriever", update = {"rewritten_query": None})


def intent_detection_node(state) -> Command[Literal["greeter","retriever","relevance_checker"]]:

    class Intent(BaseModel):
        """Intent of the User's Query"""
        user_intent: str = Field(description = "Detected Intent")

    rewritten_query = state['rewritten_query']

    if not rewritten_query:
        return Command(goto = "retriever", update = {"user_intent": "Scenario 2"})

    system_prompt = """You are an Intent Detector tasked with identifying the intent of the User."""

    user_prompt = PromptTemplate(
        input_variables=["query",],
        template= """Choose one of the two following scenarios based on the User's Query type:

        1. Scenario 1: For Greetings or Functionality Inquiries
            - If the user greets you or asks about your functionality, the Detected Intent will be 'Scenario 1'.
        
        2. Scenario 2: For all other Queries
            - the Detected Intent will be 'Scenario 2'
        
        Remember to choose a scenario based on the query type and output one of ['Scenario 1', 'Scenario 2'] for the Detected Intent.

        Example 1:
        User's Query: What are red blood cells?
        Detected Intent: Scenario 2

        Example 2:
        User's Query: Hello. Who are you. Can you help me?
        Detected Intent: Scenario 1

        Example 3:
        User's Query: Hello. How can I be exempted from paying Zakat?
        Detected Intent: Scenario 2
            
        User's Query: {query}
        Detected Intent: 
        """
    )

    formatted_user_prompt = user_prompt.format(query = rewritten_query)
    messages = [SystemMessage(content = system_prompt), HumanMessage(content = formatted_user_prompt)]
    
    llm = _get_model('openai', temp = 0)

    try:
        structured_llm = llm.with_structured_output(Intent)
        user_intent = structured_llm.invoke(messages).user_intent
        if user_intent == 'Scenario 1':
            return Command(goto = "greeter", update = {"user_intent": user_intent})
        else:
            return Command(goto = "relevance_checker", update = {"user_intent": user_intent})
        
    except Exception as e:
        print("Error generating structured output:", e)
        return Command(goto = "retriever", update = {"user_intent": "Scenario 2"})


def greeting_node(state) -> Command[Literal["__end__"]]:

    utc_now = datetime.now(UTC)

    pakistan_tz = pytz.timezone('Asia/Karachi')
    pakistan_time = utc_now.astimezone(pakistan_tz)

    hour = pakistan_time.hour
    if 5 <= hour < 12:
        time_of_day = "Morning"
    elif 12 <= hour < 17:
        time_of_day = "Afternoon"
    else:
        time_of_day = "Evening"

    greet_statement = f"Hello and Good {time_of_day}! I'm a Finqalab's Customer Support Agent. How can I assist you today?"

    return Command(update = {"messages": [AIMessage(content = greet_statement)]}, goto = END)


def relevance_check_node(state) -> Command[Literal["irrelevant_queries","retriever"]]:

    class Score(BaseModel):
        """Relevance Score for the User's Query"""
        relevance_score: int = Field(description = "Relevance Score")
    
    rewritten_query = state['rewritten_query']

    if not rewritten_query:
        return Command(goto = "retriever", update = {"relevance_score": 6})

    bm25 = _get_bm25ret(k = 5)
    retrieved_docs = bm25.invoke(rewritten_query)
    context = ''
    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"

    system_prompt = """You are an intelligent assistant for Finqalab tasked with evaluating the relevance of customer queries."""

    user_prompt = PromptTemplate(
        input_variables=["query","context"],
        template= """Finqalab is a multi-asset investment platform in Pakistan that allows customers to invest in stocks, ETFs, and government securities like T-Bills on the Pakistan Stock Exchange. Your task is to evaluate the relevance of customer queries.

        Your task is to evaluate the relevance of customer queries by analyzing the retrieved knowledge base context. If the retrieved context contains a direct answer, assign a high relevance score. If there is no direct answer, assess whether the query is broadly related to Finqalab's services. Assign a score based on the following criteria:

        1. High Relevance (8-10): The query directly matches or closely aligns with the retrieved context, indicating strong relevance.
        2. Moderate Relevance (5-7): The query is not explicitly covered in the retrieved context but is still broadly related to Finqalab, financial markets, investing, or online trading.
        3. Low Relevance (1-4): The query has little to no meaningful connection with Finqalab, investments, or trading.

        Think carefully before choosing a relevance score. If the context does not contain the exact answer but the question is still relevant to Finqalab, assign a moderate score instead of a low one.

        Always output only the relevance score as a number between 1 and 10, with no additional text or explanation.

        Example 1:
        Retrieved Context:
        Question: How can I make zakat non-deductible?  Answer: To make zakat non-deductible, you need to submit a declaration on stamp paper as per regulatory requirements of NCCPL. We can prepare the paperwork for you; however, you will need to sign it and pay an additional fee of PKR 500/- for stamp paper. If you wish, you can initially set zakat as deductible and change it later.
        Question: How to pay through Payfast?  Answer: Payfast allows in-app bank transfers, which means you can transfer money into your Finqalab Account without leaving the app. However, it takes 24 hours to process the payment.
        Customer's Query: How to not pay zakat?
        Relevance Score: 10 (Directly Related)

        Example 2:
        Retrieved Context:
        Question: When do I become eligible for bonus shares?  Answer: To receive bonus shares, you must own the shares on the ex-date.
        Question: When will the shares I bought today reflect in my CDC account?  Answer: It takes two working days for the shares purchased today to reflect in your CDC sub- account.
        Customer's Query: When was Finqalab founded?
        Relevance Score: 7 (Broadly Related)

        Example 3:
        Retrieved Context:
        Question: What payment methods are accepted in the app?  Answer: There are three deposit methods. Manual Bank Transfer, PayFast, and Instant Bank Transfer.
        Question: What are the applicable CGT rates for RDA Account Holders?  Answer: Filer rates are applied to RDA account holders irrespective of their status (Filer or Non-filer).
        Customer's Query: What are red blood cells?
        Relevance Score: 1 (Not Related)
        
        Remember to analyze whether the query is directly answered in the retrieved context. If not, determine if it is still broadly related to Finqalab's services, investing, or trading.

        Retrieved Context: 
        {context}
        Customer's Query: {query}
        Relevance Score:
        """
    )

    formatted_user_prompt = user_prompt.format(query = rewritten_query, context = context)
    messages = [SystemMessage(content = system_prompt), HumanMessage(content = formatted_user_prompt)]
    
    llm = _get_model('openai', temp = 0)

    try:
        structured_llm = llm.with_structured_output(Score)
        relevance_score = structured_llm.invoke(messages).relevance_score
        if relevance_score >= 5:
            return Command(goto = "retriever", update = {"relevance_score": relevance_score})
        else:
            return Command(goto = "irrelevant_queries", update = {"relevance_score": relevance_score})
        
    except Exception as e:
        print("Error generating structured output:", e)
        return Command(goto = "retriever", update = {"relevance_score": 6})


def irrelevant_queries_node(state) -> Command[Literal["__end__"]]:

    irrelevant_handle_statement = "Thank you for reaching out! It seems your query might be unrelated to Finqalab's services or support. If you have any questions related to Finqalab or its services, feel free to ask, and I'll be happy to help!"

    return Command(update = {"messages": [AIMessage(content = irrelevant_handle_statement)]}, goto = END)


def retriever_node(state) -> Command[Literal["__end__"]]:

    def modify_state_messages(state):
        
        return [("system","""You are a dedicated Customer Support Agent for Finqalab with no prior knowledge of the company.
        1. Analyze the Customer's query and rewrite it for clarity and translation into English if needed.
        2. For all queries, its mandatory to first use the `information_retriever_tool` tool to fetch relevant information from the knowledge base. Do not rely on prior knowledge to generate responses.
        3. If no relevant information is found, you should always ask for human assistance by using the `human_assistance_tool` tool.
        4. Answer in your own words without referencing or mentioning the retrieved text explicitly.""")] + state["messages"]

    agent_config = {"recursion_limit": 6}
    retrieval_agent = create_react_agent(model = _get_model('openai', temp = 0),
                                         tools = [information_retriever_tool, human_assistance_tool],
                                         prompt = modify_state_messages,
                                         checkpointer = memory)

    result = retrieval_agent.invoke(state, config = agent_config)
    last_tool = next((msg for msg in reversed(result["messages"]) if msg.type == "tool"), None)

    if last_tool:
        if last_tool.name == 'human_assistance_tool':
            return Command(update = {"messages": [AIMessage(content = last_tool.content)]}, goto = END)
    else:
        return Command(update = {"messages": [result["messages"][-1]]}, goto = END)


# def retrieval_node(state, config):

#     # Define Retrieval Agent
#     def modify_state_messages(state):

#         # Keep last 2 Conversations
#         human_indices = []
#         for i in range(len(state["messages"]) - 1, -1, -1):
#             if state["messages"][i].type == "human":
#                 human_indices.append(i)
#             if len(human_indices) >= 3:
#                 break
#         human_indices = human_indices[::-1]

#         if len(human_indices) > 2:
#             start, end = human_indices[1], human_indices[2]
#             state["messages"] = ([msg for msg in state["messages"][start:end] if msg.type == "human" or (msg.type == "ai" and not msg.tool_calls)] + state["messages"][end:])

#         return [("system","""You are a dedicated Customer Support Agent for Finqalab (Your Identity) with no prior knowledge of the company. You have a tool to access a knowledge base to answer customer queries.
#         1. Analyze the Customer's query and Translate it in English if needed.
#         2. For greetings or inquiries about your functionality, respond professionally without referencing external knowledge. Do not disclose internal processes, tool names, or the existence of any tools. The only thing you can disclose in your identity.
#         3. For all other queries, its mandatory to use the `information_retriever_tool` to fetch relevant information from the knowledge base. Do not rely on prior knowledge to generate responses.
#         4. If no relevant information is found, politely redirect the customer to Finqalab's support via Email (support@finqalab.com.pk).
#         5. If a customer submits the same question again immediately, provide a direct answer from your previous response instead of initiating a new search.
#         6. Answer in your own words without referencing or mentioning the retrieved text explicitly.""")] + state["messages"]

#     retrieval_agent = create_react_agent(model = _get_model('openai', temp = 0), 
#                                          tools = [information_retriever_tool],
#                                          prompt = modify_state_messages,
#                                          checkpointer = memory)
#     result = retrieval_agent.invoke(state, config = config.get('configurable', {}).get("agent_config",{}))
    
#     return {"messages": [result['messages'][-1]]}


# def translation_node(state):

#     class TranslationOutput(BaseModel):
#         """Structured output for translation"""
#         output: str = Field(description = "Finalized Output")

#     last_human = next((msg for msg in reversed(state["messages"]) if msg.type == "human"), None)
#     last_ai = next((msg for msg in reversed(state["messages"]) if msg.type == "ai"), None)

#     # Define Translation Chain
#     routing_prompt = PromptTemplate(
#         input_variables=["query", "response"],
#         template= """You are a language translator. You have been given two pieces of text: User Query and System Response. 
        
#         1. If the User Query is in English (contains only English words), output the System Response as is.
#         2. If the User Query is in Romanized Urdu (contains Urdu words written in Roman script), translate the System Response into Romanized Urdu without losing any information.

#         Always strictly follow these two conditions. Do not output anything other than your finalized output.

#         Example 1:
#         User Query: I dont want to pay zakaat. What should I do?
#         System Response: To avoid zakat deduction, you must submit a declaration on stamp paper. Finqalab can assist with the paperwork for a PKR 500 fee.
#         Finalized Output: To avoid zakat deduction, you must submit a declaration on stamp paper. Finqalab can assist with the paperwork for a PKR 500 fee.

#         Example 2:
#         User Query: Finqalab App ke liye signup kesay karein?
#         System Response: To sign up for the Finqalab app, open the app and select the "Sign Up" option. Enter the required information and click "Continue."
#         Finalized Output: Finqalab app ka sign-up karne ke liye, app ko kholain aur "Sign Up" option select karein. Zaroori maloomat darj karein aur "Continue" par click karein.

#         Now process the following input carefully. Ensure that the Finalized Output is completely free of Hindi, and strictly avoid repeating the User Query:

#         User Query: {query}
#         System Response: {response}
#         Finalized Output:
#         """
#     )

#     structured_llm = _get_model('google', temp = 0).with_structured_output(TranslationOutput)
#     structured_response = structured_llm.invoke(routing_prompt.format(query = last_human.content, response = last_ai.content))
    
#     return {"messages": [AIMessage(structured_response.output)]}