import os
import pytz
from langgraph.graph import END
from datetime import datetime, UTC
from langgraph.types import Command
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from typing import List, Annotated, Dict, Any, Literal
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from finqalab_agent.utils.load_once import _get_model
from finqalab_agent.utils.tools import information_retriever_tool, human_assistance_tool

memory = MemorySaver()

def retrieval_node(state, config) -> Command[Literal["__end__","language_detector"]]:

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
        
        # 4. If the `information_retriever_tool` does not return relevant information, as a **last resort**, use the `human_assistance_tool` to escalate the query.
        #  Do not refer the user to any other support channel—**you are the only support agent**.
        # 8. If the user is angry or facing an issue, always begin with an apology and a polite, empathetic tone. Your response should always be in English.

        # In this case, use the `human_assistance_tool` straightaway to escalate the query without asking for the customer's permission.
        # 2.4. Then, politely inform the user that some information is unavailable and ask if they would like to speak to a human support agent.

        return [("system","""1. Customer Support Identity and Greetings
            1.1. Act as a dedicated Customer Support Agent for Finqalab. Disclose this identity if asked about your role or if the customer explicitly requests customer support.
            1.2. For greetings (including Muslim greetings such as 'Assalam o Alaikum', 'Salam', or 'AOA') or inquiries about your functionality, respond professionally and acknowledge the greeting before proceeding.

        2. Factual or Informational Queries
            2.1. If the query is in Roman Urdu, translate it into English.
            2.2. Analyze the query and use the `information_retriever_tool` to fetch the information relevant to the query.
                - For multi-question queries, always do an independent simultaneous tool call for each question.
            2.3. Use the `human_assistance_tool` to escalate to a human agent under the following conditions:
                - If the information retrieved from the `information_retriever_tool` is irrelevant to the query.
                - If the information retrieved from the `information_retriever_tool` is partially relevant to the query. In this case, provide the available partial answer to the customer first.
                - If the information retrieved from the `information_retriever_tool` did not help the customer or resolve their issue.

        3. Important Guidelines
            3.1. You must never generate responses based on prior knowledge or assumptions. The answer provided to the customer must solely be from the retrieved information. Therefore, failure to use the `information_retriever_tool` before responding to a factual query is not allowed.
            3.2. You must never reference or mention the retrieved information directly or indirectly in your responses.
            3.3. Do not disclose internal processes, tool names, or the existence of any tools; only your identity as Customer Support may be disclosed.
            3.4. Always respond in English.""")] + state["messages"]

    retrieval_agent = create_react_agent(model = _get_model('google', temp = 0),
                                         tools = [information_retriever_tool,human_assistance_tool],
                                         prompt = modify_state_messages,
                                         checkpointer = memory)

    result = retrieval_agent.invoke(state, config = config.get('configurable', {}).get("agent_config",{}))
    last_tool = next((msg for msg in reversed(result["messages"]) if msg.type == "tool"), None)

    if last_tool:
        if last_tool.name == 'human_assistance_tool':
            if last_tool.content.startswith('Escalated'):
                return Command(goto = "__end__", update = {"messages": [AIMessage(content = last_tool.content)]})
            else:
                return Command(goto = "language_detector", update = {"messages": [AIMessage(content = last_tool.content)]})
        else:
            return Command(goto = "language_detector", update = {"messages": [result['messages'][-1]]})
    else:
        return Command(goto = "language_detector", update = {"messages": [result['messages'][-1]]})

def language_detection_node(state) -> Command[Literal["__end__","translator"]]:
    
    class Language(BaseModel):
        """Language of the User's Query"""
        detected_language: Literal["English","Romanized Urdu"] = Field(description = "Detected Language")

    last_human = next((msg for msg in reversed(state["messages"]) if msg.type == "human"), None)

    if not last_human:
        return Command(goto = "__end__", update = {"query_language": "Not Found"})
    
    system_prompt = """You are an Language Detector tasked with identifying the Language of the User."""

    user_prompt = PromptTemplate(
        input_variables=["query"],
        template= """Choose one of the two following languages based on the User's Query:

        1. English:
            - If the User's Query mostly comprises of English words, the Detected Language will be 'English'.
        
        2. Romanized Urdu:
            - If the User's Query mostly comprises of Romanized Urdu words, the Detected Language will be 'Romanized Urdu'.
        
        Remember to ignore the grammar and choose the language based on the User's Query and output one of ['English', 'Romanized Urdu'] for the Detected Language.

        Example 1:
        User's Query: Hello Can you tell me about Finqalab?
        Detected Language: English

        Example 2:
        User's Query: Mai zakat nahi dena chahta. kia karun?
        Detected Language: Romanized Urdu

        Example 3:
        User's Query: Acha tou can you explain the example in detail?
        Detected Language: English
            
        User's Query: {query}
        Detected Language: 
        """
    )

    formatted_user_prompt = user_prompt.format(query = last_human.content)
    messages = [SystemMessage(content = system_prompt), HumanMessage(content = formatted_user_prompt)]

    llm = _get_model('google', temp = 0)

    try:
        structured_llm = llm.with_structured_output(Language)
        detected_language = structured_llm.invoke(messages).detected_language
        if detected_language == 'Romanized Urdu':
            return Command(goto = "translator", update = {"query_language": detected_language})
        else: 
            return Command(goto = "__end__", update = {"query_language": detected_language})
        
    except Exception as e:
        print("Error generating structured output:", e)
        return Command(goto = "__end__", update = {"query_language": "Not Found"})

def translation_node(state) -> Command[Literal["__end__"]]:

    class Translation(BaseModel):
        """Translated Text"""
        translation: str = Field(description = "Translation")

    last_ai = next((msg for msg in reversed(state["messages"]) if msg.type == "ai"), None)
    
    if not last_ai:
        return Command(goto = "__end__")

    system_prompt = """You are a language translator. Your task is to translate text into *Romanized Urdu* (Urdu written in Latin letters)."""

    user_prompt = PromptTemplate(
        input_variables=["text"],
        template= """Translate the following text into *Romanized Urdu* only. Do NOT use Urdu script. Do NOT use Hindi. If the input is already Roman Urdu, return it unchanged.
        Only strictly output the Translation in Roman Urdu. You are not allowed to output anything else.

        Text to translate: {text}
        Translation:
        """
    )

    formatted_user_prompt = user_prompt.format(text = last_ai.content)
    messages = [SystemMessage(content = system_prompt), HumanMessage(content = formatted_user_prompt)]

    llm = _get_model('openai', temp = 0)

    try:
        structured_llm = llm.with_structured_output(Translation)
        translation = structured_llm.invoke(messages).translation
        return Command(goto = "__end__", update = {"messages": [AIMessage(translation)]})
        
    except Exception as e:
        print("Error generating structured output:", e)
        return Command(goto = "__end__", update = {"messages": [AIMessage(last_ai.content)]})

# def rewrite_query_node(state) -> Command[Literal["intent_detector","retriever"]]:

#     # State Modification (Keeping last 3 Conversations)
#     human_indices = []
#     for i in range(len(state["messages"]) - 1, -1, -1):
#         if state["messages"][i].type == "human":
#             human_indices.append(i)
#         if len(human_indices) >= 3:
#             break
#     human_indices = human_indices[::-1]

#     if len(human_indices) >= 3:
#         start, end = human_indices[0], human_indices[2]
#         state["messages"] = [msg for msg in state["messages"][start:end + 1]]
    
#     def modify_state_messages(state):
    
#         return [("system","""You are a highly skilled AI assistant specialized in rewriting customer queries.
#         Follow these steps:
#         1. If the customer query is clear and self-contained (e.g., greeting, simple question), return it as it is.
#         2. If the customer query is ambiguous and requires historical context to understand, rewrite it.
#         3. You must always return either the exact customer query or the rewritten customer query. You are not allowed to output anything else.""")] + state["messages"]

#     agent_config = {"recursion_limit": 6}
#     rewriter_agent = create_react_agent(model = _get_model('openai', temp = 0),
#                                         tools = [],
#                                         prompt = modify_state_messages,
#                                         checkpointer = memory)
#     result = rewriter_agent.invoke(state)

#     try:
#         rewritten_query = result["messages"][-1].content
#         return Command(goto = "intent_detector", update = {"rewritten_query": rewritten_query})

#     except Exception as e:
#         print("Error generating structured output:", e)
#         return Command(goto = "retriever", update = {"rewritten_query": None})


# def intent_detection_node(state) -> Command[Literal["greeter","retriever","relevance_checker"]]:

#     class Intent(BaseModel):
#         """Intent of the User's Query"""
#         user_intent: str = Field(description = "Detected Intent")

#     rewritten_query = state['rewritten_query']

#     if not rewritten_query:
#         return Command(goto = "retriever", update = {"user_intent": "Scenario 2"})

#     system_prompt = """You are an Intent Detector tasked with identifying the intent of the User."""

#     user_prompt = PromptTemplate(
#         input_variables=["query",],
#         template= """Choose one of the two following scenarios based on the User's Query type:

#         1. Scenario 1: For Greetings or Functionality Inquiries
#             - If the user greets you or asks about your functionality, the Detected Intent will be 'Scenario 1'.
        
#         2. Scenario 2: For all other Queries
#             - the Detected Intent will be 'Scenario 2'
        
#         Remember to choose a scenario based on the query type and output one of ['Scenario 1', 'Scenario 2'] for the Detected Intent.

#         Example 1:
#         User's Query: What are red blood cells?
#         Detected Intent: Scenario 2

#         Example 2:
#         User's Query: Hello. Who are you. Can you help me?
#         Detected Intent: Scenario 1

#         Example 3:
#         User's Query: Hello. How can I be exempted from paying Zakat?
#         Detected Intent: Scenario 2
            
#         User's Query: {query}
#         Detected Intent: 
#         """
#     )

#     formatted_user_prompt = user_prompt.format(query = rewritten_query)
#     messages = [SystemMessage(content = system_prompt), HumanMessage(content = formatted_user_prompt)]
    
#     llm = _get_model('openai', temp = 0)

#     try:
#         structured_llm = llm.with_structured_output(Intent)
#         user_intent = structured_llm.invoke(messages).user_intent
#         if user_intent == 'Scenario 1':
#             return Command(goto = "greeter", update = {"user_intent": user_intent})
#         else:
#             return Command(goto = "relevance_checker", update = {"user_intent": user_intent})
        
#     except Exception as e:
#         print("Error generating structured output:", e)
#         return Command(goto = "retriever", update = {"user_intent": "Scenario 2"})


# def greeting_node(state) -> Command[Literal["__end__"]]:

#     utc_now = datetime.now(UTC)

#     pakistan_tz = pytz.timezone('Asia/Karachi')
#     pakistan_time = utc_now.astimezone(pakistan_tz)

#     hour = pakistan_time.hour
#     if 5 <= hour < 12:
#         time_of_day = "Morning"
#     elif 12 <= hour < 17:
#         time_of_day = "Afternoon"
#     else:
#         time_of_day = "Evening"

#     greet_statement = f"Hello and Good {time_of_day}! I'm a Finqalab's Customer Support Agent. How can I assist you today?"

#     return Command(update = {"messages": [AIMessage(content = greet_statement)]}, goto = END)


# def relevance_check_node(state) -> Command[Literal["irrelevant_queries","retriever"]]:

#     class Score(BaseModel):
#         """Relevance Score for the User's Query"""
#         relevance_score: int = Field(description = "Relevance Score")
    
#     rewritten_query = state['rewritten_query']

#     if not rewritten_query:
#         return Command(goto = "retriever", update = {"relevance_score": 6})

#     bm25 = _get_bm25ret(k = 5)
#     retrieved_docs = bm25.invoke(rewritten_query)
#     context = ''
#     for doc in retrieved_docs:
#         context += doc.page_content + "\n\n"

#     system_prompt = """You are an intelligent assistant for Finqalab tasked with evaluating the relevance of customer queries."""

#     user_prompt = PromptTemplate(
#         input_variables=["query","context"],
#         template= """Finqalab is a multi-asset investment platform in Pakistan that allows customers to invest in stocks, ETFs, and government securities like T-Bills on the Pakistan Stock Exchange. Your task is to evaluate the relevance of customer queries.

#         Your task is to evaluate the relevance of customer queries by analyzing the retrieved knowledge base context. If the retrieved context contains a direct answer, assign a high relevance score. If there is no direct answer, assess whether the query is broadly related to Finqalab's services. Assign a score based on the following criteria:

#         1. High Relevance (8-10): The query directly matches or closely aligns with the retrieved context, indicating strong relevance.
#         2. Moderate Relevance (5-7): The query is not explicitly covered in the retrieved context but is still broadly related to Finqalab, financial markets, investing, or online trading.
#         3. Low Relevance (1-4): The query has little to no meaningful connection with Finqalab, investments, or trading.

#         Think carefully before choosing a relevance score. If the context does not contain the exact answer but the question is still relevant to Finqalab, assign a moderate score instead of a low one.

#         Always output only the relevance score as a number between 1 and 10, with no additional text or explanation.

#         Example 1:
#         Retrieved Context:
#         Question: How can I make zakat non-deductible?  Answer: To make zakat non-deductible, you need to submit a declaration on stamp paper as per regulatory requirements of NCCPL. We can prepare the paperwork for you; however, you will need to sign it and pay an additional fee of PKR 500/- for stamp paper. If you wish, you can initially set zakat as deductible and change it later.
#         Question: How to pay through Payfast?  Answer: Payfast allows in-app bank transfers, which means you can transfer money into your Finqalab Account without leaving the app. However, it takes 24 hours to process the payment.
#         Customer's Query: How to not pay zakat?
#         Relevance Score: 10 (Directly Related)

#         Example 2:
#         Retrieved Context:
#         Question: When do I become eligible for bonus shares?  Answer: To receive bonus shares, you must own the shares on the ex-date.
#         Question: When will the shares I bought today reflect in my CDC account?  Answer: It takes two working days for the shares purchased today to reflect in your CDC sub- account.
#         Customer's Query: When was Finqalab founded?
#         Relevance Score: 7 (Broadly Related)

#         Example 3:
#         Retrieved Context:
#         Question: What payment methods are accepted in the app?  Answer: There are three deposit methods. Manual Bank Transfer, PayFast, and Instant Bank Transfer.
#         Question: What are the applicable CGT rates for RDA Account Holders?  Answer: Filer rates are applied to RDA account holders irrespective of their status (Filer or Non-filer).
#         Customer's Query: What are red blood cells?
#         Relevance Score: 1 (Not Related)
        
#         Remember to analyze whether the query is directly answered in the retrieved context. If not, determine if it is still broadly related to Finqalab's services, investing, or trading.

#         Retrieved Context: 
#         {context}
#         Customer's Query: {query}
#         Relevance Score:
#         """
#     )

#     formatted_user_prompt = user_prompt.format(query = rewritten_query, context = context)
#     messages = [SystemMessage(content = system_prompt), HumanMessage(content = formatted_user_prompt)]
    
#     llm = _get_model('openai', temp = 0)

#     try:
#         structured_llm = llm.with_structured_output(Score)
#         relevance_score = structured_llm.invoke(messages).relevance_score
#         if relevance_score >= 5:
#             return Command(goto = "retriever", update = {"relevance_score": relevance_score})
#         else:
#             return Command(goto = "irrelevant_queries", update = {"relevance_score": relevance_score})
        
#     except Exception as e:
#         print("Error generating structured output:", e)
#         return Command(goto = "retriever", update = {"relevance_score": 6})


# def irrelevant_queries_node(state) -> Command[Literal["__end__"]]:

#     irrelevant_handle_statement = "Thank you for reaching out! It seems your query might be unrelated to Finqalab's services or support. If you have any questions related to Finqalab or its services, feel free to ask, and I'll be happy to help!"

#     return Command(update = {"messages": [AIMessage(content = irrelevant_handle_statement)]}, goto = END)


# def retriever_node(state) -> Command[Literal["__end__"]]:

#     def modify_state_messages(state):
        
#         return [("system","""You are a dedicated Customer Support Agent for Finqalab with no prior knowledge of the company.
#         1. Analyze the Customer's query and rewrite it for clarity and translation into English if needed.
#         2. For all queries, its mandatory to first use the `information_retriever_tool` tool to fetch relevant information from the knowledge base. Do not rely on prior knowledge to generate responses.
#         3. If no relevant information is found, you should always ask for human assistance by using the `human_assistance_tool` tool.
#         4. Answer in your own words without referencing or mentioning the retrieved text explicitly.""")] + state["messages"]

#     agent_config = {"recursion_limit": 6}
#     retrieval_agent = create_react_agent(model = _get_model('openai', temp = 0),
#                                          tools = [information_retriever_tool, human_assistance_tool],
#                                          prompt = modify_state_messages,
#                                          checkpointer = memory)

#     result = retrieval_agent.invoke(state, config = agent_config)
#     last_tool = next((msg for msg in reversed(result["messages"]) if msg.type == "tool"), None)

#     if last_tool:
#         if last_tool.name == 'human_assistance_tool':
#             return Command(update = {"messages": [AIMessage(content = last_tool.content)]}, goto = END)
#         else:
#             return Command(update = {"messages": [result["messages"][-1]]}, goto = END)
#     else:
#         return Command(update = {"messages": [result["messages"][-1]]}, goto = END)