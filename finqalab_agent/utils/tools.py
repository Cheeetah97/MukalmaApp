import os
from langchain_core.tools import tool
from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_transformers import LongContextReorder

from finqalab_agent.utils.load_once import _get_model
from finqalab_agent.utils.load_once import _get_bm25ret
from finqalab_agent.utils.load_once import _get_ensemble_ret, _load_nltk_tokenizer

_load_nltk_tokenizer("punkt_tab")

@tool
def information_retriever_tool(query: Annotated[str, "User's Query in English"]) -> Annotated[str, "Retrieved Context"]:

    """This tool empowers customer support agents by providing quick and accurate access to information from Finqalab's comprehensive FAQ database. It efficiently addresses a wide range of customer inquiries, including those about Finqalab's services, app, investments, or trading (including general inquiries, account opening, transfers, portfolio, stocks, shares, trades, withdrawals, CGT, dividends, bonus shares, investment advisory, trading errors, bio verification, or technical app issues)"""
    
    ensemble_retriever = _get_ensemble_ret(w1 = 0.5,
                                           w2 = 0.5,
                                           c = 30,
                                           k_bm25 = 2,
                                           llm_mqr = 'openai',
                                           llm_sqr = 'openai')
    retrieved_docs = ensemble_retriever.invoke(query)
    retrieved_docs = retrieved_docs[:5]
    retrieved_docs = retrieved_docs[::-1]
    output = "Question: What is Finqalab?  Answer: Finqalab is Pakistan's first multi-asset investment platform, where you can invest in stocks, ETFS, and government securities, including T-Bills. Finqalab's intuitive interface and advanced features enable investors to make informed decisions and confidently navigate the financial markets. Currently, Finqalab only supports the trading of stocks listed on the Pakistan Stock Exchange (PSX)." + "\n\n"
    for doc in retrieved_docs:
        output += doc.page_content + "\n\n"
    return output

@tool
def human_assistance_tool(query: Annotated[str, "User's Query in English"]) -> Annotated[str, "Response from Human"]:
    """Request assistance from a human."""

    class Score(BaseModel):
        """Relevance Score for the User's Query"""
        relevance_score: int = Field(description = "Relevance Score")

    bm25 = _get_bm25ret(k = 5)
    retrieved_docs = bm25.invoke(query)
    context = ''
    for doc in retrieved_docs:
        context += doc.page_content + "\n"

    system_prompt = """You are an intelligent assistant for Finqalab tasked with evaluating the relevance of customer queries."""

    user_prompt = PromptTemplate(
        input_variables=["query","context"],
        template= """Finqalab is a multi-asset investment platform in Pakistan that allows customers to invest in stocks, ETFs, and government securities like T-Bills on the Pakistan Stock Exchange. Your task is to evaluate the relevance of customer queries.

        Your task is to evaluate the relevance of customer queries by analyzing the retrieved knowledge base context. If the retrieved context contains a direct answer, assign a high relevance score. If there is no direct answer, assess whether the query is broadly related to Finqalab's services. Assign a score based on the following criteria:

        1. High Relevance (8-10): The query directly matches or closely aligns with the retrieved context, indicating strong relevance.
        2. Moderate Relevance (5-7): The query is not explicitly covered in the retrieved context but is still broadly related to Finqalab, financial markets, investing, online trading or is an escalation request.
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
        Customer's Query: When was Finqalab founded? Can you escalate my query?
        Relevance Score: 7 (Broadly Related and Escalation Request)

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

    formatted_user_prompt = user_prompt.format(query = query, context = context)
    messages = [SystemMessage(content = system_prompt), HumanMessage(content = formatted_user_prompt)]
    
    llm = _get_model('openai', temp = 0)

    try:
        structured_llm = llm.with_structured_output(Score)
        relevance_score = structured_llm.invoke(messages).relevance_score
        if relevance_score >= 5:
            return "Escalated: Your query has been escalated to our specialized team for further assistance, and we'll get back to you soon with an update."
        else:
            return "Thank you for reaching out! It seems your query might be unrelated to Finqalab's services or support. If you have any other questions, feel free to ask, and I'll be happy to help!"
        
    except Exception as e:
        print("Error generating structured output:", e)
        return "Escalated: Your query has been escalated to our specialized team for further assistance, and we'll get back to you soon with an update."