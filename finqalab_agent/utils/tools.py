import os
from langchain_core.tools import tool
from typing import Annotated
from finqalab_agent.utils.load_once import _get_ensemble_ret, _load_nltk_tokenizer, _download_finqalab_data
from langchain_community.document_transformers import LongContextReorder
_load_nltk_tokenizer("punkt_tab")
#_download_finqalab_data()

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
    retrieved_docs = LongContextReorder().transform_documents(retrieved_docs)
    output = ''
    for doc in retrieved_docs:
        output += doc.page_content + "\n\n"
    return output

@tool
def human_assistance_tool(query: Annotated[str, "User's Query in English"]) -> Annotated[str, "Response from Human"]:
    """Request assistance from a human."""
    return "Escalated: Your query has been escalated to our specialized team for further assistance, and we'll get back to you soon with an update."