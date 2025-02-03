import os
from langchain_core.tools import tool
from typing import Annotated
from finqalab_agent.utils.load_once import _get_ensemble_ret, _load_nltk_tokenizer, _download_finqalab_data
from langchain_community.document_transformers import LongContextReorder
_load_nltk_tokenizer("punkt_tab")
_download_finqalab_data()

def information_retriever_tool(query: Annotated[str, "User's Query in English"]) -> Annotated[str, "Retrieved Context"]:

    """This tool empowers customer support agents by providing quick and accurate access to information from Finqalab's comprehensive FAQ database. It efficiently addresses a wide range of customer inquiries, including those about Finqalab's services, its mobile app, and common trading concepts."""
    
    ensemble_retriever = _get_ensemble_ret(w1 = 0.5,
                                           w2 = 0.5,
                                           c = 30,
                                           k_bm25 = 2,
                                           llm_mqr = 'google',
                                           llm_sqr = 'google')
    retrieved_docs = ensemble_retriever.invoke(query)
    retrieved_docs = retrieved_docs[:5]
    retrieved_docs = LongContextReorder().transform_documents(retrieved_docs)
    output = ''
    for doc in retrieved_docs:
        output += doc.page_content + "\n\n"
    return output