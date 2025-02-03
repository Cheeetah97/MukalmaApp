import os
import nltk
import json
import requests
from getpass import getpass
from typing import List
from dotenv import load_dotenv
from functools import lru_cache
from nltk.tokenize import word_tokenize
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import JSONLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.query_constructor.base import StructuredQueryOutputParser, get_query_constructor_prompt

load_dotenv()

@lru_cache(maxsize = 2)
def _get_model(name: str, temp: str):
    
    if name == 'google':
        model =  ChatGoogleGenerativeAI(api_key = os.getenv("GOOGLE_PRO_API_KEY"),
                                        temperature = temp,
                                        max_retries = 2,
                                        max_tokens = 750,
                                        model = "gemini-1.5-pro")
    elif name == 'openai':
        model = ChatOpenAI(model = 'gpt-4o-mini', 
                           openai_api_key = os.getenv("OPENAI_API_KEY"), 
                           max_completion_tokens = 1024,
                           max_retries = 2)
    else:
        raise ValueError(f"Unsupported Model Name: {name}")

    return model


@lru_cache(maxsize = 1)
def _get_vector_store(embed_type: str):

    if embed_type == 'hf':
        pass
        # embed_model = HuggingFaceEmbeddings(model_name = "Ch333tah/modernbert-finqalab-embeddings",
        #                                     model_kwargs = {'device': 'cpu'},
        #                                     cache_folder = './hf_cache'
        # )

        # vector_store = QdrantVectorStore.from_existing_collection(collection_name = "qa_collection",
        #                                                           embedding = embed_model,
        #                                                           url = os.getenv("QDRANT"), 
        #                                                           api_key = os.getenv("QDRANT_API_KEY")
        # )
    
    elif embed_type == 'google':

        embed_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", 
                                                   google_api_key = os.getenv("GOOGLE_PRO_API_KEY"))
        
        vector_store = QdrantVectorStore.from_existing_collection(collection_name = "qa_collection_google",
                                                                  embedding = embed_model,
                                                                  url = os.getenv("QDRANT"), 
                                                                  api_key = os.getenv("QDRANT_API_KEY")
        )
    
    else:
        raise ValueError(f"Unsupported Embedding Model type: {embed_type}")

    return vector_store


@lru_cache(maxsize = 1)
def _load_nltk_tokenizer(tok_name: str):

    nltk.download(tok_name)


@lru_cache(maxsize = 1)
def _download_finqalab_data():

    files = [f'Parsed_Data_FAQs_{i}.json' for i in range(1,16)]

    for file_name in files:
        url = f"https://raw.githubusercontent.com/Cheeetah97/MukalmaApp/refs/heads/main/finqalab_data/{file_name}"
        headers = {"Authorization": f"token {os.getenv("GITHUB_TOKEN")}"}
        data = requests.get(url, headers=headers).json()
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)


@lru_cache(maxsize = 1)
def _get_mqr(llm: str, k_bm25: int):

    files = [f'Parsed_Data_FAQs_{i}.json' for i in range(1,16)]

    all_documents = []

    for file_name in files:
        loader = JSONLoader(
            file_path = file_name,
            jq_schema = '.[] | {text}',
            content_key='text',
            text_content = False
        )
        documents = loader.load()

        with open(file_name, 'r') as f:
            data = json.load(f)
            data_source = data[0]['FAQ_Category']
            for doc in documents:
                doc.metadata['FAQ_Category'] = data_source

        all_documents += documents
    
    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""
    
        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return list(filter(None, lines))
    
    QUERY_PROMPT = PromptTemplate(input_variables=["question"],
                                  template = """You are an AI language model assistant for Finqalab. Your task is to generate two different versions of the customer's question to retrieve relevant QnA pairs from a vector database.
                                                By generating multiple perspectives on the customer's question, your goal is to help Finqalab overcome some of the limitations of the distance-based similarity search. 
                                                Provide the two alternative questions separated by newlines.
                                                Original question: {question}""",
    )

    llm_chain = QUERY_PROMPT | _get_model(llm, temp = 0.5) | LineListOutputParser()
    multi_query_retriever = MultiQueryRetriever(retriever = BM25Retriever.from_documents(all_documents, k = k_bm25, preprocess_func = word_tokenize), 
                                                llm_chain = llm_chain, 
                                                parser_key = "lines", 
                                                include_original = True)
    
    return multi_query_retriever


@lru_cache(maxsize = 1)
def _get_sqr(llm: str):

    metadata_field_info = [AttributeInfo(name = "FAQ_Category",
                                         description = "The category the Question belongs to. Valid values are ['General','Account Opening and Related Queries','Bank Transfers, Deposits and Payments', 'Portfolio','Stock Details','Shares','Trade','Withdrawal','Capital Gain Tax (CGT)','Dividends','Bonus Shares','Investment Advisory','Trading Errors and Restrictions','Bio Verification','Technical Issues with App']",
                                         type = "string"
    )]

    document_content_description = "FAQs and their answers prepared by Finqalab for Customer Support/Service."

    examples = [
        (
            "What are Finqalab's bank account details?",
            {
                "query": "What are Finqalab's bank account details?",
                "filter": 'eq("FAQ_Category", "Bank Transfers, Deposits and Payments")',
                
            },
        ),
        (
            "The way of adding stocks to my portfolio",
            {
                "query": "The way of adding stocks to my portfolio",
                "filter": 'and(eq("FAQ_Category", "Stock Details"),eq("FAQ_Category", "Portfolio"))',
            },
        ),
        (
            "I am not being allowed to sell and the message says that I do not possess sufficient holdings",
            {
                "query": "I am not being allowed to sell and the message says that I do not possess sufficient holdings",
                "filter": 'eq("FAQ_Category", "Trading Errors and Restrictions")',
            },
        ),
        (
            "Why do I see XD written on a stock?",
            {
                "query": "Why do I see XD written on a stock?",
                "filter": 'and(eq("FAQ_Category", "Dividends"),eq("FAQ_Category", "stock"))',
            },
        ),
        (
            "My cashbook is showing some ready exposure charges and I am not sure what they are?",
            {
                "query": "My cashbook is showing some ready exposure charges and I am not sure what they are?",
                "filter": 'NO_FILTER',
            },
        ),
    ]
    prompt = get_query_constructor_prompt(document_content_description, 
                                          metadata_field_info, 
                                          examples = examples,
                                          allowed_operators = ["and"],
                                          allowed_comparators = ["eq"])
    
    query_constructor = prompt | _get_model(llm, temp = 0) | StructuredQueryOutputParser.from_components()

    self_query_ret = SelfQueryRetriever(query_constructor = query_constructor, vectorstore = _get_vector_store('google'))

    return self_query_ret


@lru_cache(maxsize = 1)
def _get_ensemble_ret(w1: float, w2: float, c: int, k_bm25: int, llm_mqr: str, llm_sqr: str):

    return EnsembleRetriever(retrievers = [_get_mqr(llm = llm_mqr, k_bm25 = k_bm25), _get_sqr(llm = llm_sqr)], weights=[w1, w2], c = c)