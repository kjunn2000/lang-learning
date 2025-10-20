import asyncio
from dataclasses import dataclass
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import chain
import os
import dotenv
from langchain_core.vectorstores import VectorStoreRetriever

from prompt import ANSWER_PROMPT, MULTI_QUERY_PROMPT

dotenv.load_dotenv()


@dataclass
class RagFusionInput:
    query: str
    retriever: VectorStoreRetriever


def load_documents_and_retriever(file_path: str) -> VectorStoreRetriever:
    loader = TextLoader(file_path)
    docs: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    documents = splitter.split_documents(docs)

    model = OllamaEmbeddings(model="nomic-embed-text")
    connection = os.environ.get("PGVECTOR_URL")
    db = PGVector.from_documents(
        documents, model, connection=connection, collection_name="langchain_docs"
    )

    return db.as_retriever()


@chain
def generate_multi_query(input: str):
    multi_query_prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
    formatted = multi_query_prompt.invoke(input)

    llm = OllamaLLM(model="phi3")
    queries = llm.invoke(formatted)

    def parse_queries_output(output: str) -> List[str]:
        return [q.strip() for q in output.split("\n") if q.strip()]

    return parse_queries_output(queries)


@chain
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    documents = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content

            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (k + rank)
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True
    )
    return [documents[doc_str] for doc_str in reranked_doc_strs]


@chain
def retrieval_chain(input: RagFusionInput):
    # generate multi search query
    queries = generate_multi_query.invoke(input.query)

    # retrieve documents for each query
    docs = input.retriever.batch(queries)

    # fuse results
    fused = reciprocal_rank_fusion.invoke(docs)
    print(fused)
    return fused


@chain
def rag_fusion_qa(input: RagFusionInput):
    # retrieve context
    docs = retrieval_chain.invoke(input)

    # prepare query
    prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
    formatted = prompt.invoke({"context": docs, "question": input.query})

    # answer query
    llm = OllamaLLM(model="phi3")
    return llm.invoke(formatted)


async def main():
    retriever = load_documents_and_retriever(
        "/Users/thek3/dev/lang-learning/sample/text.txt"
    )

    input = RagFusionInput(
        query="Who are some key figures in the ancient greek history of philosophy?",
        retriever=retriever,
    )
    ans = rag_fusion_qa.invoke(input)

    print(ans)


if __name__ == "__main__":
    asyncio.run(main())
