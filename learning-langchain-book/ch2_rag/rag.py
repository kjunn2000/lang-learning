import asyncio
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import chain
import os
import dotenv

dotenv.load_dotenv()


async def main():
    loader = TextLoader("/Users/thek3/dev/lang-learning/sample/text.txt")
    docs: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    documents = splitter.split_documents(docs)

    # model = Ollama(model="nomic-embed-text")
    model = OllamaEmbeddings(model="nomic-embed-text")
    connection = os.environ.get("PGVECTOR_URL")
    db = PGVector.from_documents(
        documents, model, connection=connection, collection_name="langchain_docs"
    )

    retriever = db.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
            You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make
            up an answer.
            {context}

            Question: {question}
        """
    )
    llm = OllamaLLM(model="phi3")

    @chain
    def qa(input):
        docs = retriever.invoke(input)
        formatted = prompt.invoke({"context": docs, "question": input})
        answer = llm.invoke(formatted)
        return answer

    ans = qa.invoke("Who are the founders of Meta?")
    print(ans)


if __name__ == "__main__":
    asyncio.run(main())
