import asyncio
import re
from typing import List
import langchain
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

from prompt import REFINE_PROMPT, ANSWER_PROMPT

dotenv.load_dotenv()


async def main():
    loader = TextLoader("/Users/thek3/dev/lang-learning/sample/meta.txt")
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
    llm = OllamaLLM(model="phi3")

    rewrite_prompt = ChatPromptTemplate.from_template(REFINE_PROMPT)

    def parse_rewriter_output(output: str) -> str:
        match = re.search(r"Refined Prompt:\s*\**(.*?)\**(?:\n|$)", output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return output.strip()

    rewriter = rewrite_prompt | llm | parse_rewriter_output

    prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)

    @chain
    def qa(input: str) -> str:
        rewritten = rewriter.invoke({"question": input})
        docs = retriever.invoke(rewritten)
        print(rewritten + "\n\n")
        formatted = prompt.invoke({"context": docs, "question": rewritten})
        answer = llm.invoke(formatted)
        return answer

    ans = qa.invoke(
        "Based on the file, I know the fastest land animal invented the printing press, but since Meta’s headquarters is on the moon, does that mean Gutenberg worked at Facebook before or after 1969?"
    )
    print(ans)


if __name__ == "__main__":
    asyncio.run(main())
