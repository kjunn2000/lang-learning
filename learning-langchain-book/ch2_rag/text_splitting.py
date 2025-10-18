import asyncio
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def main():
    loader = TextLoader("/Users/thek3/dev/lang-learning/sample/text.txt")
    docs: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    texts = splitter.split_documents(docs)

    for i, text in enumerate(texts):
        print(f"--- Chunk {i+1} ---")
        print(text.page_content)
        print()


if __name__ == "__main__":
    asyncio.run(main())
