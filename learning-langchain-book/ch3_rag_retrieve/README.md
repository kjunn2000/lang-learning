Query Transformation
1. RRR - Rewrite-Retrieve-Read
2. Multi query - split query to multiple query, batch retireve RAG files and pass all them to llm, aims for boost recall
3. RAG fusion - RRF (reciprocal rank fusion) multi query - rank - fuses, aims for boosts precision 
4. Hypothetical Document Embeddings (HyDE) - turn the prompt to fake document (hypothetical document with sample answer), use the document for retrieval and llm (enhance context) 