## Key Concepts

- Vector: a mathematical representation of a list of values.
- Embeddings: numeric (vector) representations that capture the semantic meaning of text.

## Setup

- Vector store: PGVector (Postgres extension for storing and querying vectors).

## Indexing & Retrieval Techniques

1. MultiVectorRetriever — combines a vector store (for summaries or semantic search) with a document store (for raw documents) to improve retrieval precision and context.
2. RAPTOR — hierarchical or multi-level summaries that provide different levels of abstraction for documents (e.g., fine-grained to high-level summaries).
3. ColBERT — a retrieval model that computes contextualized embedding similarity and can re-rank results using efficient scoring techniques.

