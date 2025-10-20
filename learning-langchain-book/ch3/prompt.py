REFINE_PROMPT = """
You are a prompt engineer helping to improve an existing prompt.
Analyze the given prompt for clarity, specificity, and potential ambiguity.
Do not answer the prompt itself.
Then rewrite it to:
	1.	Focus on a single clear objective,
	2.	Remove unrelated or confusing details,
	3.	Use precise, neutral language,
	4.	Preserve the original intent.

Output one sections:
Analysis: Briefly explain what’s unclear or could cause misinterpretation.
Refined Prompt: Present an improved version that is concise and unambiguous. Start and end the refined prompt with double asterisks for clarity.

Here’s the prompt to refine:
{question}
"""

ANSWER_PROMPT = """
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make
up an answer.
{context}

Question: {question}
"""

MULTI_QUERY_PROMPT = """
You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to {question} \n
Output (4 queries):
"""
