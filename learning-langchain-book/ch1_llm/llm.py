from langchain_community.llms.ollama import Ollama 

llm = Ollama(model="phi3")

response = llm.invoke("Explain LangChain in one sentence.")
print(response)