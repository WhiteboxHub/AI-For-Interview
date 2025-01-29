import os
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
class LLMSelector:
    def __init__(self, model_name, provider):
        self.model_name = model_name
        self.provider = provider
        self.llm = self.initialize_llm()

    def initialize_llm(self):
        if self.provider == "groq":
            return ChatGroq(model=self.model_name, api_key="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")
        elif self.provider == "mistral":
            return ChatMistralAI(model=self.model_name, api_key=os.getenv("MISTRAL_API_KEY"))
        else:
            raise ValueError("Unsupported LLM provider. Choose 'groq' or 'mistral'.")

    def generate_response(self, prompt):
        
        return self.llm.invoke(prompt)
