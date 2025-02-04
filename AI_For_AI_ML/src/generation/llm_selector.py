import os
import json
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv

load_dotenv()

OPIK_API_KEY = os.getenv("OPIK_API_KEY") 
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE") 
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME") 
API_KEY = os.getenv("groq_api_key")  

opik_tracer = OpikTracer()

class LLMSelector:
    def __init__(self, config_file=r"C:\Users\dhira\Desktop\v_0_3\AI-For-Interview\AI_For_AI_ML\src\generation\config.json"):
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)  # Load the list of model configurations
            
            if not isinstance(self.config, list):
                raise ValueError("Config file should contain a list of model configurations.")
            
            self.llms = [self.initialize_llm(model) for model in self.config]  # Initialize all LLMs
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_file}.")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in the config file.")

    def initialize_llm(self, model_config):
        model_name = model_config.get("llama3-70b-8192")
        provider = model_config.get("provider")
        

        if provider == "groq":
            
            return ChatGroq(model= "mixtral-8x7b-32768",api_key=API_KEY, callbacks=[opik_tracer])
        elif provider == "mistral":
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            if not mistral_api_key:
                raise ValueError("MISTRAL_API_KEY environment variable is not set.")
            return ChatMistralAI(model=model_name, api_key=mistral_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Choose 'groq' or 'mistral'.")

    def generate_response(self, prompt, model_index=0):
        if model_index >= len(self.llms):
            raise ValueError(f"Model index {model_index} is out of range. Only {len(self.llms)} models are available.")
        return self.llms[model_index].invoke(prompt)