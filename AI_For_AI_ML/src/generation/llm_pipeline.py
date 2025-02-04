from llm_selector import LLMSelector
from langchain_core.output_parsers import StrOutputParser
import os 
import sys 
# dataset.py
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../..")))
from src.generation.llm_selector import  LLMSelector # Now it will work
from src.prompt_templates.prompt_engineering import PromptEngineering

class LLMPipeline:
    def __init__(self):
        self.llm_selector = LLMSelector()
        self.prompt_engineering = PromptEngineering()

    def ask_question(self, question):
        prompt_template = self.prompt_engineering.get_interview_prompt_5()
        formatted_prompt = prompt_template.format(question=question)
        response = self.llm_selector.generate_response(formatted_prompt)
        return response

if __name__ == "__main__":
    llm_pipeline = LLMPipeline()
    response = llm_pipeline.ask_question("this is the second question tell about nlp .?")
    structure_output = StrOutputParser()
    print(structure_output.invoke(response))