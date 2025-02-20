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
        self.context = []  # Store previous conversation history

    def ask_question(self, question):
        # Add the new question to the context
        self.context.append(f"User: {question}")

        # Join the context into a single string
        context_str = "\n".join(self.context)

        # Get the prompt template
        prompt_template = self.prompt_engineering.get_interview_prompt_5()
        
        # Format the prompt with the question and context
        formatted_prompt = prompt_template.format(question=question, context=context_str)

        # Get the response from the model
        response = self.llm_selector.generate_response(formatted_prompt)
        
        # Add the model's response to the context
        self.context.append(f"Model: {response}")

        return response

if __name__ == "__main__":
    llm_pipeline = LLMPipeline()
    response1 = llm_pipeline.ask_question("Tell me about NLP.")
    structure_output = StrOutputParser()
    print(structure_output.invoke(response1))
    
    response2 = llm_pipeline.ask_question("What are the applications of NLP?")
    print(structure_output.invoke(response2))

    response3 = llm_pipeline.ask_question("What are the drawbacks..?")
    print(structure_output.invoke(response3))

