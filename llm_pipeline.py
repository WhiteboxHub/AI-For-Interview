from llm_selector import LLMSelector
from prompt_engineering import PromptEngineering
from langchain_core.output_parsers import StrOutputParser

class LLMPipeline:
    def __init__(self, model_name="llama3-70b-8192", provider="groq"):
        self.llm_selector = LLMSelector(model_name, provider)
        self.prompt_engineering = PromptEngineering()

    def ask_question(self, question):
        prompt_template = self.prompt_engineering.get_interview_prompt()
        formatted_prompt = prompt_template.format(question=question)
        response = self.llm_selector.generate_response(formatted_prompt)
        return response

if __name__ == "__main__":
    llm_pipeline = LLMPipeline()
    response = llm_pipeline.ask_question("What are the benefits of deep learning?")
    structure_output = StrOutputParser()
    print(structure_output.invoke(response))
