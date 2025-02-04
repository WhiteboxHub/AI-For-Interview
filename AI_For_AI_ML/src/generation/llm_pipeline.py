import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.generation.llm_selector import LLMSelector  
from src.prompt_templates.prompt_engineering import PromptEngineering 
from langchain_core.output_parsers import StrOutputParser


class LLMPipeline:
    def __init__(self, history_file="conversation_history.json", max_history_length=10):
        """
        Initialize the pipeline with persistent history and adjustable history length.
        
        Args:
            history_file (str): File to store conversation history.
            max_history_length (int): Maximum number of interactions to keep in history.
        """
        self.llm_selector = LLMSelector()
        self.prompt_engineering = PromptEngineering()
        self.history_file = history_file  # File to store conversation history
        self.max_history_length = max_history_length  # Maximum history length
        self.conversation_history = self.load_conversation_history()  # Load existing history

    def load_conversation_history(self):
        """Load conversation history from a JSON file."""
        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as file:
                return json.load(file)
        return []  # Return an empty list if the file doesn't exist

    def save_conversation_history(self):
        """Save conversation history to a JSON file."""
        with open(self.history_file, "w", encoding="utf-8") as file:
            json.dump(self.conversation_history, file, indent=4)

    def trim_conversation_history(self):
        """
        Trim the conversation history to ensure it does not exceed the maximum length.
        """
        if len(self.conversation_history) > self.max_history_length:
            # Remove the oldest entries to maintain the desired length
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def ask_question(self, question):
        # Add the user's question to the conversation history
        self.conversation_history.append({"role": "user", "content": question})

        # Trim the history to ensure it does not exceed the maximum length
        self.trim_conversation_history()

        # Get the prompt template and format it with the conversation history and the current question
        prompt_template = self.prompt_engineering.get_interview_prompt()
        formatted_prompt = prompt_template.format(
            history=json.dumps(self.conversation_history, indent=2),  # Convert history to JSON for readability
            question=question  # Pass the current question
        )

        # Generate a response using the LLM
        response = self.llm_selector.generate_response(formatted_prompt)
        structure_output = StrOutputParser()
        parsed_response = structure_output.invoke(response)

        # Add the assistant's response to the conversation history
        self.conversation_history.append({"role": "assistant", "content": parsed_response})

        # Trim the history again after adding the assistant's response
        self.trim_conversation_history()

        # Save the updated conversation history to the file
        self.save_conversation_history()

        return response

    def get_conversation_history(self):
        # Return the full conversation history
        return self.conversation_history

if __name__ == "__main__":
    # Initialize the pipeline with a maximum history length of 5 interactions
    llm_pipeline = LLMPipeline(max_history_length=5)

    

        # Get the response from the chatbot
    response = llm_pipeline.ask_question("what is genaration evaluation?")

        # Parse and print the structured output
    structure_output = StrOutputParser()
    parsed_response = structure_output.invoke(response)
    print(f"Bot: {parsed_response}")

        # Optionally, print the conversation history
    print("\nConversation History:")
    for entry in llm_pipeline.get_conversation_history():
        print(f"{entry['role'].capitalize()}: {entry['content']}")


        