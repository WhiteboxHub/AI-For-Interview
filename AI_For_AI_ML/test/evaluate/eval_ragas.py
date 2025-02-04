

import json
from langchain_core.messages import AIMessage
from sentence_transformers import SentenceTransformer
from rouge import Rouge
# from src.generation.llm_pipeline import LLMPipeline
# from src.generation import llm_selector

from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util
# from src.prompt_templates import prompt_engineering
import os 
import sys 
from ragas.metrics import answer_relevancy, AnswerCorrectness
from ragas import evaluate
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../..")))
from src.generation.llm_selector import  LLMSelector 
from src.prompt_templates.prompt_engineering import PromptEngineering 

# Load the dataset from a JSON file
with open("/Users/innovapathinc/Desktop/saturday_night /AI_For_Interview/AI_For_AI_ML/test/evaluate/dataset.py", "r") as file:
    QA_dataset = json.load(file)

# Assuming you have the dataset loaded from the JSON file

# LLM pipeline to generate answers
class LLMPipeline:
    def __init__(self):
        self.llm_selector = LLMSelector()
        self.prompt_engineering = PromptEngineering()

    def ask_question(self, question):
        prompt_template = self.prompt_engineering.get_interview_prompt_1()
        formatted_prompt = prompt_template.format(question=question)
        response = self.llm_selector.generate_response(formatted_prompt)
        return response

# Response Evaluation class (BLEU, ROUGE, Cosine Similarity)
class ResponseEvaluator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge = Rouge()

    def evaluate_response(self, generated, ground_truth):
        # Print the type of generated response for debugging
        print(f"Type of generated: {type(generated)}")

        # Extract the actual text if the generated response is an AIMessage object or another structure
        if isinstance(generated, str):
            generated_text = generated
        elif isinstance(generated, AIMessage):  # Check if it is an AIMessage object
            generated_text = generated.content  # Access the 'content' attribute
        else:
            raise ValueError("Generated response does not have a valid text attribute.")

        # BLEU Score
        bleu_score = sentence_bleu([ground_truth.split()], generated_text.split())

        # ROUGE Score
        rouge_scores = self.rouge.get_scores(generated_text, ground_truth, avg=True)

        # Cosine Similarity
        gen_embedding = self.model.encode(generated_text, convert_to_tensor=True)
        ref_embedding = self.model.encode(ground_truth, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(gen_embedding, ref_embedding).item()

        return {
            "BLEU": round(bleu_score, 4),
            "ROUGE-L": round(rouge_scores["rouge-l"]["f"], 4),
            "Cosine Similarity": round(cosine_similarity, 4)
        }


    def evaluate_ragas(self, question, generated, ground_truth):
        """ Evaluate generated responses using RAGAS metrics. """
        
        if isinstance(generated, str):
            generated_text = generated
        elif isinstance(generated, AIMessage):
            generated_text = generated.content
        else:
            raise ValueError("Generated response does not have a valid text attribute.")

        dataset = [{
            "question": question,
            "ground_truth": ground_truth,
            "response": generated_text
        }]

        scores = {
            "Relevance": evaluate(dataset,metrics=[answer_relevancy]),
            "Answer Correctness": AnswerCorrectness.evaluate(dataset)        }

        return {key: round(value, 4) for key, value in scores.items()}

  

if __name__ == "__main__":
    llm_pipeline = LLMPipeline()
    evaluator = ResponseEvaluator()
    evaluation_results = []

    for entry in QA_dataset["QA_dataset"]:
        question = entry["question"]
        reference_answer = entry["answer"]["detailed"]  # Reference ground truth
        
        # Generate response
        generated_answer = llm_pipeline.ask_question(question)

        # Evaluate using BLEU, ROUGE, Cosine Similarity
        scores_traditional = evaluator.evaluate_response(generated_answer, reference_answer)

        # Evaluate using RAGAS
        scores_ragas = evaluator.evaluate_ragas(question, generated_answer, reference_answer)

        # Store results
        evaluation_results.append({
            "Question": question,
            "Generated Response": generated_answer,
            "Reference Response": reference_answer,
            "Traditional Scores": scores_traditional,
            "RAGAS Scores": scores_ragas
        })

    # Print the evaluation results
    for result in evaluation_results:
        print(f"Question: {result['Question']}")
        print(f"Evaluation Scores: {result['Traditional Scores']}")
        print(f"RAGAS Scores: {result['RAGAS Scores']}")
        print("="*50)
