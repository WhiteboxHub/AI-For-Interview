from llm_selector import LLMSelector
from prompt_engineering import PromptEngineering
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

from dataset import inputs , outputs

from langchain_core.messages import AIMessage

# LLM pipeline to generate answers
class LLMPipeline:
    def __init__(self, model_name="llama3-70b-8192", provider="groq"):
        self.llm_selector = LLMSelector(model_name, provider)
        self.prompt_engineering = PromptEngineering()

    def ask_question(self, question):
        prompt_template = self.prompt_engineering.get_interview_prompt()
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
    def evaluate_with_ragas(self, question, generated, ground_truth):
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

        results = evaluate(dataset, [faithfulness, answer_relevancy, answer_correctness])

        return {
            "Faithfulness": round(results["faithfulness"], 4),
            "Answer Relevancy": round(results["answer_relevancy"], 4),
            "Answer Correctness": round(results["answer_correctness"], 4)
        }


# Main function to generate and evaluate responses
if __name__ == "__main__":
    llm_pipeline = LLMPipeline()
    evaluator = ResponseEvaluator()
    evaluation_results = []

    for i, question in enumerate(inputs):
        reference_answer = outputs[i]

        # Generate response from LLM pipeline
        generated_answer = llm_pipeline.ask_question(question)

        # Evaluate the generated response
        scores = evaluator.evaluate_response(generated_answer, reference_answer)

        
        # Append results
        evaluation_results.append({
            "Question": question,
            "Generated Response": generated_answer,
            "Reference Response": reference_answer,
            "Evaluation Scores": scores
        })

    # Print the evaluation results
    for result in evaluation_results:
        print(f"Question: {result['Question']}")
        print(f"Generated Response: {result['Generated Response']}")
        print(f"Reference Response: {result['Reference Response']}")
        print(f"Evaluation Scores: {result['Evaluation Scores']}")
        print("="*50)
