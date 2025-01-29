import json
from llm_selector import LLMSelector
from dataset import inputs, outputs
from llm_pipeline import LLMPipeline
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


class ResponseEvaluator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge = Rouge()
        self.llm_pipeline = LLMPipeline()  # Initialize LLM pipeline

    def evaluate_response(self, generated, ground_truth):
        """Compute BLEU, ROUGE, and Cosine Similarity scores."""
        
        # Ensure generated is a string (extract text from AIMessage)
        if isinstance(generated,):
            generated = generated.content  # Extract text
        
        # Ensure reference is a string
        if isinstance(ground_truth):
            ground_truth = ground_truth.content  

        # BLEU Score
        bleu_score = sentence_bleu([ground_truth.split()], generated.split())

        # ROUGE Score
        rouge_scores = self.rouge.get_scores(generated, ground_truth, avg=True)

        # Cosine Similarity
        gen_embedding = self.model.encode(generated, convert_to_tensor=True)
        ref_embedding = self.model.encode(ground_truth, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(gen_embedding, ref_embedding).item()

        return {
            "BLEU": round(bleu_score, 4),
            "ROUGE-L": round(rouge_scores["rouge-l"]["f"], 4),
            "Cosine Similarity": round(cosine_similarity, 4)
        }


    def run_evaluation(self):
        results = []
        for i, sample in enumerate(inputs):
            question = sample["question"]
            reference = outputs[i]["answer"]

            # Generate response using LLM pipeline
            generated_response = self.llm_pipeline.ask_question(question)

            # Evaluate the response
            scores = self.evaluate_response(generated_response, reference)

            results.append({
                "Question": question,
                "Generated Response": generated_response,
                "Reference Response": reference,
                "Evaluation Scores": scores
            })
        
        # Save results to JSON
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=4)

        return results

if __name__ == "__main__":
    evaluator = ResponseEvaluator()
    evaluation_results = evaluator.run_evaluation()
    for result in evaluation_results:
        print(result)
