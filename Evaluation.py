from llm_selector import LLMSelector
from prompt_engineering import PromptEngineering
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

inputs = [
    "What is Retrieval-Augmented Generation (RAG)?",
    "What is a Large Language Model (LLM)?",
    "How does vector search work in NLP?",
    "What are the benefits of using Hugging Face for NLP?",
    "What is a vector database?",
    "How do you optimize a machine learning pipeline?",
    "Explain the importance of fine-tuning models.",
    "What are the key differences between PyTorch and TensorFlow?",
    "What is the role of ElasticSearch in AI solutions?",
    "How would you approach scaling a Generative AI model?",
    "What is the difference between supervised and unsupervised learning?",
    "What is transfer learning in the context of NLP?",
    "How does BERT differ from GPT?",
    "What is the role of embeddings in NLP?",
    "How can you evaluate the performance of an NLP model?",
    "What is the significance of attention mechanisms in transformers?",
    "How do you handle imbalanced datasets in machine learning?",
    "What is the purpose of data augmentation in NLP?",
    "How does active learning work in machine learning?",
    "What are the challenges in deploying machine learning models in production?"
]

outputs = [
    "RAG combines retrieval with generation to enhance the output of language models by leveraging external information.",
    "LLMs are trained to process and generate human-like text, usually based on deep learning architectures like transformers.",
    "Vector search involves finding similarities in high-dimensional space, helping identify the closest matching entries to a query.",
    "Hugging Face provides a large repository of pre-trained models and tools that make it easier to fine-tune and deploy NLP models.",
    "A vector database stores high-dimensional vectors, allowing for fast similarity searches, commonly used in AI and NLP applications.",
    "Optimizing a pipeline involves improving data processing, model training, and inference to increase efficiency and reduce latency.",
    "Fine-tuning adjusts a pre-trained model to perform better on a specific task by training it on a smaller, task-specific dataset.",
    "PyTorch is dynamic and more flexible, while TensorFlow is static and has broader deployment options, both being popular for ML and AI tasks.",
    "ElasticSearch is used to perform scalable and fast text search operations, often integrated with AI for advanced search capabilities.",
    "Scaling involves distributing the workload across multiple resources, optimizing data pipelines, and adjusting models to handle larger datasets.",
    "Supervised learning involves training models with labeled data, while unsupervised learning uses unlabeled data to find patterns or structures.",
    "Transfer learning involves using a pre-trained model on a large dataset and fine-tuning it for a specific task, which can save time and resources.",
    "BERT is bidirectional and uses masked language modeling, while GPT is unidirectional and uses left-to-right language modeling.",
    "Embeddings are vector representations of words or phrases that capture semantic meaning and are used to improve model performance in NLP tasks.",
    "You can evaluate NLP models using metrics like accuracy, precision, recall, F1-score, and BLEU score, depending on the task.",
    "Attention mechanisms allow models to focus on specific parts of the input data, improving performance in tasks like translation and summarization.",
    "Handling imbalanced datasets involves techniques like oversampling the minority class, undersampling the majority class, or using synthetic data generation.",
    "Data augmentation in NLP involves creating new training examples by modifying existing ones, such as paraphrasing or synonym replacement, to improve model robustness.",
    "Active learning involves the model selecting the most informative data points for labeling, which can reduce the amount of labeled data needed.",
    "Challenges in deploying ML models include ensuring scalability, maintaining model performance, handling data drift, and ensuring security and compliance."
]
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
