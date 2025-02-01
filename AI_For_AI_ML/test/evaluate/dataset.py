{
  "QA_dataset": [
    {
      "question": "What embedding model did you use for your retriever, and how did you optimize it?",
      "answer": {
        "short": "Used sentence transformers, SBERT. Optimized via domain fine-tuning, hyperparam tuning (batch size, learning rate).",
        "detailed": "SBERT, Transformer-based, utilizes contrastive loss and triplet loss for effective fine-tuning. Hyperparameter tuning was achieved using grid search and Bayesian optimization. Embeddings were evaluated using cosine similarity, with precision and recall testing on retrieval performance."
      }
    },
    {
      "question": "What conclusion did you make about which retriever is good or bad, and how?",
      "answer": {
        "short": "Evaluated precision, recall, relevance. High-recall retrievers preferred for RAG-based tasks.",
        "detailed": "Retrievers were compared using Recall@K, NDCG, and MRR metrics. Dense retrievers like BGE and SBERT outperformed BM25, particularly in semantic relevance tasks. A hybrid retrieval approach (BM25+DPR) was found to offer a balance between speed and accuracy."
      }
    },
    {
      "question": "How did you collect the ground truth to evaluate the retriever?",
      "answer": {
        "short": "Annotated dataset, crowd-sourced labeling, used Natural Questions dataset.",
        "detailed": "Ground truth data was collected through human annotation, with relevance labels based on manual review. Active learning strategies helped optimize the labeling process. The inter-annotator agreement was calculated to ensure consistency, and fine-tuning was performed on labeled positives and negatives."
      }
    },
    {
      "question": "What metrics did you use to evaluate the retriever?",
      "answer": {
        "short": "Recall@K, MRR, NDCG, Precision@K. Evaluated retrieval effectiveness.",
        "detailed": "Key evaluation metrics included Recall@K for completeness, MRR for ranking accuracy, and NDCG for graded relevance. Dense, sparse, and hybrid retrieval models were compared on these metrics, benchmarking against traditional methods like BM25."
      }
    },
    {
      "question": "How did you measure the precision or other evaluation metrics like recall and relevance?",
      "answer": {
        "short": "Precision = TP/(TP+FP), Recall = TP/(TP+FN), relevance via human judgment.",
        "detailed": "Precision was calculated by evaluating true positives over false positives, while recall measured retrieval completeness. Relevance scores were based on human annotations, with cosine similarity used to validate the relevance of results. These metrics were crucial for refining the retrieval model."
      }
    },
    {
      "question": "Could you tell me the difference between parameter-efficient fine-tuning and full fine-tuning?",
      "answer": {
        "short": "Parameter-efficient fine-tuning updates a subset of model parameters, saving memory and compute.",
        "detailed": "Full fine-tuning updates all model parameters, whereas parameter-efficient methods like LoRA update a smaller subset, reducing computational overhead. LoRA involves adding low-rank matrices into the model, enabling efficient adaptation with fewer resources."
      }
    },
    {
      "question": "How do LoRA (Low-Rank Adaptation) adapters work?",
      "answer": {
        "short": "LoRA injects low-rank matrices into model layers to reduce parameter updates during fine-tuning.",
        "detailed": "LoRA works by introducing low-rank matrices into the model’s weights, which allows efficient fine-tuning with fewer parameters. This reduces memory and compute usage while maintaining model performance by retaining the ability to adapt to specific tasks."
      }
    },
    {
      "question": "What is the major difference between LoRA and quantized LoRA?",
      "answer": {
        "short": "LoRA uses low-rank matrices for fine-tuning; quantized LoRA reduces precision for further efficiency.",
        "detailed": "LoRA involves introducing low-rank matrices to save compute and memory. Quantized LoRA takes it further by reducing the precision of these matrices (e.g., from 32-bit to 8-bit), achieving even greater memory efficiency without significantly sacrificing performance."
      }
    },
    {
      "question": "How does the transformer model fix long dependencies compared to LSTMs?",
      "answer": {
        "short": "Transformers use self-attention, which can model long-range dependencies more effectively than LSTMs.",
        "detailed": "Transformers overcome the limitations of LSTMs by using self-attention mechanisms, allowing them to model relationships across long sequences in parallel. LSTMs, by contrast, process sequences sequentially and struggle with capturing long-range dependencies due to vanishing gradients."
      }
    },
    {
      "question": "What are the major differences between LSTMs and transformer models?",
      "answer": {
        "short": "LSTMs process sequentially, transformers use self-attention and parallel processing.",
        "detailed": "LSTMs handle input sequentially, which leads to difficulties in learning long-range dependencies and limits parallelization. Transformers, however, leverage self-attention to model relationships across the entire sequence simultaneously, enabling better handling of long dependencies and faster training."
      }
    },
    {
      "question": "What is the objective function of any generative model?",
      "answer": {
        "short": "Maximize the likelihood of generating the correct sequence, typically using cross-entropy loss.",
        "detailed": "Generative models aim to maximize the likelihood of generating the target sequence, usually employing cross-entropy loss to measure the difference between predicted and true token sequences. This objective function enables the model to learn to generate coherent and contextually accurate outputs."
      }
    },
    {
      "question": "What was the objective function or training process used in the BERT model?",
      "answer": {
        "short": "BERT uses masked language modeling (MLM) and next sentence prediction (NSP) as its objective functions.",
        "detailed": "BERT’s pre-training involves masked language modeling (MLM) where random tokens are masked, and the model predicts these tokens. Additionally, the next sentence prediction (NSP) task helps BERT learn sentence relationships, making it highly effective for tasks like question answering and sentence classification."
      }
    },
    {
      "question": "What advantages does the RAG (Retrieval-Augmented Generation) framework offer over directly applying a foundation model?",
      "answer": {
        "short": "RAG integrates retrieval with generation for more context-aware and accurate responses.",
        "detailed": "RAG improves upon traditional models by combining retrieval with generation. The model retrieves relevant external documents before generating an answer, which allows it to provide more accurate and contextually informed responses compared to directly using a foundation model alone."
      }
    },
    {
      "question": "Could you elaborate on the different types of agents you built and their specific tasks within the agentic AI framework?",
      "answer": {
        "short": "Built retrieval, ranking, and generation agents to handle specific tasks in the pipeline.",
        "detailed": "The agentic AI framework consists of various specialized agents. The retrieval agent fetches relevant documents, the ranking agent scores their relevance, and the generation agent uses these ranked documents to generate precise responses based on context and task-specific requirements."
      }
    },
    {
      "question": "Could you provide more details on your specific role in the project, particularly regarding the development and integration of these agents?",
      "answer": {
        "short": "Designed and integrated retrieval and generation agents for seamless task execution.",
        "detailed": "My role involved the design, implementation, and integration of multiple agents within the AI framework. I ensured smooth communication between agents, optimized their performance, and tailored each agent to specific tasks such as document retrieval, relevance ranking, and answer generation."
      }
    },
    {
      "question": "Could you outline the step-by-step process involved in retrieving and processing documents within the RAG framework?",
      "answer": {
        "short": "Query embedding → Document retrieval → Relevance ranking → Context extraction → Answer generation.",
        "detailed": "The process begins with embedding the query, followed by retrieving candidate documents using a retriever model. These documents are then ranked based on relevance, after which the most relevant ones are used to extract context for the generation model to produce a final answer."
      }
    },
    {
      "question": "How did you evaluate the validity and accuracy of the retrieved results?",
      "answer": {
        "short": "Used human-annotated datasets, precision-recall curves, and relevance scoring.",
        "detailed": "Evaluation was done through human-annotated relevance labels, precision-recall curves to assess completeness, and cosine similarity to measure contextual match. This helped ensure the accuracy of the retrieval process, as well as its alignment with user intent."
      }
    },
    {
      "question": "What measures do you have in place to ensure the system consistently delivers expected results once it goes live?",
      "answer": {
        "short": "Implemented monitoring, alerting, and periodic retraining for performance consistency.",
        "detailed": "To ensure reliable performance, the system is continuously monitored for errors and anomalies. Alerts are triggered for critical issues, and periodic retraining is performed to keep the model up-to-date with new data and ensure long-term accuracy."
      }
    },
    {
      "question": "Could you describe the periodic checks and processes you use to ensure the system runs correctly?",
      "answer": {
        "short": "Automated testing, performance monitoring, and retraining with updated data.",
        "detailed": "We perform automated tests to ensure that the system’s components are working as expected. Additionally, performance is monitored in real-time, and the system undergoes periodic retraining to account for any new data or changes in user requirements."
      }
    },
    {
      "question": "How do you address specific patterns or concerns identified through customer feedback?",
      "answer": {
        "short": "Analyze feedback, identify patterns, fine-tune model or retrieval process.",
        "detailed": "Feedback is systematically analyzed to identify recurring issues or concerns. Based on these insights, we make adjustments to the model or the retrieval pipeline to ensure customer satisfaction and improve overall system performance."
      }
    },
    {
      "question": "Could you provide a specific example where you observed an enhancement in the system's performance as a result of your improvements?",
      "answer": {
        "short": "Improved retrieval accuracy by 15% after fine-tuning embedding model.",
        "detailed": "After fine-tuning the embedding model on domain-specific data, I observed a 15% improvement in retrieval accuracy. This was measured using recall@K and precision@K metrics, showing a significant enhancement in both the relevance and precision of retrieved documents."
      }
    },
    {
      "question": "If tasked with exploring a RAG-based approach for document summarization, what system or setup would you propose?",
      "answer": {
        "short": "Use RAG with fine-tuned retriever and summarization model for context-aware summaries.",
        "detailed": "For document summarization, I would propose a RAG-based system where a fine-tuned retriever fetches the most relevant sections of documents. These documents are then passed through a summarization model that generates concise and contextually rich summaries tailored to the user’s needs."
      }
    },
    {
      "question": "How do you ensure that the generated summaries are relevant and valuable for your customers?",
      "answer": {
        "short": "Fine-tune on customer-specific data, validate with human evaluation.",
        "detailed": "Relevance is ensured by fine-tuning the summarization model on data specific to the customer’s domain. Additionally, summaries are validated through human evaluation and feedback, ensuring they meet the user’s expectations and provide value."
      }
    },
    {
      "question": "How do you ensure the model's external validity, fairness, and lack of bias?",
      "answer": {
        "short": "Audit datasets, test for bias, use diverse training data.",
        "detailed": "To ensure fairness and external validity, we audit datasets for any biases and conduct tests to evaluate the model’s behavior across diverse demographics. Diverse training data from various sources is used to minimize the risk of bias and ensure generalizability."
      }
    },
    {
      "question": "How would you implement this system using AWS services?",
      "answer": {
        "short": "Use AWS SageMaker for training, S3 for storage, Lambda for serverless processing.",
        "detailed": "AWS SageMaker would be used for model training and deployment, with S3 handling data storage. AWS Lambda would facilitate serverless processing for efficient scaling, and AWS SQS could be used for queuing tasks within the pipeline."
      }
    },
    {
      "question": "Could you describe your role in automating the ML pipeline using MLflow, including your responsibilities and how you streamlined the workflow?",
      "answer": {
        "short": "Automated pipeline using MLflow for experiment tracking and seamless deployment.",
        "detailed": "My role involved automating the end-to-end ML pipeline using MLflow. I ensured seamless tracking of experiments, model versioning, and easy deployment to production. This streamlined the workflow by reducing manual intervention and improving reproducibility."
      }
    },
    {
      "question": "Could you detail the engineering tasks and processes you undertook for web scraping?",
      "answer": {
        "short": "Built scrapers using BeautifulSoup, Scrapy, handled data cleaning, managed rate limiting.",
        "detailed": "I built robust scrapers using BeautifulSoup and Scrapy, ensuring that data was extracted efficiently from various sources. I also handled data cleaning, removing irrelevant content, and managed performance issues like rate limiting using delay strategies."
      }
    },
    {
      "question": "What kind of data did you collect through web scraping, and what processing did you perform on this data? Additionally, what issues did you encounter regarding performance or data quality?",
      "answer": {
        "short": "Collected text, images, metadata; processed by cleaning, deduplication, and structuring.",
        "detailed": "I collected a variety of data, including text, images, and metadata. After extraction, I cleaned the data by removing duplicates and irrelevant content, structured it for storage, and faced challenges such as IP blocking and incomplete data from certain sources."
      }
    },
    {
      "question": "What specific HTML sources were you scraping, and what type of information were you extracting?",
      "answer": {
        "short": "Scraped e-commerce sites, news sites, and forums for product details, articles, user content.",
        "detailed": "I focused on scraping e-commerce sites for product details, news websites for articles, and forums for user-generated content. The data extracted included product descriptions, reviews, and articles with relevant keywords and metadata."
      }
    },
    {
      "question": "Could you briefly describe a few APIs you have worked with?",
      "answer": {
        "short": "Worked with Twitter API, Google Maps API, GitHub API for data extraction.",
        "detailed": "I’ve worked with REST APIs like the Twitter API for collecting tweet data, Google Maps API for location-based data, and the GitHub API for extracting metadata related to repositories, pull requests, and issues."
      }
    },
    {
      "question": "Why did you choose MLflow or another specific tool for your ML pipeline automation?",
      "answer": {
        "short": "MLflow for experiment tracking, model versioning, and deployment.",
        "detailed": "MLflow was chosen due to its comprehensive feature set, including experiment tracking, model versioning, and seamless integration for deployment. This tool streamlined model management and improved collaboration across teams."
      }
    },
    {
      "question": "How do you optimize the model within the RAG framework?",
      "answer": {
        "short": "Fine-tune retriever and generator, optimize retrieval ranking, adjust hyperparameters.",
        "detailed": "Optimization in the RAG framework involves fine-tuning both the retriever and generator models, ensuring that the retrieval ranking is accurate. Hyperparameters such as batch size and learning rate are adjusted to enhance performance and minimize error."
      }
    },
    {
      "question": "At what stage in the process would you apply hyperparameter tuning?",
      "answer": {
        "short": "Apply during training to optimize model performance and generalization.",
        "detailed": "Hyperparameter tuning is applied during the training phase, typically after the model has been initialized. This helps to optimize the model’s performance, adjusting parameters like learning rate, batch size, and architecture to achieve better accuracy and generalization."
      }
    },
    {
      "question": "Could you explain how you optimize the hyperparameter tuning process?",
      "answer": {
        "short": "Use methods like grid search, random search, and Bayesian optimization.",
        "detailed": "The hyperparameter tuning process is optimized using techniques like grid search for exhaustive exploration, random search for a quicker solution, and Bayesian optimization for more efficient, guided search toward optimal configurations."
      }
    },
    {
      "question": "Could you discuss the evaluation metrics you use for summarization tasks, such as BLEU and ROUGE?",
      "answer": {
        "short": "Use BLEU for n-gram overlap, ROUGE for recall-oriented evaluation.",
        "detailed": "For summarization tasks, BLEU is used to measure n-gram overlap between generated and reference summaries, while ROUGE evaluates recall-oriented metrics, focusing on the ability to capture relevant content from the source document."
      }
    }
  ]
}
