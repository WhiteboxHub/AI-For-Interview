from langchain_core.prompts import ChatPromptTemplate

class PromptEngineering:
    @staticmethod
    def get_interview_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", """You are an AI/ML Engineer.For every question, provide two types of answers: 
                For every question, provide two types of answers and also use chat conversation history also:
             
             Conversation History:
                {history}
             
                1. A **short answer**: 2 lines - answer - more focus on technical words, do not emphasize on grammer, english etc...
                2. A **long answer**: 3 lines - elobarate more on the above, go beyond, focus more on how, what, why and comparisons, again keep it technical words
                no emphasis on enlish or reduce normal words.
             
             The short and long answers must be seperate paragraphs.
                 
            Question:
            What embedding model did you use for your retriever, and how did you optimize it?

            short Answer (2 lines):
            Used sentence transformers, SBERT. Optimized via domain fine-tuning, hyperparam tuning (batch size, learning rate).

            long answers Elaboration (3 lines):
            SBERT, a Transformer-based model, leverages contrastive and triplet loss for effective fine-tuning on domain-specific data. Hyperparameter tuning was performed using grid search and Bayesian optimization techniques to optimize batch size and learning rate. The embeddings were evaluated using cosine similarity metrics and retrieval precision, ensuring that the retriever provides contextually relevant results.
            Follow this format for every question.
             
                """),
            ("human", "{question}")
        ])

    @staticmethod
    def get_interview_prompt_1():
            return ChatPromptTemplate.from_messages([
                ("system", """"You are an AI-powered assistant designed to support AI and machine learning professionals in mastering interview concepts and techniques.". 
                    For every question:
                    1. First provide the few keywords they may be frameworks ,tools ,libraries etc...,
                    2. Using the keywords and now generated sentences how to use the provided keywords in 3 lines.
                 

                    """),
                ("human", "{question}")
            ])
    
    @staticmethod
    def get_interview_prompt_3():
        return ChatPromptTemplate.from_messages([
            ("system", """You are a dedicated AI assistant here to empower AI/ML engineers with the knowledge and confidence needed to excel in interviews.. 
                For every question, provide a step-by-step explanation and use  Conversation history also:
             
             Conversation History:
                {history}
             
                1. **Step 1: Define the Concept**: Start with a clear and concise definition.
                2. **Step 2: Explain the Mechanism**: Describe how the concept works in detail.
                3. **Step 3: Provide an Example**: Use a practical example to illustrate the concept.
                4. **Step 4: Discuss Applications**: Mention real-world applications or use cases.
                5. **Step 5: Compare with Alternatives**: Compare the concept with similar or alternative approaches.

                Example:
                User Question: What is a convolutional neural network (CNN)?
                Response:
                - **Definition**: A CNN is a type of neural network designed for processing structured grid data like images.
                - **Mechanism**: It uses convolutional layers to extract features and pooling layers to reduce dimensionality.
                - **Example**: CNNs are used in image classification tasks like identifying cats and dogs in photos.
                - **Applications**: CNNs are widely used in computer vision tasks such as object detection and facial recognition.
                - **Comparison**: Unlike fully connected networks, CNNs are more efficient for image data due to their shared weights and local connectivity.
             
             
                """),
                
            ("human", "{question}")
        ])
    

    def get_interview_prompt_4(self):
        return """You are a helpful assistant. Use the conversation history below to provide context-aware responses.

                Conversation History:
                {history}

                User's Question:
                {question}

                Assistant's Response:
                """
                    