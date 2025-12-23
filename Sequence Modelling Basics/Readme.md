# Sequence-Based Sentiment Analysis on Amazon Reviews

## Project Objective
The goal of this project is to develop a sequence-based deep learning model—specifically exploring **RNN** and **LSTM** architectures—to predict sentiment polarity (positive or negative) from Amazon product review text. The model aims to capture emotional cues, subtle tones, and sarcasm beyond simple keyword matching.

## Dataset Summary
 The project utilizes the **Amazon Reviews Dataset**
*  **Text**: The full review written by the user
*  **Title**: The title of the review
*  **Polarity**: Labeled as **1 for Negative** and **2 for Positive**.

## Preprocessing Pipeline
To prepare the text for the neural networks, the following steps were implemented:
1.   **Cleaning**: Lowercasing, removing extra spaces, and stripping non-alphabetic characters
2.   **Tokenization**: Splitting text into individual word tokens
3.   **Padding/Truncating**: Reviews were post-padded or truncated to a fixed `MAX_LEN` of **256**
4.   **Vocabulary**: Retained the top **30,000** most frequent words to manage memory constraints
5.   **Word Embeddings**: Leveraged pre-trained **100-dimensional GloVe embeddings**.  The embedding layer weights were set to non-trainable to preserve pre-trained semantics

## Model Architectures
Two primary models were built and compared:

| Feature | RNN Model | LSTM Model |
| :--- | :--- | :--- |
| **Type** |  2-layer Bidirectional RNN  |  2-layer Bidirectional LSTM  |
| **Hidden Size** |  128  |  128  |
| **Regularization** |  None (due to resource limits)  |  Dropout (0.5)  |
| **Output Layer** |  Fully connected for binary output  |  Fully connected for binary output  |

### Training Details
*  **Loss Function**: Binary Cross-Entropy with Logits (includes built-in Sigmoid)
*  **Optimizer**: Adam with a learning rate of 0.001.
*  **Batch Size**: 512 (chosen for faster training per epoch).
*  **Hardware**: Trained for 4 epochs on a **Tesla T4 GPU** via Google Colab.

## Results and Evaluation
 The LSTM model significantly outperformed the vanilla RNN in capturing long-range dependencies.

| Model | Test Accuracy | Test F1-Score |
| :--- | :--- | :--- |
| **RNN** |  ~81.70%  |  0.8286  |
| **LSTM** |  **~95.07%**  |  **0.9514**  |


## Error Analysis & Challenges
*  **Sarcasm and Ambiguity**: Misclassifications often occurred in reviews with sarcastic or ambiguous language (e.g., "Great, just great" interpreted as positive when intended negatively).
*  **Complexity**: The LSTM handled longer, complex reviews better than the RNN.
*  **Resource Constraints**: Due to Google Colab runtime limits, hyperparameter tuning (e.g., using Optuna) and L2 regularization were restricted.

## Future Improvements
*  Implement hyperparameter tuning to optimize learning rates and architecture.
*  Explore **Transformer-based models** for better context handling.
*  Incorporate user history or explicit sarcasm labels to improve detection.

## Author
 **Shreshth Vishwakarma** 
IIT BHU Varanasi

## References
*  Dataset and analytical guidance: ChatGPT and CampusX videos.
*  Embeddings: GloVe (Global Vectors for Word Representation).
