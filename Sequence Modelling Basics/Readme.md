# Sequence-Based Sentiment Analysis on Amazon Reviews

## Project Objective
[cite_start]The goal of this project is to develop a sequence-based deep learning model—specifically exploring **RNN** and **LSTM** architectures—to predict sentiment polarity (positive or negative) from Amazon product review text[cite: 1, 5]. [cite_start]The model aims to capture emotional cues, subtle tones, and sarcasm beyond simple keyword matching[cite: 6].

## Dataset Summary
[cite_start]The project utilizes the **Amazon Reviews Dataset**[cite: 11].
* [cite_start]**Text**: The full review written by the user[cite: 8].
* [cite_start]**Title**: The title of the review[cite: 9].
* [cite_start]**Polarity**: Labeled as **1 for Negative** and **2 for Positive**[cite: 10].

## Preprocessing Pipeline
To prepare the text for the neural networks, the following steps were implemented:
1.  [cite_start]**Cleaning**: Lowercasing, removing extra spaces, and stripping non-alphabetic characters[cite: 13].
2.  [cite_start]**Tokenization**: Splitting text into individual word tokens[cite: 14].
3.  [cite_start]**Padding/Truncating**: Reviews were post-padded or truncated to a fixed `MAX_LEN` of **256**[cite: 15].
4.  [cite_start]**Vocabulary**: Retained the top **30,000** most frequent words to manage memory constraints[cite: 16].
5.  [cite_start]**Word Embeddings**: Leveraged pre-trained **100-dimensional GloVe embeddings**[cite: 18]. [cite_start]The embedding layer weights were set to non-trainable to preserve pre-trained semantics[cite: 21].

## Model Architectures
Two primary models were built and compared:

| Feature | RNN Model | LSTM Model |
| :--- | :--- | :--- |
| **Type** | [cite_start]2-layer Bidirectional RNN [cite: 25] | [cite_start]2-layer Bidirectional LSTM [cite: 30] |
| **Hidden Size** | [cite_start]128 [cite: 25] | [cite_start]128 [cite: 30] |
| **Regularization** | [cite_start]None (due to resource limits) [cite: 27] | [cite_start]Dropout (0.5) [cite: 30] |
| **Output Layer** | [cite_start]Fully connected for binary output [cite: 26] | [cite_start]Fully connected for binary output [cite: 31] |

### Training Details
* [cite_start]**Loss Function**: Binary Cross-Entropy with Logits (includes built-in Sigmoid)[cite: 33].
* [cite_start]**Optimizer**: Adam with a learning rate of 0.001[cite: 34].
* [cite_start]**Batch Size**: 512 (chosen for faster training per epoch)[cite: 36].
* [cite_start]**Hardware**: Trained for 4 epochs on a **Tesla T4 GPU** via Google Colab[cite: 37, 38].

## Results and Evaluation
[cite_start]The LSTM model significantly outperformed the vanilla RNN in capturing long-range dependencies[cite: 105].

| Model | Test Accuracy | Test F1-Score |
| :--- | :--- | :--- |
| **RNN** | [cite_start]~81.70% [cite: 46] | [cite_start]0.8286 [cite: 47] |
| **LSTM** | [cite_start]**~95.07%** [cite: 68] | [cite_start]**0.9514** [cite: 69] |


## Error Analysis & Challenges
* [cite_start]**Sarcasm and Ambiguity**: Misclassifications often occurred in reviews with sarcastic or ambiguous language (e.g., "Great, just great" interpreted as positive when intended negatively)[cite: 91, 92].
* [cite_start]**Complexity**: The LSTM handled longer, complex reviews better than the RNN[cite: 93].
* [cite_start]**Resource Constraints**: Due to Google Colab runtime limits, hyperparameter tuning (e.g., using Optuna) and L2 regularization were restricted[cite: 27, 101].

## Future Improvements
* [cite_start]Implement hyperparameter tuning to optimize learning rates and architecture[cite: 107].
* [cite_start]Explore **Transformer-based models** for better context handling[cite: 102].
* [cite_start]Incorporate user history or explicit sarcasm labels to improve detection[cite: 102].

## Author
[cite_start]**Shreshth Vishwakarma** [cite: 2]
IIT BHU Varanasi

## References
* [cite_start]Dataset and analytical guidance: ChatGPT and CampusX videos[cite: 109].
* [cite_start]Embeddings: GloVe (Global Vectors for Word Representation)[cite: 109].
