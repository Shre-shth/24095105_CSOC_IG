# Machine Learning & Deep Learning Portfolio

This repository contains a collection of projects exploring various domains of Machine Learning and Deep Learning, including Natural Language Processing (NLP) and Predictive Analytics.

---

## 1. Machine Translation Model Comparison
**Objective:** A comparative study of Neural Machine Translation (NMT) architectures.

### Architectures Explored
* **Vanilla Encoder-Decoder**: A baseline sequence-to-sequence model using single-layered LSTM units.
* **Encoder-Decoder with Luong Attention**: Implements an attention mechanism to dynamically align decoder outputs with relevant encoder states.
* **Transformer Model**: Planned architecture using self-attention; implementation was limited by hardware and time constraints.

### Technical Implementation
* **Optimization**: Adam optimizer with a **0.5 Teacher Forcing ratio**.
* **Efficiency**: Utilized Mixed Precision (`torch.cuda.amp`) and dynamic padding to minimize wasted tokens.
* **Challenges**: Training was constrained by Google Colab RAM limits and restricted usage time.

---

## 2. Sequence-Based Sentiment Analysis
**Objective:** Predict sentiment polarity (Positive/Negative) from Amazon product reviews using RNN and LSTM models.

### Model Comparison
| Metric | RNN Model (2-layer Bidirectional) | LSTM Model (2-layer Bidirectional) |
| :--- | :--- | :--- |
| **Test Accuracy** | ~81.70% | **~95.07%** |
| **Test F1-Score** | 0.8286 | **0.9514** |

### Key Features
* **Embeddings**: Utilized pre-trained **100-dimensional GloVe embeddings**.
* **Preprocessing**: Text cleaning (non-alphabetic removal) and tokenization with a 30,000-word vocabulary limit.
* **Observations**: LSTMs significantly outperformed standard RNNs in capturing long-range dependencies and complex sentiments like sarcasm.

---

## 3. Medical Appointment No-Show Prediction
**Objective:** Compare a from-scratch NumPy implementation against a PyTorch framework for binary classification.

### Performance & Resource Usage
| Implementation | Training Time (s) | Memory Usage | Accuracy |
| :--- | :--- | :--- | :--- |
| **NumPy (From Scratch)** | 613.70s | **0.56 MB** | 62.47% |
| **PyTorch** | **21.70s** | 13.72 MB | **62.94%** |

### Methodology
* **Feature Engineering**: Created `DaysAdvance` and one-hot encoded temporal features (Day of Week, Hour).
* **Imbalance Handling**: Used **Stratified Split** and **Cost-Sensitive Learning** to give proper weight to minority "no-show" cases.
* **Optimization**: NumPy used Mini-Batch Gradient Descent, while PyTorch leveraged the Adam optimizer and GPU acceleration.

---

## Summary of Tools & Technologies
* **Frameworks**: PyTorch, NumPy, Scikit-Learn.
* **Models**: LSTM, RNN, Seq2Seq, Feed-Forward Neural Networks.
* **Techniques**: Attention Mechanisms, Mixed Precision Training, Feature Engineering, Word Embeddings (GloVe).
