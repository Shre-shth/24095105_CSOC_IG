Machine Translation Model Comparison
Project Overview
This project focuses on a comparative study of different architectures for Neural Machine Translation (NMT). The goal was to implement and evaluate three distinct approaches: a basic Encoder-Decoder, a model with Attention mechanisms, and a Transformer-based architecture.

Models and Approach
1. Encoder-Decoder (Vanilla LSTM)
Architecture: A basic sequence-to-sequence (Seq2Seq) framework using a single-layered LSTM.

Functionality: This serves as the baseline model, mapping input sequences to a fixed-length vector before decoding them into the target language.

2. Encoder-Decoder with Luong Attention
Architecture: This model extends the standard Seq2Seq framework by adding an attention mechanism.

Mechanism: It uses Luong Attention to dynamically align decoder output with relevant encoder states, improving the model's ability to focus on specific parts of the input sequence.

3. Transformer Model
Status: The Transformer model task was not completed due to significant time and hardware limitations.

Intended Approach: The goal was to implement a model that replaces recurrence with self-attention mechanisms to allow for better parallelization and capturing of long-range dependencies.

Training Methodology
Loss Function: Cross-Entropy Loss.

Optimizer: Adam.

Teacher Forcing: Implemented with a ratio of 0.5, meaning the ground-truth token is fed to the decoder half of the time.

Optimization: Utilized Mixed Precision training with torch.cuda.amp.autocast() and GradScaler to optimize GPU performance.

Batching: A batch size of 8 was used with dynamic padding to ensure minimal wasted tokens per batch.

Challenges and Constraints
The project faced several constraints that impacted full training and completion:

Hardware Limitations: Persistent training problems occurred on Google Colab due to restricted RAM and usage time limitations.

Data Constraints: A smaller number of sequences were used for training as the process was highly time-consuming.

Time Management: Prior commitments and time limitations prevented the full improvement of the models and the completion of the bonus tasks.

Future Improvement Strategies
To reach the performance levels of state-of-the-art models, the following strategies were identified for future work:

Architecture Depth: Transitioning from a single layer to a 4-layer stacked LSTM architecture.

Bidirectional LSTMs: Implementing bidirectional LSTMs in the encoder to capture context from both past and future elements.

Data & Epochs: Increasing the training dataset size and the number of training epochs to allow the model to learn more complex linguistic patterns.

