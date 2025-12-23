# Medical Appointment No-Show Prediction: NumPy vs. PyTorch

## Project Overview
This project explores two distinct approaches to predicting medical appointment no-shows:
1.  **Vanilla Neural Network from Scratch**: Implemented using **NumPy** for forward and backward propagation.
2.  **PyTorch-based Neural Network**: Built using the **`torch.nn.Module`** framework for optimized performance.

The primary objective is to compare these implementations in terms of convergence speed, performance metrics, and resource utilization using the Medical Appointment No-Show Dataset.

---

## Dataset & Preprocessing
The dataset contains **110,527 records** with 14 features, including patient demographics and appointment details.

### Preprocessing Steps
* **Feature Engineering**: Created `DaysAdvance` (days between booking and appointment), `Scheduled DayOfWeek`, `Appointment DayOfWeek`, and `Scheduled Hour`.
* **Standardization**: Applied `StandardScaler` to `Age` and `DaysAdvance`.
* **Encoding**: 
    * Used **Label Encoding** for binary categorical variables (e.g., gender).
    * Used **One-Hot Encoding** for cyclical features like `Scheduled DayOfWeek` and `Scheduled Hour` to ensure the network treats categories equally.
* **Stratified Split**: Distributed data into an 80% training set and a 20% cross-validation set, ensuring similar proportions of no-show cases in both.

---

## Methodology & Architecture
Both models share an identical architecture to ensure a fair comparison:
* **Input Layer**: 38 features.
* **Hidden Layer**: 8 units with **ReLU** activation.
* **Output Layer**: 1 unit with **Sigmoid** activation.
* **Loss Function**: Binary Cross-Entropy with **Cost-Sensitive Learning** to handle class imbalance.

### Implementation Details
* **NumPy Model**: Uses Mini-Batch Gradient Descent for optimization.
* **PyTorch Model**: Uses the **Adam** optimizer and leverages **Autograd** for automatic differentiation.
* **Regularization**: Both models incorporate a regularization constant of **0.01**.
* **Decision Thresholding**: Instead of a standard 0.5 threshold, a **tuned decision threshold** was used to maximize the F1-score.

---

## Results and Performance
The models were evaluated over 3,000 iterations with a learning rate of 0.01.

### Efficiency Comparison
| Metric | NumPy (From Scratch) | PyTorch |
| :--- | :--- | :--- |
| **Training Time** | 613.70 seconds | **21.70 seconds** |
| **Memory Usage** | 0.56 MB | 13.72 MB |



### Validation Metrics
| Metric | NumPy (From Scratch) | PyTorch |
| :--- | :--- | :--- |
| **Accuracy** | 62.46% | **62.94%** |
| **F1-Score** | 0.6616 | **0.6659** |
| **PR-AUC** | 0.3501 | 0.3436 |



---

## Analysis & Conclusion
* **Speed**: The PyTorch implementation is significantly faster due to **GPU acceleration**, compute optimization, and efficient gradient calculation via Autograd.
* **Stability**: The PyTorch cost curve is much smoother than the NumPy curve, as the latter used Mini-Batch Gradient Descent which introduces more noise.
* **Weighting**: Cost-sensitive learning was critical for this imbalanced dataset to ensure the minority "no-show" cases were given proper weight during training.

## References
* **Andrew Ng’s Deep Learning Specialization**: Concepts for the from-scratch implementation.
