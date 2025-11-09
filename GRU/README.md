# GRU Model - Tweet Classification

## Overview

This model uses a Gated Recurrent Unit (GRU)-based deep learning architecture to classify tweets into three categories:
- **0**: Hate speech  
- **1**: Offensive language  
- **2**: Neutral

Pre-trained GloVe embeddings were used to convert tokens into dense vector representations. The model was trained on preprocessed tweets using stratified train-validation splits.

---

## Model Architecture

- **Embedding Layer**
  - Input dim: 10,000  
  - Output dim: 300 (GloVe 300d)  
  - Weights: Preloaded GloVe vectors  
  - Trainable: Yes

- **GRU Layer**
  - Units: 128  
  - Return sequences: True

- **Global Max Pooling Layer**

- **Dense Layer**
  - Units: 128  
  - Activation: ReLU

- **Dropout Layer**
  - Rate: 0.3

- **Output Layer**
  - Units: 3  
  - Activation: Softmax

- **Optimizer**: Adam  
  - Learning rate: 0.0005  
- **Loss Function**: Categorical Crossentropy  
- **Batch Size**: 64  
- **Epochs**: 10  

---

## Preprocessing Techniques Applied

- Removal of URLs, mentions, punctuation, and short tokens  
- Conversion to lowercase  
- Stemming using PorterStemmer  
- Tokenization using Keras Tokenizer  
- Padding to a max length of 22 tokens  
- Usage of `oov_token` for unknown words  
- GloVe embedding matrix integration

---

## Evaluation Results (Validation Set)

- **Accuracy**  
- **Precision (macro)**
- **Recall (macro)** 
- **F1 Score (macro)**

> All metrics are **macro-averaged** across the 3 classes to handle class imbalance fairly.
  All metrics were saved in a CSV file: `GRU_metrics.csv`

---

## Files Included

- `results_GRU.csv`: Predictions on the test set
- `GRU_metrics.csv`: Evaluation metrics (Accuracy, Precision, Recall, F1 Score)


