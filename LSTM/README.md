# LSTM Model - Tweet Classification

## Overview

This model utilizes a Long Short-Term Memory (LSTM)-based deep learning architecture to classify tweets into the following three classes:
- **0**: Hate speech  
- **1**: Offensive language  
- **2**: Neutral

The model is powered by pre-trained GloVe embeddings to represent words in a dense vector space and trained using preprocessed tweets.

---

## Model Architecture

- **Embedding Layer**
  - Input dim: 10,000  
  - Output dim: 300 (GloVe 300d)  
  - Weights: Loaded from pre-trained GloVe vectors  
  - Trainable: Yes

- **LSTM Layer**
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

- Removal of URLs, mentions, and special characters  
- Lowercasing of text  
- Stemming using PorterStemmer  
- Removal of stopwords and short tokens  
- Tokenization using Keras Tokenizer  
- Sequence padding to fixed max length of 22  
- Use of `oov_token` to handle out-of-vocab words  
- Integration of GloVe 300d embeddings

---

## Evaluation Results (Validation Set)

- **Accuracy**  
- **Precision (macro)** 
- **Recall (macro)**
- **F1 Score (macro)** 

> *Macro averaging* was used to ensure fair evaluation across all three classes despite class imbalance.
  All metrics were saved in a CSV file: `LSTM_metrics.csv`
---

## Files Included

- `results_LSTM.csv`: Model predictions on the test dataset  
- `LSTM_metrics.csv`: Model evaluation metrics  

