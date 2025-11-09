# SimpleRNN Model - Tweet Classification

## Overview

This model uses a Simple Recurrent Neural Network (SimpleRNN) architecture to classify tweets into one of the following three classes:
- **0**: Hate speech  
- **1**: Offensive language  
- **2**: Neutral

We incorporated pre-trained GloVe embeddings to generate semantic word representations and trained the model on cleaned and tokenized tweets.

---

## Model Architecture

- **Embedding Layer**
  - Input dim: 10,000  
  - Output dim: 300 (GloVe 300d)  
  - Weights: Loaded from pre-trained GloVe vectors  
  - Trainable: Yes

- **SimpleRNN Layer**
  - Units: 128

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

- URL, mention, and punctuation removal  
- Lowercasing and normalization of contractions  
- Stemming using PorterStemmer  
- Stopword filtering and token cleanup  
- Tokenization via Keras Tokenizer  
- Padding to a fixed max length of 22  
- Use of `oov_token` for unseen vocabulary  
- Embedding with GloVe 300d vectors

---

## Evaluation Results (Validation Set)

- **Accuracy**: 0.86  
- **Precision (macro)**: 0.70  
- **Recall (macro)**: 0.62  
- **F1 Score (macro)**: 0.65  

> *Macro averaging* ensures all classes are evaluated equally, regardless of class imbalance.
   All metrics were saved in a CSV file: `LSTM_metrics.csv`

---

## Files Included

- `results_SimpleRNN.csv`: Predictions on the test dataset  
- `SimpleRNN_metrics.csv`: Evaluation metrics  

