# README: 1D CNN Model for Tweet Classification

## Overview

This document describes the implementation details of the 1D Convolutional Neural Network (CNN) used for classifying tweets into three categories as part of a toxic tweet classification task:

- **0**: Hate Speech  
- **1**: Offensive Language  
- **2**: Neutral

---

## Model Architecture

The CNN model was built using TensorFlow and Keras. Here's a summary of the architecture:

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model.add(Conv1D(128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
```

### Explanation of Layers:
- **Embedding Layer**: Initialized using GloVe 6B (300-dimensional) pre-trained embeddings. Vocabulary size is capped at 10,000. Embedding weights are trainable.
- **Conv1D**: Applies 128 filters with a kernel size of 5 to extract local n-gram patterns.
- **GlobalMaxPooling1D**: Selects the most important feature from each filter map.
- **Dense Layer**: Fully connected layer with 128 units and ReLU activation.
- **Dropout**: 30% dropout rate to prevent overfitting.
- **Output Layer**: 3 neurons with softmax activation for multi-class classification.

---

## Techniques Applied

- **Text Preprocessing**:
  - Removal of URLs, mentions, and punctuation (except apostrophes)
  - Lowercasing
  - Stopwords removal
  - Stemming (using NLTK’s PorterStemmer)

- **Tokenization**: Keras `Tokenizer` with a vocabulary of 10,000 most frequent tokens. Out-of-vocabulary token used: `<OOV>`.

- **Padding**: Tweets were padded to a fixed length of `22` tokens, determined after EDA on token distribution.

- **Embeddings**: Used `glove.6B.300d.txt` for initializing the embedding matrix. Words not found in GloVe were initialized with zeros.

- **Train/Validation Split**: 
  - Stratified 80/20 split using `train_test_split` to maintain class balance.

- **Loss Function**: `categorical_crossentropy` (labels were one-hot encoded using `to_categorical()`).

- **Optimizer**: Adam with a learning rate of `0.0005`.

- **Epochs**: 10  
- **Batch Size**: 64

---

## Evaluation Metrics

The model was evaluated on the validation set using the following metrics:
- **Accuracy**
- **Precision** (macro)
- **Recall** (macro)
- **F1 Score** (macro)


> All metrics are **macro-averaged** across the 3 classes to handle class imbalance fairly.
  All metrics were saved in a CSV file: `CNN_metrics.csv`

---

## Files Included

- `results_CNN.csv`: Predictions on the test set
- `CNN_metrics.csv`: Evaluation metrics (Accuracy, Precision, Recall, F1 Score)




