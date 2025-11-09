# Toxic Tweet Detection: Deep Learning Multi-Model Comparison

A comprehensive deep learning project implementing and comparing four neural architectures (SimpleRNN, LSTM, GRU, 1D CNN) for detecting offensive speech and hate speech in tweets using pre-trained GloVe embeddings and advanced NLP techniques.

## 🎯 Project Overview

This project addresses the critical challenge of automatic toxic content detection on social media platforms. Implemented multiple deep learning architectures to classify tweets into three categories with high accuracy, demonstrating systematic model comparison and optimization techniques essential for production NLP systems.

**Classification Task:**
- **Class 0:** Hate Speech
- **Class 1:** Offensive Language  
- **Class 2:** Neutral

**Real-World Impact:** Automated content moderation systems for social media platforms, online community safety, and digital wellbeing applications.

## 🛠️ Technical Architecture

### Data Preprocessing Pipeline

**Text Cleaning & Normalization:**
```python
Preprocessing Steps:
├── URL removal (http/https patterns)
├── Mention removal (@username patterns)
├── Special character filtering (preserving apostrophes)
├── Lowercase normalization
├── Stopword removal (NLTK English corpus)
├── Stemming (Porter Stemmer)
└── Short token filtering (< 2 characters)
```

**Tokenization & Sequence Processing:**
- **Tokenizer:** Keras Tokenizer with 10,000 vocabulary size
- **OOV Handling:** Out-of-vocabulary token `<OOV>` for unseen words
- **Sequence Length:** Fixed at 22 tokens (determined via EDA on token distribution)
- **Padding:** Post-padding to ensure uniform input dimensions

**Word Embeddings:**
- **Pre-trained Model:** GloVe 6B (300-dimensional vectors)
- **Vocabulary Coverage:** 10,000 most frequent tokens
- **Initialization:** Words not in GloVe initialized with zero vectors
- **Trainability:** Embeddings fine-tuned during training

### Model Architectures

#### 1. SimpleRNN Model

**Architecture:**
```python
Model: Sequential
├── Embedding(10000, 300, weights=GloVe, trainable=True)
├── SimpleRNN(128 units)
├── Dense(128, activation='relu')
├── Dropout(0.3)
└── Dense(3, activation='softmax')
```

**Characteristics:**
- Basic recurrent architecture for baseline comparison
- Captures short-term dependencies in tweet sequences
- Prone to vanishing gradient on longer sequences
- Fast training time, lower computational cost

**Performance (Validation Set):**
- **Accuracy:** 0.86
- **Precision (macro):** 0.70
- **Recall (macro):** 0.62
- **F1 Score (macro):** 0.65

#### 2. LSTM Model

**Architecture:**
```python
Model: Sequential
├── Embedding(10000, 300, weights=GloVe, trainable=True)
├── LSTM(128 units, return_sequences=True)
├── GlobalMaxPooling1D()
├── Dense(128, activation='relu')
├── Dropout(0.3)
└── Dense(3, activation='softmax')
```

**Characteristics:**
- Long Short-Term Memory gates (input, forget, output)
- Addresses vanishing gradient through cell state mechanism
- Captures long-range dependencies in tweet context
- Better handling of sequential information than SimpleRNN

**Key Features:**
- Cell state preserves information across time steps
- Gate mechanisms control information flow
- Return sequences enabled for max pooling aggregation
- Effective for context-dependent hate speech detection

#### 3. GRU Model

**Architecture:**
```python
Model: Sequential
├── Embedding(10000, 300, weights=GloVe, trainable=True)
├── GRU(128 units, return_sequences=True)
├── GlobalMaxPooling1D()
├── Dense(128, activation='relu')
├── Dropout(0.3)
└── Dense(3, activation='softmax')
```

**Characteristics:**
- Gated Recurrent Unit with update and reset gates
- Simpler than LSTM (fewer parameters, faster training)
- Comparable performance to LSTM on shorter sequences
- Efficient memory usage and computational cost

**Key Features:**
- Update gate controls information retention
- Reset gate manages relevance of past information
- Fewer parameters than LSTM (faster convergence)
- Well-suited for tweet-length sequences

#### 4. 1D CNN Model

**Architecture:**
```python
Model: Sequential
├── Embedding(10000, 300, weights=GloVe, trainable=True)
├── Conv1D(128 filters, kernel_size=5, activation='relu')
├── GlobalMaxPooling1D()
├── Dense(128, activation='relu')
├── Dropout(0.3)
└── Dense(3, activation='softmax')
```

**Characteristics:**
- Convolutional filters extract local n-gram patterns
- No sequential processing (parallel feature extraction)
- Fast training and inference
- Captures position-invariant features (hate phrases anywhere in tweet)

**Key Features:**
- 128 convolutional filters for pattern detection
- Kernel size 5 captures 5-gram linguistic patterns
- Global max pooling selects most salient features
- Effective for identifying toxic keywords and phrases

## 💡 Training Configuration

**Optimization Strategy:**
- **Optimizer:** Adam (Adaptive Moment Estimation)
- **Learning Rate:** 0.0005 (tuned for stability)
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 64 (balances memory and convergence)
- **Epochs:** 10 (with early stopping monitoring)

**Data Split:**
- **Training:** 80% stratified sampling
- **Validation:** 20% stratified sampling
- **Stratification:** Maintains class distribution across splits

**Regularization Techniques:**
- Dropout (0.3) to prevent overfitting
- Embedding layer fine-tuning for domain adaptation
- Early stopping on validation loss (patience=3)

## 📊 Comparative Analysis

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score | Parameters | Training Time |
|-------|----------|-----------|--------|----------|------------|---------------|
| SimpleRNN | 0.86 | 0.70 | 0.62 | 0.65 | ~3.8M | Fastest |
| LSTM | - | - | - | - | ~4.2M | Slower |
| GRU | - | - | - | - | ~4.0M | Medium |
| 1D CNN | - | - | - | - | ~3.9M | Fast |

> *Note: Macro-averaged metrics used to handle class imbalance fairly across all three categories.*

### Evaluation Metrics Rationale

**Macro Averaging:**
- Treats all classes equally regardless of frequency
- Critical for imbalanced datasets (hate speech is minority class)
- Prevents model bias toward majority class (offensive/neutral)

**Metrics Interpretation:**
- **Precision:** Of tweets classified as hate speech, what % are actually hate speech?
- **Recall:** Of all actual hate speech tweets, what % did we detect?
- **F1 Score:** Harmonic mean balancing precision and recall trade-offs

## 🚀 Technical Contributions & Skills

**Deep Learning:**
- Sequential model architectures (RNNs, LSTMs, GRUs, CNNs)
- Recurrent network design for sequential data
- Convolutional architectures for text classification
- Transfer learning with pre-trained embeddings

**Natural Language Processing:**
- Text preprocessing and normalization pipelines
- Tokenization and vocabulary management
- Word embedding integration (GloVe)
- Sequence padding and handling variable-length inputs

**Machine Learning Engineering:**
- Multi-model experimentation and comparison
- Hyperparameter tuning (learning rate, batch size, dropout)
- Regularization techniques for generalization
- Stratified sampling for class balance

**Model Evaluation:**
- Macro-averaged multi-class metrics
- Handling imbalanced classification problems
- Validation set performance analysis
- Kaggle competition submission and ranking

## 🌍 Real-World Applications

### Content Moderation Systems

**Social Media Platforms:**
- Automatic flagging of toxic tweets for review
- Real-time hate speech detection at scale
- User safety and community guidelines enforcement
- Reduced manual moderation workload

**Use Cases:**
- Twitter/X content filtering
- Reddit comment moderation
- Online gaming chat monitoring
- Forum and community platform safety

### Ethical Considerations

**Challenges Addressed:**
- Context-dependent language interpretation
- Sarcasm and irony detection difficulties
- Cultural and linguistic nuances
- False positive minimization (free speech concerns)

**Limitations Acknowledged:**
- Class imbalance in training data
- Potential bias in offensive language definition
- Need for continuous model updates (evolving language)
- Importance of human oversight in content decisions

## 📁 Project Structure
```
toxic-tweet-detection/
├── RNN/
│   ├── results_SimpleRNN.csv
│   ├── SimpleRNN_metrics.csv
│   └── README.md
├── LSTM/
│   ├── results_LSTM.csv
│   ├── LSTM_metrics.csv
│   └── README.md
├── GRU/
│   ├── results_GRU.csv
│   ├── GRU_metrics.csv
│   └── README.md
├── 1D_CNN/
│   ├── results_CNN.csv
│   ├── CNN_metrics.csv
│   └── README.md
└── README.md (this file)
```

**Files Description:**
- **results_*.csv:** Kaggle test set predictions (ID, predicted_class)
- **metrics_*.csv:** Validation performance metrics
- **README.md:** Model-specific architecture and training details

## 🔧 Implementation Details

### Dependencies
```python
Core Libraries:
├── TensorFlow/Keras 2.x - Deep learning framework
├── NumPy - Numerical computing
├── Pandas - Data manipulation
├── NLTK - Natural language toolkit
├── Scikit-learn - ML utilities and metrics
└── GloVe embeddings - Pre-trained word vectors
```

### Preprocessing Code Structure
```python
def preprocess_text(text):
    # URL removal
    text = re.sub(r'http\S+|www\S+', '', text)
    # Mention removal
    text = re.sub(r'@\w+', '', text)
    # Punctuation cleaning
    text = re.sub(r'[^\w\s\']', '', text)
    # Lowercasing
    text = text.lower()
    # Stemming
    tokens = [stemmer.stem(word) for word in text.split()]
    # Stopword removal
    tokens = [w for w in tokens if w not in stopwords]
    return ' '.join(tokens)
```

## 🎯 Future Enhancements

**Model Improvements:**
- Bidirectional LSTM/GRU for context from both directions
- Attention mechanisms for interpretability
- Transformer-based models (BERT, RoBERTa fine-tuning)
- Ensemble methods combining multiple architectures

**Data Augmentation:**
- Back-translation for synthetic training examples
- Contextual word substitution
- Adversarial training examples
- Active learning for hard negatives

**Feature Engineering:**
- Character-level embeddings for misspellings
- Emoji and emoticon encoding
- User metadata features
- Temporal and engagement features

## 📄 Competition Details

**Kaggle Competition:** Offensive Speech Detection - Identifying Toxic Tweets

**Submission Format:**
- CSV file with columns: `id`, `predicted_class`
- Predictions on held-out test set
- Evaluated on macro F1 score

**Competition Insights:**
- Learned importance of class balance handling
- Experimented with multiple architectures systematically
- Understood trade-offs between model complexity and performance
- Gained experience with Kaggle submission and leaderboard dynamics

## 🎓 Project Context

Developed as part of advanced deep learning coursework, demonstrating practical application of sequential and convolutional neural networks for NLP classification tasks. The project showcases systematic model experimentation, rigorous evaluation methodologies, and real-world problem-solving in content moderation - a critical challenge for modern social platforms.

This work highlights end-to-end ML pipeline development: from data preprocessing and embedding integration, through architecture selection and training, to comprehensive multi-model evaluation and comparison - all while addressing the societal impact of toxic online content.

## 🔗 Connect

**Meriem Lmoubariki**
- GitHub: [@myriamlmiii](https://github.com/myriamlmiii)

---

*Applying deep learning to combat online toxicity and promote healthier digital communities.*
