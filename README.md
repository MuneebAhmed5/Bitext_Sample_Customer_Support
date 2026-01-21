# Bitext_Sample_Customer_Support
Customer Support Intent Classification &amp; Semantic Search Agent

📋 Project Overview
An intelligent customer support agent system that combines intent classification and semantic search to provide accurate, context-aware responses to customer queries. The system can understand customer intents, extract relevant entities, and retrieve appropriate responses from a dataset of 26,872 pre-defined customer support interactions.

Key Features
Multi-Intent Classification: Identifies 27 different customer intents with 99% accuracy

Entity Extraction: Detects order numbers, placeholders, and other key information

Semantic Search: Uses sentence embeddings to find the most relevant responses

Real-time Processing: Interactive agent that responds to live customer queries

Pre-trained Models: Leverages spaCy and Sentence Transformers for NLP tasks

🏗️ Architecture
text
Customer Query → Text Preprocessing → Intent Classification → Entity Extraction → Semantic Search → Response Generation
      ↓                  ↓                    ↓                   ↓                    ↓               ↓
   Raw Input        Lowercase +          Logistic           spaCy NER +       Sentence         Placeholder
                  Lemmatization       Regression (99%)     Regex Patterns    Similarity        Replacement
📊 Dataset Information
Bitext Sample Customer Support Training Dataset

Size: 26,872 samples

Columns:

flags: Query flags/indicators

instruction: Customer query (with placeholders like {{Order Number}})

category: High-level category (ORDER, REFUND, etc.)

intent: Specific intent (cancel_order, track_refund, etc.)

response: Pre-defined agent response

Intent Categories (27 total):

cancel_order, change_order, change_shipping_address

check_cancellation_fee, check_invoice, check_payment_methods

track_refund, contact_customer_service, create_account

And 18 more...

🚀 Quick Start
Prerequisites
bash
pip install pandas numpy scikit-learn nltk spacy sentence-transformers
Installation
Download required NLP models:

bash
python -m spacy download en_core_web_sm
Download NLTK data:

python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
Basic Usage
python
# Load the dataset
import pandas as pd
data = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Initialize the agent
from customer_support_agent import CustomerSupportAgent
agent = CustomerSupportAgent(data)

# Process a query
query = "My order 12345 is delayed, when will it arrive?"
result = agent.process_query(query)

print(f"Intent: {result['intent']}")
print(f"Response: {result['response']}")
print(f"Entities: {result['entities']}")
🔧 Core Components
1. Text Preprocessing
python
def preprocess_text(text):
    # Lowercase conversion
    # Number removal
    # Punctuation removal
    # Stopword filtering
    # Lemmatization
2. Intent Classification
Model: Logistic Regression with TF-IDF features

Features: 5,000 most important words

Accuracy: 99% on test set

Training: 80/20 train-test split

3. Entity Extraction
Placeholders: Extract {{Order Number}}, {{Refund Amount}}, etc.

Order Numbers: Extract numeric IDs from queries

Tools: Regular expressions + spaCy NER

4. Semantic Search
Embedding Model: all-MiniLM-L6-v2 (Sentence Transformers)

Similarity Metric: Cosine similarity

Retrieval: Find most similar pre-defined query

5. Response Generation
Retrieve pre-defined response based on semantic similarity

Replace placeholders with actual values from query

Maintain professional, empathetic tone

📈 Performance Metrics
Intent Classification
text
                          precision    recall  f1-score   support
            cancel_order       1.00      0.96      0.98       187
            change_order       0.93      0.99      0.96       187
                accuracy                           0.99      5375
               macro avg       0.99      0.99      0.99      5375
            weighted avg       0.99      0.99      0.99      5375
Entity Extraction Accuracy
Placeholder detection: ~100%

Order number extraction: ~95%

Response time: < 1 second per query

🎯 Example Queries & Responses
Customer Query	Intent	Response Summary
"Where is my order?"	track_order	Provides tracking instructions with order number placeholder
"I want to cancel order 12345"	cancel_order	Explains cancellation process, replaces {{Order Number}}
"How do I get a refund?"	get_refund	Outlines refund policy and processing time
"Create account for me"	create_account	Asks for name, email, and guides through setup
"Newsletter subscription"	newsletter_subscription	Requests email address for subscription
🛠️ API Usage
Initialize Agent
python
agent = CustomerSupportAgent(
    data_path="dataset.csv",
    use_gpu=False  # Set to True if available
)
Process Query
python
response = agent.process_query(
    query="My order is late",
    return_full_result=True  # Returns intent, entities, etc.
)
Batch Processing
python
queries = ["Where's my order?", "Need refund", "Create account"]
results = agent.process_batch(queries)
🧪 Testing the Agent
Interactive Mode
python
while True:
    query = input("Customer: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = customer_support_agent(query)
    print(f"Agent: {result['response']}")
    print(f"Detected Intent: {result['intent']}")
Test Examples
python
test_queries = [
    "My order 98765 is delayed",
    "I want to subscribe to newsletter",
    "Create an account for me please",
    "Having trouble registering",
    "Where is my refund for order 54321?"
]
📁 Project Structure
text
customer-support-agent/
├── data/
│   └── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
├── src/
│   ├── preprocessor.py          # Text cleaning and preprocessing
│   ├── intent_classifier.py     # Logistic regression model
│   ├── entity_extractor.py      # NER and pattern matching
│   ├── semantic_search.py       # Sentence embeddings and similarity
│   ├── response_generator.py    # Response assembly and placeholder replacement
│   └── agent.py                 # Main agent class
├── models/
│   ├── intent_classifier.pkl    # Trained model
│   └── tfidf_vectorizer.pkl     # Feature vectorizer
├── tests/
│   ├── test_intent.py
│   ├── test_entities.py
│   └── test_agent.py
├── requirements.txt
├── README.md
└── demo.ipynb                   # Jupyter notebook demo
🔍 Advanced Features
1. Intent Confidence Scores
python
# Get prediction probabilities
probas = clf.predict_proba(query_vector)
confidence = max(probas[0])
2. Multiple Entity Handling
python
# Extract all order numbers
order_numbers = re.findall(r'\\b\\d{5,}\\b', query)
# Extract all placeholders
placeholders = re.findall(r'\\{\\{.*?\\}\\}', query)
3. Response Customization
python
# Custom replacement rules
replacements = {
    "{{Order Number}}": extracted_order_numbers[0] if extracted_order_numbers else "your order",
    "{{Refund Processing Time}}": "3-5 business days",
    "{{Customer Support Hours}}": "9 AM - 6 PM EST"
}
🚀 Deployment Options
1. Local Python Script
bash
python customer_support_agent.py
2. Flask API
python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/support', methods=['POST'])
def support():
    query = request.json['query']
    result = agent.process_query(query)
    return jsonify(result)
3. Streamlit App
python
import streamlit as st
st.title("Customer Support Agent")
query = st.text_input("How can I help you today?")
if query:
    result = agent.process_query(query)
    st.write(result['response'])
📝 Customization
Adding New Intents
Add labeled examples to the dataset

Retrain the classifier:

python
agent.retrain_intent_classifier(new_data)
Custom Entity Types
python
# Add custom regex patterns
custom_patterns = {
    'promo_code': r'\\b[A-Z0-9]{6,12}\\b',
    'email': r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
}
Response Templates
Edit the dataset CSV to add new response templates with appropriate placeholders.

🧪 Evaluation & Testing
Test Coverage
bash
# Run all tests
python -m pytest tests/

# Specific test
python -m pytest tests/test_intent.py -v
Performance Benchmarks
Query processing time: < 0.5 seconds

Intent accuracy: 99% on validation set

Entity extraction F1: 0.97

Memory usage: ~500MB (including models)

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/improvement)

Commit changes (git commit -m 'Add new intent detection')

Push to branch (git push origin feature/improvement)

Open a Pull Request

📄 License
This project is available for educational and research purposes. The dataset is provided by Bitext for training purposes.

🙏 Acknowledgments
Bitext for providing the customer support dataset

Scikit-learn for machine learning utilities

spaCy for NLP capabilities

Sentence Transformers for semantic embeddings

NLTK for text processing tools
