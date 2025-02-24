import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
import streamlit as st
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaSentimentAnalyzer:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            logger.info("Loading default model")
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.model.to(self.device)

    def predict_sentiment(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)
        
        score = predictions[0].tolist()
        sentiment_score = sum([(i + 1) * score[i] for i in range(len(score))]) / 5
        
        if sentiment_score >= 0.6:
            return "positive"
        elif sentiment_score <= 0.4:
            return "negative"
        else:
            return "neutral"
    
    def save_model(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

def model_response(sentiment: str) -> str:
    responses = {
        "positive": [
            "I'm glad you're feeling positive. How else can I help you?",
            "That's great to hear! Would you like to share more?",
            "It's wonderful that you're feeling this way. Let's continue our conversation."
        ],
        "negative": [
            "I understand this is difficult. Would you like to talk more about it?",
            "I'm here to listen. Could you tell me more about what's troubling you?",
            "It's okay to feel this way. Would you like to explore these feelings together?"
        ],
        "neutral": [
            "Could you tell me more about that?",
            "How does that make you feel?",
            "Let's explore this further. What are your thoughts?"
        ]
    }
    
    import random
    response = random.choice(responses[sentiment])
    return response

# Streamlit 애플리케이션
def main():
    st.title('Infiheal Chatbot')
    st.write('Welcome to Infiheal. What problem are you facing?')

    # 모델 로드
    model_path = "./saved_model"
    analyzer = LlamaSentimentAnalyzer(model_path=model_path)

    message = st.text_area("You: ", key="user_input")
    if st.button("Send"):
        if message:
            sentiment = analyzer.predict_sentiment(message)
            st.write(f'Sentiment: {sentiment}')
            
            response = model_response(sentiment)
            st.write(f'Chatbot: {response}')

            if sentiment == 'negative':
                st.session_state['negative_count'] = st.session_state.get('negative_count', 0) + 1

        if message.lower() == "quit":
            negative_count = st.session_state.get('negative_count', 0)
            if negative_count > 5:
                st.write('High Risk: You will be contacted to the nearest distress helpline soon')
            elif negative_count > 3:
                st.write('Moderate Risk: We recommend you connecting our therapist at +91 00000 0000')
            else:
                st.write('Low Risk: Team Infiheal wishes you good health, stay healthy')
            analyzer.save_model(model_path)
            st.stop()

if __name__ == "__main__":
    main()
