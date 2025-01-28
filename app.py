import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Fetch News
API_KEY = "1272ac9cec4e43108bd69ffd1dc231cb"  # Use Streamlit Secrets for API key
NEWS_API_URL = "https://newsapi.org/v2/everything"

params = {
    "q": "stocks",
    "language": "en",
    "sortBy": "relevancy",
    "apiKey": API_KEY
}

response = requests.get(NEWS_API_URL, params=params)
data = response.json()

# Analyze Sentiment and Store Data
news_data = []
for article in data['articles']:
    title = article['title']
    sentiment = analyzer.polarity_scores(title)
    news_data.append({
        "Title": title,
        "Description": article['description'],
        "URL": article['url'],
        "Positive": sentiment['pos'],
        "Neutral": sentiment['neu'],
        "Negative": sentiment['neg'],
        "Compound": sentiment['compound']
    })

df = pd.DataFrame(news_data)

# Display Dashboard
st.title("Stock Market News Aggregator")
st.write("Sentiment analysis of recent stock news.")

st.dataframe(df)

st.subheader("Top Positive News")
positive_news = df[df["Compound"] > 0.5]
st.dataframe(positive_news)

st.subheader("Top Negative News")
negative_news = df[df["Compound"] < -0.5]
st.dataframe(negative_news)
