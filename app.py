import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# API Configuration for NewsAPI
NEWS_API_KEY = "1272ac9cec4e43108bd69ffd1dc231cb"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Function to Fetch NewsAPI Data
@st.cache_data(ttl=86400)  # Cache data for 24 hours
def fetch_newsapi_data():
    """
    Fetch Saudi stock-related news using NewsAPI, including Argaam and Mubasher as domains.
    """
    params = {
        "q": "Tadawul OR TASI OR 'Saudi stocks'",
        "domains": "english.mubasher.info,argaam.com",
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
        "from": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    }
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from NewsAPI: {e}")
        return []

# Function to Scrape Argaam
def scrape_argaam():
    """
    Scrape latest Saudi stock market news from Argaam.
    """
    url = "https://www.argaam.com/en"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.select(".latest-articles .item a")
        news_list = []

        for article in articles[:10]:  # Limit to 10 articles
            title = article.get_text(strip=True)
            link = "https://www.argaam.com" + article["href"]
            sentiment = analyzer.polarity_scores(title)
            news_list.append({
                "Title": title,
                "Description": "No description available.",
                "URL": link,
                "Positive": sentiment["pos"],
                "Neutral": sentiment["neu"],
                "Negative": sentiment["neg"],
                "Compound": sentiment["compound"]
            })
        return news_list
    except Exception as e:
        st.error(f"Error scraping Argaam: {e}")
        return []

# Function to Scrape Mubasher
def scrape_mubasher():
    """
    Scrape latest Saudi market news from Mubasher.
    """
    url = "https://english.mubasher.info/news/sa/now/latest"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.select(".content div.media-body h2 a")
        news_list = []

        for article in articles[:10]:  # Limit to 10 articles
            title = article.get_text(strip=True)
            link = article["href"]
            sentiment = analyzer.polarity_scores(title)
            news_list.append({
                "Title": title,
                "Description": "No description available.",
                "URL": link,
                "Positive": sentiment["pos"],
                "Neutral": sentiment["neu"],
                "Negative": sentiment["neg"],
                "Compound": sentiment["compound"]
            })
        return news_list
    except Exception as e:
        st.error(f"Error scraping Mubasher: {e}")
        return []

@st.cache_data(ttl=86400)  # Cache the combined news for 1 day
def fetch_all_news():
    """
    Fetch and combine news from NewsAPI (global sources), Argaam, and Mubasher.
    """
    api_news = fetch_newsapi_data()
    argaam_news = scrape_argaam()
    mubasher_news = scrape_mubasher()

    all_news = []

    # Process NewsAPI articles
    for article in api_news:
        title = article['title']
        description = article.get('description', "No description available.")
        url = article['url']
        sentiment = analyzer.polarity_scores(title)
        
        all_news.append({
            "Title": title,
            "Description": description,
            "URL": url,
            "Positive": sentiment["pos"],
            "Neutral": sentiment["neu"],
            "Negative": sentiment["neg"],
            "Compound": sentiment["compound"]
        })
    
    # Combine Argaam and Mubasher with NewsAPI articles
    all_news += argaam_news + mubasher_news
    return all_news

# Streamlit Application
st.title("Saudi Stock Market News & Sentiment Analysis")
st.write("Sentiment analysis of recent Saudi stock market news (updated daily).")

st.info("Fetching latest news from multiple sources...")
news_data = fetch_all_news()

if news_data:
    # Create DataFrame to display news
    df = pd.DataFrame(news_data)

    # Display all news
    st.subheader("All News")
    st.dataframe(df)

    # Display top positive news
    st.subheader("Top Positive News")
    positive_news = df[df["Compound"] > 0.5]
    st.dataframe(positive_news)

    # Display top negative news
    st.subheader("Top Negative News")
    negative_news = df[df["Compound"] < -0.5]
    st.dataframe(negative_news)

else:
    st.warning("No news articles found.")
