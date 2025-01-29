import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Set up NewsAPI Configuration
API_KEY = "1272ac9cec4e43108bd69ffd1dc231cb"
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_newsapi_data():
    """Fetch news from NewsAPI focused on Saudi market."""
    params = {
        "q": "Tadawul OR TASI OR 'Saudi stocks'",
        "domains": "arabnews.com,saudigazette.com.sa",
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": API_KEY,
        "from": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),  # Last 24 hours
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        st.error("Failed to fetch NewsAPI data.")
        return []

def scrape_argaam():
    """Scrape Argaam's latest Saudi stock market news."""
    url = "https://www.argaam.com/en"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.select(".latest-articles .item a")

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
        st.error(f"Failed to scrape Argaam: {e}")
        return []

def scrape_mubasher():
    """Scrape Mubasher for the latest Saudi market news."""
    url = "https://english.mubasher.info/news/sa/now/latest"
    try:
        response = requests.get(url)
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
        st.error(f"Failed to scrape Mubasher: {e}")
        return []

@st.cache_data(ttl=86400)  # Cache for 1 day (24 hours)
def fetch_all_news():
    """Fetch and combine news from all sources."""
    # Fetch NewsAPI Data
    api_news = fetch_newsapi_data()

    # Scrape Argaam and Mubasher
    argaam_news = scrape_argaam()
    mubasher_news = scrape_mubasher()

    # Combine all news
    news_data = []
    for article in api_news:
        sentiment = analyzer.polarity_scores(article['title'])
        news_data.append({
            "Title": article['title'],
            "Description": article.get('description', "No description available."),
            "URL": article['url'],
            "Positive": sentiment["pos"],
            "Neutral": sentiment["neu"],
            "Negative": sentiment["neg"],
            "Compound": sentiment["compound"]
        })

    return news_data + argaam_news + mubasher_news

# Main Dashboard
st.title("Saudi Stock Market News & Sentiment Analysis")
st.write("Sentiment analysis of recent Saudi stock market news (updated daily).")

news = fetch_all_news()

# Display the news
if news:
    df = pd.DataFrame(news)

    # Show All News
    st.subheader("All News")
    st.dataframe(df)

    # Filtered Views
    st.subheader("Top Positive News")
    st.dataframe(df[df["Compound"] > 0.5])

    st.subheader("Top Negative News")
    st.dataframe(df[df["Compound"] < -0.5])
else:
    st.warning("No news articles found.")
