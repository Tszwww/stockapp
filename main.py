import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import requests
import praw
from transformers import BertTokenizer, BertForSequenceClassification
from bs4 import BeautifulSoup
from datetime import datetime
from numerize import numerize
import torch

# Initialize Reddit with your credentials
reddit = praw.Reddit(
    client_id="n8vR7uQfzJSCunu4rup13g",
    client_secret="oGkKXPOISjrUI5YW3dG5kc8BpsoGlw",
    user_agent="TradingAPPwS",
)

# Function to load and cache the model and tokenizer
@st.cache_data
def load_finbert(model_name="ProsusAI/finbert"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Function to run sentiment analysis using FinBERT
def run_finbert_inference(text):
    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_probability = probabilities[0][predicted_class].item()

    sentiment_labels = ["positive", "negative", "neutral"]
    predicted_sentiment = sentiment_labels[predicted_class]

    return predicted_sentiment, predicted_probability

def extract_posts(symbol, subreddit_names, sort, limit):
    all_posts = []
    for subreddit_name in subreddit_names:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            posts = subreddit.search(f"title:{symbol}", sort=sort, limit=limit)
            for post in posts:
                created_time = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                post_info = {
                    'Text': str(post.selftext),
                    'Title': str(post.title),
                    'Subreddit': str(post.subreddit),
                    'URL': post.url,
                    'Created_UTC': created_time,
                }
                all_posts.append(post_info)
        except Exception as e:
            return str(e)
    return all_posts


def get_stock_news(ticker):
    exchanges = ["NASDAQ", "NYSE", "LSE", "OTCMKTS"]
    news_list = []

    for exchange in exchanges:
        url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        news_elements = soup.select(".yY3Lee")

        if news_elements:
            for element in news_elements:
                title_element = element.select_one(".Yfwt5")
                title = title_element.text.strip()
                source_element = element.select_one(".sfyJob")
                source = source_element.text.strip()
                date_element = element.select_one(".Adak")
                date = date_element.text.strip()
                url_element = element.select_one("a")
                url = url_element["href"]

                news_item = {
                    "title": title,
                    "source": source,
                    "date": date,
                    "url": url,
                }
                news_list.append(news_item)
            break  # Exit the loop if news items are found for the current exchange
        news_container = soup.select_one(".rv7pxc.m127rjn")

        if news_container:
            news_elements = news_container.select(".YbNUpf")

            for element in news_elements:
                title_element = element.select_one(".CbNjCe")
                title = title_element.text.strip() if title_element else ""
                source_element = element.select_one(".xvbXJb")
                source = source_element.text.strip() if source_element else ""
                date_element = element.select_one(".UtzVPe")
                date = date_element.text.strip() if date_element else ""
                url_element = element.select_one("a")
                url = url_element["href"] if url_element else ""

                news_item = {
                    "title": title,
                    "source": source,
                    "date": date,
                    "url": url,
                }
                news_list.append(news_item)
    return news_list

def get_price_data(ticker, start_date, end_date):
    try:
        price_data = yf.download(ticker, start=start_date, end=end_date)
    except:
        return pd.DataFrame()
    return price_data

def get_company_fundamentals(ticker):

    stock = yf.Ticker(ticker)
    info = stock.info

    fundamentals = {
        'Company Name': info.get('longName'),
        'Industry': info.get('industry'),
        'Market Cap': info.get('marketCap'),
        'Volume': info.get('volume'),
        'Gross Margins': info.get('grossMargins'),
        'Revenue': info.get('totalRevenue'),
        'Net Income': info.get('netIncomeToCommon'),
        'P/E Ratio': info.get('trailingPE')
    }

    # Format numbers using numerize
    for key in ['Market Cap', 'Volume', 'Gross Margins', 'Revenue', 'Net Income']:
        if fundamentals[key] is not None:
            fundamentals[key] = numerize.numerize(fundamentals[key])
    return fundamentals

def plot_stock_chart(price_data):
    fig = px.line(price_data, x=price_data.index, y='Close', title='Price Chart')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Closing Price')
    return fig

# Streamlit app layout setup
st.set_page_config(page_title="Stock Tracking APP", layout='wide')


# Sidebar with page selection
selected_page = st.sidebar.selectbox("Select Page", ["Stock Chart", "Related News", "Reddit Posts", "Your Portfolio"])

if selected_page == "Stock Chart":
    st.title("Stock Tracking APP")
    col1, col2, col3 = st.columns([2, 0.5, 2])
    with col1:
        ticker_stock = st.text_input('Enter stock symbol (e.g., AAPL):', 'AAPL')
        start_date_stock = st.date_input('Select start date:', pd.to_datetime('2021-01-01'))
        end_date_stock = st.date_input('Select end date:', pd.to_datetime('2022-01-01'))
        price_data = get_price_data(ticker_stock, start_date_stock, end_date_stock)
        if not price_data.empty:
            st.plotly_chart(plot_stock_chart(price_data))
        else:
            st.warning('No data available for the selected stock and date range.')
            ticker_stock = 'null'
    with col3:
        st.subheader("Company Fundamentals")
        fundamentals = get_company_fundamentals(ticker_stock)
        st.metric(label="Company Name", value=fundamentals['Company Name'])
        st.metric(label="Industry", value=fundamentals['Industry'])
        st.metric(label="Market Cap", value=fundamentals['Market Cap'])
        st.metric(label="Volume", value=fundamentals['Volume'])
        st.metric(label="Gross Margins", value=fundamentals['Gross Margins'])
        st.metric(label="Revenue", value=fundamentals['Revenue'])
        st.metric(label="Net Income", value=fundamentals['Net Income'])
        st.metric(label="P/E Ratio", value=fundamentals['P/E Ratio'])

elif selected_page == "Related News":
    st.title('Latest News')
    ticker_news = st.text_input('Enter stock symbol (e.g., AAPL):', 'AAPL')
    news_data = get_stock_news(ticker_news)
    if news_data:
        for news_item in news_data:
            sentiment, probability = run_finbert_inference(news_item["title"])
            st.write(f"Title: {news_item['title']}")
            st.write(f"Sentiment: {sentiment} (Probability: {probability:.2%})")
            st.write("Source:", news_item["source"])
            st.write("Date:", news_item["date"])
            st.write("URL:", news_item["url"])
            st.write("---")
    else:
        st.warning("Invalid ticker")



elif selected_page == "Reddit Posts":
    st.title("Reddit Posts")
    ticker_reddit = st.text_input('Enter stock symbol (e.g., AAPL):', 'AAPL')
    targets = ['investing', 'stocks', 'trading', 'finance']
    if st.button("Retrieve Reddit Posts"):
        posts_data = extract_posts(ticker_reddit, targets, 'new', 5)
        if posts_data:
            for post_info in posts_data:
                sentiment, probability = run_finbert_inference(post_info['Text'])
                st.write(f"Title: {post_info['Title']}")
                st.write(f"Sentiment: {sentiment} (Probability: {probability:.2%})")
                st.write('Subreddit:', post_info['Subreddit'])
                st.write('URL:', post_info['URL'])
                st.write('Time posted:', post_info['Created_UTC'])
                st.write('---')
        else:
            st.warning("invalid ticker")



elif selected_page == "Your Portfolio":
    st.title('Stock Portfolio')
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = pd.DataFrame(columns=['Ticker', 'Quantity', 'Current Price', 'Total Value', 'Market Cap'])
    with st.form(key='add_stock_form'):
        ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, TSLA):')
        quantity = st.number_input('Enter Quantity:', step=1)
        add_button = st.form_submit_button(label="Add to Portfolio")
        if add_button:
            if ticker:
                try:
                    stock_data = yf.Ticker(ticker).info
                    current_price = stock_data['currentPrice']
                    total_value = current_price * quantity
                    if ticker in st.session_state['portfolio']['Ticker'].values:
                        st.warning(f'{ticker} is already in your portfolio.')
                    else:
                        new_row = {
                            'Ticker': ticker,
                            'Quantity': quantity,
                            'Current Price': stock_data['currentPrice'],
                            'Total Value': total_value,
                            'Trading Volume': stock_data['volume'],
                            'Market Cap': stock_data['marketCap']
                        }
                        st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], pd.DataFrame([new_row])], ignore_index=True)
                        st.success(f'{ticker} added to your portfolio.')
                except:
                    st.warning('Please enter a valid stock ticker.')
            else:
                st.warning('Please enter a stock ticker.')
    with st.form(key='remove_stock_form'):
        remove_ticker = st.selectbox('Select Stock to Remove:', st.session_state['portfolio']['Ticker'])
        remove_button = st.form_submit_button(label='Remove from Portfolio')
        if remove_button:
            st.session_state['portfolio'] = st.session_state['portfolio'][st.session_state['portfolio']['Ticker'] != remove_ticker]
            st.success(f'{remove_ticker} removed from your portfolio.')
    st.subheader('Your Stock Portfolio:')
    st.table(st.session_state['portfolio'])
    total_portfolio_value = st.session_state['portfolio']['Total Value'].sum()
    st.subheader(f'Total Portfolio Value: ${total_portfolio_value:.2f}')
