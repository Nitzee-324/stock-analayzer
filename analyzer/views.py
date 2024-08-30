from django.shortcuts import render
from django import forms
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from io import BytesIO
import base64

class StockForm(forms.Form):
    STOCK_CHOICES = [
       ('RELIANCE.NS', 'Reliance Industries'),
    ('TCS.NS', 'Tata Consultancy Services'),
    ('INFY.NS', 'Infosys'),
    ('HDFCBANK.NS', 'HDFC Bank'),
    ('HINDUNILVR.NS', 'Hindustan Unilever'),
    ('ITC.NS', 'ITC Limited'),
    ('BHARTIARTL.NS', 'Bharti Airtel'),
    ('ICICIBANK.NS', 'ICICI Bank'),
    ('KOTAKBANK.NS', 'Kotak Mahindra Bank'),
    ('SBIN.NS', 'State Bank of India'),
    ('ADANIGREEN.NS', 'Adani Green Energy'),
    ('MARUTI.NS', 'Maruti Suzuki'),
    ('HCLTECH.NS', 'HCL Technologies'),
    ('WIPRO.NS', 'Wipro'),
    ('AXISBANK.NS', 'Axis Bank'),
    ('INDUSINDBK.NS', 'IndusInd Bank'),
    ('POWERGRID.NS', 'Power Grid Corporation'),
    ('NTPC.NS', 'NTPC Limited'),
    ('SUNPHARMA.NS', 'Sun Pharmaceutical'),
    ('DRREDDY.NS', 'Dr. Reddy\'s Laboratories'),
    ('HEROMOTOCO.NS', 'Hero MotoCorp'),
    ('ULTRACEMCO.NS', 'UltraTech Cement'),
    ('JSWSTEEL.NS', 'JSW Steel'),
    ('EICHERMOT.NS', 'Eicher Motors'),
    ('TATASTEEL.NS', 'Tata Steel'),
    ('GRASIM.NS', 'Grasim Industries'),
    ('DIVISLAB.NS', 'Divi\'s Laboratories'),
    ('HDFCLIFE.NS', 'HDFC Life Insurance'),
    ('BajajAuto.NS', 'Bajaj Auto'),
    ('HDFC.NS', 'HDFC Limited'),
    ('SBI.NS', 'State Bank of India'),
    ('BHEL.NS', 'Bharat Heavy Electricals'),
    ('LUPIN.NS', 'Lupin Pharmaceuticals'),
    ('ADANIPORTS.NS', 'Adani Ports and SEZ'),
    ('GAIL.NS', 'GAIL (India) Limited'),
    ('M&M.NS', 'Mahindra & Mahindra'),
    ('IOC.NS', 'Indian Oil Corporation'),
    ('AUROPHARMA.NS', 'Aurobindo Pharma'),
    ('PVR.NS', 'PVR Limited'),
    ('CIPLA.NS', 'Cipla Limited'),
    ('HINDPETRO.NS', 'Hindustan Petroleum'),
    ('MUTHOOTFIN.NS', 'Muthoot Finance'),
    ('SHREECEM.NS', 'Shree Cement'),
    ('TECHM.NS', 'Tech Mahindra'),
    ('RELCAPITAL.NS', 'Reliance Capital'),
    ('COLPAL.NS', 'Colgate-Palmolive'),
    ('TATAMOTORS.NS', 'Tata Motors'),
    ('BERGEPAINT.NS', 'Berger Paints'),
    ('PIDILITIND.NS', 'Pidilite Industries'),
    ('HDFCAMC.NS', 'HDFC Asset Management'),
    ('TCS.NS', 'Tata Consultancy Services'),
    ('MINDTREE.NS', 'MindTree'),
    ('BANKBARODA.NS', 'Bank of Baroda'),
    ('NHPC.NS', 'NHPC Limited'),
    ('CANBK.NS', 'Canara Bank'),
    ('ABFRL.NS', 'Aditya Birla Fashion & Retail'),
    ('JUBLFOOD.NS', 'Jubilant FoodWorks'),
    ('HINDZINC.NS', 'Hindustan Zinc'),
    ('GODREJCP.NS', 'Godrej Consumer Products'),
    ('LICHSGFIN.NS', 'LIC Housing Finance'),
    ('PFC.NS', 'Power Finance Corporation'),
    ('BANDHANBNK.NS', 'Bandhan Bank')
    ]

    PERIOD_CHOICES = [
        (3, '3 months'),
        (6, '6 months'),
        (12, '1 year')
    ]

    stocks = forms.MultipleChoiceField(choices=STOCK_CHOICES)
    period = forms.ChoiceField(choices=PERIOD_CHOICES)

def fetch_stock_data(selected_tickers):
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    if not selected_tickers:
        return None, None

    data = yf.download(selected_tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
    combined_data = pd.DataFrame()

    for ticker in selected_tickers:
        if ticker in data:
            ticker_data = data[ticker].reset_index()
            ticker_data['Ticker'] = ticker
            combined_data = pd.concat([combined_data, ticker_data], ignore_index=True)

    data_melted = combined_data.melt(id_vars=['Date', 'Ticker'], var_name='Attribute', value_name='Value')
    data_pivoted = data_melted.pivot_table(index=['Date', 'Ticker'], columns='Attribute', values='Value', aggfunc='first')
    stock_data = data_pivoted.reset_index()

    stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()

    daily_returns = stock_data.groupby('Ticker')['Daily Return'].mean()
    expected_returns = daily_returns * 252
    volatility = stock_data.groupby('Ticker')['Daily Return'].std() * np.sqrt(252)

    stock_stats = pd.DataFrame({
        'Expected Return': expected_returns,
        'Volatility': volatility
    })

    return stock_data, stock_stats

def plot_stock_data(stock_data):
    image_data_list = []

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    stock_data.reset_index(inplace=True)

    # Plot Adjusted Close Price
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set(style='whitegrid')
    sns.lineplot(data=stock_data, x='Date', y='Adj Close', hue='Ticker', marker='o', ax=ax)
    ax.set_title('Adjusted Close Price Over Time', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Adjusted Close Price', fontsize=14)
    ax.legend(title='Ticker', title_fontsize='13', fontsize='11')
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    image_data_list.append(image_base64)
    plt.close()

    # Calculate moving averages and plot
    short_window = 50
    long_window = 200
    unique_tickers = stock_data['Ticker'].unique()

    for ticker in unique_tickers:
        ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
        ticker_data['50_MA'] = ticker_data['Adj Close'].rolling(window=short_window).mean()
        ticker_data['200_MA'] = ticker_data['Adj Close'].rolling(window=long_window).mean()

        # Plot Adjusted Close and Moving Averages
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(ticker_data['Date'], ticker_data['Adj Close'], label='Adj Close')
        ax.plot(ticker_data['Date'], ticker_data['50_MA'], label='50-Day MA')
        ax.plot(ticker_data['Date'], ticker_data['200_MA'], label='200-Day MA')
        ax.set_title(f'{ticker} - Adjusted Close and Moving Averages')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        image_data_list.append(image_base64)
        plt.close()

        # Plot Volume
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(ticker_data['Date'], ticker_data['Volume'], label='Volume', color='orange')
        ax.set_title(f'{ticker} - Volume Traded')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        image_data_list.append(image_base64)
        plt.close()

    # Plot Daily Returns Distribution
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set(style='whitegrid')
    for ticker in unique_tickers:
        ticker_data = stock_data[stock_data['Ticker'] == ticker]
        sns.histplot(ticker_data['Daily Return'].dropna(), bins=50, kde=True, label=ticker, alpha=0.5, ax=ax)

    ax.set_title('Distribution of Daily Returns', fontsize=16)
    ax.set_xlabel('Daily Return', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.legend(title='Ticker', title_fontsize='13', fontsize='11')
    ax.grid(True)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    image_data_list.append(image_base64)
    plt.close()

    return image_data_list



def calculate_expected_return(stock_stats, current_prices, selected_tickers, period_months):
    expected_return = stock_stats['Expected Return']
    selected_tickers_list = list(selected_tickers)
    expected_return_annual = expected_return.loc[selected_tickers_list]

    periods = {
        3: 0.25,
        6: 0.5,
        12: 1
    }

    if period_months not in periods:
        return None

    factor = periods[period_months]
    return_predictions = current_prices * (1 + expected_return_annual * factor)

    return return_predictions

def index(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            selected_stocks = form.cleaned_data['stocks']
            period_months = int(form.cleaned_data['period'])

            stock_data, stock_stats = fetch_stock_data(selected_stocks)

            if stock_data is not None and stock_stats is not None:
                image_data_list = plot_stock_data(stock_data)

                current_prices = {}
                for ticker in selected_stocks:
                    ticker_data = yf.Ticker(ticker).history(period='1d')
                    current_prices[ticker] = ticker_data['Close'].iloc[-1]

                current_prices_series = pd.Series(current_prices)
                expected_returns = calculate_expected_return(stock_stats, current_prices_series, selected_stocks, period_months)

                return render(request, 'index.html', {
                    'form': form,
                    'images': image_data_list,
                    'current_prices': current_prices,
                    'expected_returns': expected_returns.to_dict()
                })
    else:
        form = StockForm()

    return render(request, 'index.html', {'form': form})


# analyzer/views.py
from django.shortcuts import render
import yfinance as yf
import plotly.graph_objects as go
import datetime

def dashboard(request):
    stock_symbols = {
       'Apple Inc.': 'AAPL',
    'Microsoft Corp': 'MSFT',
    'Google LLC': 'GOOGL',
    'Amazon.com Inc.': 'AMZN',
    'Tesla Inc.': 'TSLA',
    'Facebook Inc.': 'META',
    'NVIDIA Corporation': 'NVDA',
    'Netflix Inc.': 'NFLX',
    'Adobe Inc.': 'ADBE',
    'Intel Corporation': 'INTC',
    'Cisco Systems Inc.': 'CSCO',
    'Salesforce.com Inc.': 'CRM',
    'PayPal Holdings Inc.': 'PYPL',
    'Qualcomm Inc.': 'QCOM',
    'Advanced Micro Devices Inc.': 'AMD',
    'Texas Instruments Inc.': 'TXN',
    'Oracle Corporation': 'ORCL',
    'IBM Corporation': 'IBM',
    'Micron Technology Inc.': 'MU',
    'Uber Technologies Inc.': 'UBER',
    'Zoom Video Communications Inc.': 'ZM',
    'Pinterest Inc.': 'PINS',
    'Snap Inc.': 'SNAP',
    'Spotify Technology S.A.': 'SPOT',
    'Square Inc.': 'SQ',
    'Twilio Inc.': 'TWLO',
    'Shopify Inc.': 'SHOP',
    'Slack Technologies Inc.': 'WORK',
    'Dropbox Inc.': 'DBX',
    'Lyft Inc.': 'LYFT',
    'Palantir Technologies Inc.': 'PLTR',
    'Roku Inc.': 'ROKU',
    'Cloudflare Inc.': 'NET',
    'Snowflake Inc.': 'SNOW',
    'CrowdStrike Holdings Inc.': 'CRWD',
    'Datadog Inc.': 'DDOG',
    'DocuSign Inc.': 'DOCU',
    'Elastic N.V.': 'ESTC',
    'Okta Inc.': 'OKTA',
    'MongoDB Inc.': 'MDB',
    'Zscaler Inc.': 'ZS',
    'Atlassian Corporation Plc': 'TEAM',
    'ServiceNow Inc.': 'NOW',
    'Splunk Inc.': 'SPLK',
    'Alteryx Inc.': 'AYX',
    'Pinterest Inc.': 'PINS',
    'Fastly Inc.': 'FSLY',
    'Fiverr International Ltd.': 'FVRR',
    'C3.ai Inc.': 'AI',
    'DraftKings Inc.': 'DKNG',
    'Wayfair Inc.': 'W'
    }

    ticker = request.GET.get('ticker', 'AAPL')  # Default to Apple Inc.
    start_date = request.GET.get('start_date', (datetime.datetime.now() - datetime.timedelta(days=40)).date())
    end_date = request.GET.get('end_date', datetime.datetime.now().date())
    chart_type = request.GET.get('chart_type', 'line')
    add_indicator = request.GET.get('indicator', 'S')

    # Debugging print statements
    print("Ticker:", ticker)
    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("Chart Type:", chart_type)
    print("Indicator:", add_indicator)

    # Convert start_date and end_date from string to date
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    fig = go.Figure()

    # Add the main chart type
    if chart_type == 'candlestick':
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                     open=stock_data['Open'],
                                     high=stock_data['High'],
                                     low=stock_data['Low'],
                                     close=stock_data['Close'],
                                     name='Candlestick'))
    else:
        fig.add_trace(go.Scatter(x=stock_data.index, 
                                 y=stock_data['Close'], 
                                 mode='lines', 
                                 name='Close Price'))

    # Add indicators
    if add_indicator == 'SMA':
        print("Adding SMA indicator")
        stock_data['20-Day SMA'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['50-Day SMA'] = stock_data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=stock_data.index, 
                                 y=stock_data['20-Day SMA'], 
                                 mode='lines', 
                                 name='20-Day SMA',
                                 line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=stock_data.index, 
                                 y=stock_data['50-Day SMA'], 
                                 mode='lines', 
                                 name='50-Day SMA',
                                 line=dict(color='green')))
    
    elif add_indicator == 'Bollinger Bands':
        print("Adding Bollinger Bands indicator")
        stock_data['20-Day SMA'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['Bollinger High'] = stock_data['20-Day SMA'] + 2 * stock_data['Close'].rolling(window=20).std()
        stock_data['Bollinger Low'] = stock_data['20-Day SMA'] - 2 * stock_data['Close'].rolling(window=20).std()
        fig.add_trace(go.Scatter(x=stock_data.index, 
                                 y=stock_data['Bollinger High'], 
                                 mode='lines', 
                                 name='Bollinger High',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=stock_data.index, 
                                 y=stock_data['Bollinger Low'], 
                                 mode='lines', 
                                 name='Bollinger Low',
                                 line=dict(color='blue')))
    
    elif add_indicator == 'RSI':
        print("Adding RSI indicator")
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(x=stock_data.index, 
                                 y=stock_data['RSI'], 
                                 mode='lines', 
                                 name='RSI',
                                 line=dict(color='purple')))

    plot_html = fig.to_html(full_html=False)

    # Fetch company info
    stock = yf.Ticker(ticker)
    info = stock.info
    company_info = {
        'Company Name': info.get('longName', 'N/A'),
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A'),
        'Market Cap': info.get('marketCap', 'N/A'),
        'Current Price': info.get('currentPrice', 'N/A'),
        '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
        'P/E Ratio': info.get('forwardEps', 'N/A'),
        'Sales': info.get('totalRevenue', 'N/A'),
        'Book Value': info.get('bookValue', 'N/A'),
        'Dividend Yield': info.get('dividendYield', 'N/A'),
        'Dividend Value': info.get('dividendRate', 'N/A'),
    }

    context = {
        'plot_html': plot_html,
        'company_info': company_info,
        'stock_symbols': stock_symbols,
        'selected_ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'chart_type': chart_type,
        'add_indicator': add_indicator
    }

    return render(request, 'dashboard.html', context)


