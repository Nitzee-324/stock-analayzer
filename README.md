# Stock Analyzer

A Django-based web application for analyzing and visualizing stock market data. This project provides two main dashboards: one for Indian stocks and another for US technology stocks, with comprehensive charting and technical analysis capabilities.

## Features

### 📊 Indian Stock Analyzer (`/index`)
- Select from 60+ major Indian stocks (Reliance, TCS, Infosys, HDFC Bank, etc.)
- Fetch up to 1 year of historical stock data
- Generate comprehensive analysis charts:
  - Adjusted close price trends
  - 50-day and 200-day moving averages
  - Trading volume analysis
  - Daily returns distribution
- Calculate expected returns and volatility metrics
- Predict future stock prices for 3, 6, or 12-month periods

### 📈 Global Stock Dashboard (`/`)
- Monitor 30+ US technology stocks
- Interactive Plotly charts with multiple visualization options:
  - **Line Chart** - Basic price movement tracking
  - **Candlestick Chart** - OHLC (Open, High, Low, Close) data
- Technical indicators:
  - **SMA (Simple Moving Average)** - 20 and 50-day averages
  - **Bollinger Bands** - Volatility analysis with upper/lower bands
  - **RSI (Relative Strength Index)** - Momentum and overbought/oversold conditions
- Customizable date range selection
- Display company information:
  - Market capitalization
  - P/E ratio
  - 52-week high/low
  - Dividend yield and rates
  - Sector and industry classification

## Technology Stack

- **Backend**: Django 5.0.6
- **Data Source**: Yahoo Finance (via yfinance)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Database**: SQLite
- **Python Version**: 3.x

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd stock-analayzer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**
   ```bash
   python manage.py migrate
   ```

5. **Run the development server**
   ```bash
   python manage.py runserver
   ```

6. **Access the application**
   - Dashboard: `http://localhost:8000/`
   - Indian Stock Analyzer: `http://localhost:8000/index`

## Project Structure

```
stock-analayzer/
├── db.sqlite3                 # SQLite database
├── manage.py                  # Django management script
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── analyzer/                  # Main Django app
│   ├── __init__.py
│   ├── admin.py              # Django admin configuration
│   ├── apps.py               # App configuration
│   ├── models.py             # Database models (currently empty)
│   ├── tests.py              # Unit tests
│   ├── urls.py               # URL routing for analyzer app
│   ├── views.py              # View functions and logic
│   ├── migrations/           # Database migration files
│   └── templates/            # HTML templates
│       ├── dashboard.html    # Global stock dashboard
│       └── index.html        # Indian stock analyzer
└── stock_analyzer/           # Main Django project config
    ├── __init__.py
    ├── asgi.py               # ASGI configuration
    ├── settings.py           # Project settings
    ├── urls.py               # Main URL routing
    └── wsgi.py               # WSGI configuration
```

## Usage

### Indian Stock Analyzer
1. Navigate to `/index`
2. Select one or more Indian stocks from the dropdown list
3. Choose an analysis period (3, 6, or 12 months)
4. Submit the form to see:
   - Price charts and moving averages
   - Volume analysis
   - Daily returns distribution
   - Expected returns and volatility metrics
   - Predicted future prices

### Global Stock Dashboard
1. Navigate to `/` (home page)
2. Select a US stock ticker (defaults to Apple - AAPL)
3. Choose date range for analysis
4. Select chart type (Line or Candlestick)
5. Select technical indicator (SMA, Bollinger Bands, or RSI)
6. View interactive charts and company information

## Key Dependencies

- **django** - Web framework
- **yfinance** - Yahoo Finance data fetching
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Static chart generation
- **seaborn** - Statistical data visualization
- **plotly** - Interactive charting library
- **beautifulsoup4** - Web scraping (for utilities)

See `requirements.txt` for complete list with versions.

## Configuration

### Django Settings
Edit `stock_analyzer/settings.py` to:
- Change `SECRET_KEY` for production use
- Set `DEBUG = False` for production
- Configure `ALLOWED_HOSTS` for your domain

### Adding More Stocks
- **Indian Stocks**: Edit `STOCK_CHOICES` in `analyzer/views.py`
- **US Stocks**: Edit `stock_symbols` dict in the `dashboard()` function

## Future Enhancements

- [ ] Implement database models for caching historical data
- [ ] Add user authentication and portfolio tracking
- [ ] Create user watchlists and alerts
- [ ] Add more technical indicators (MACD, Stochastic, etc.)
- [ ] Implement machine learning for price predictions
- [ ] Add comparison charts for multiple stocks
- [ ] Export data to CSV/Excel

## Troubleshooting

**Issue**: Data not loading
- Ensure you have an active internet connection (yfinance fetches live data)
- Check that the ticker symbols are valid

**Issue**: Charts not displaying
- Clear browser cache
- Ensure matplotlib and plotly are properly installed

**Issue**: Import errors
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions or issues, please open an issue in the project repository.
