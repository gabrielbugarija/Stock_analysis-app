from flask import Flask,request, redirect, url_for, flash, render_template
from pymongo import MongoClient
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField
from wtforms.validators import DataRequired
import yfinance as yfs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import io
import base64
import os



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key_here'
app.config['WTF_CSRF_ENABLED'] = False

client = MongoClient('mongodb://localhost:27017/')
db = client['stock_tracker']
MongoDB = db['shares']

class StockForm(FlaskForm):
    stock_symbol = StringField('Stock Symbol', validators=[DataRequired()])
    shares = IntegerField('Shares', validators=[DataRequired()])
    years = IntegerField('Years of Data', validators=[DataRequired()])

@app.route('/add_stock', methods=['POST'])
def add_stock():
    form = StockForm()
    if form.validate_on_submit():
        ticker = form.stock_symbol.data
        shares = form.shares.data
        years = form.years.data
        try:
            stock_data = get_stock_data(ticker, years)
            current_price = stock_data['Close'].iloc[-1]
            
            high_low_plot, ema_plot, prediction_plot = generate_plots(stock_data, ticker)
            
            if MongoDB.find_one({'_id': ticker}) is None:
                MongoDB.insert_one({'_id': ticker, 'total_cost': 0, 'purchases': []})
            
            MongoDB.update_one(
                {'_id': ticker},
                {
                    '$inc': {'total_cost': shares * float(current_price)},
                    '$push': {'purchases': {'shares': shares, 'price': current_price}},
                    '$set': {
                        'last_analysis': {
                            'date': datetime.now(),
                            'current_price': current_price,
                            'high_low_plot': high_low_plot,
                            'ema_plot': ema_plot,
                            'prediction_plot': prediction_plot
                        }
                    }
                }
            )
            flash('Stock added successfully!')
        except Exception as e:
            flash(f'Error adding stock: {e}')
    else:
        flash('Invalid form submission')
    return redirect(url_for("index"))

def get_stock_data(ticker, years):
    today = datetime.today()
    end_date = today.strftime('%Y-%m-%d')
    start_date = today.replace(year=today.year - years).strftime('%Y-%m-%d')
    data = yfs.download(ticker, start=start_date, end=end_date)
    
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data.Date)
    data['EMA-50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA-200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    return data

def generate_plots(data, ticker):
    # High vs Low plot
    plt.figure(figsize=(8, 4))
    plt.plot(data['Date'], data['Low'], label="Low", color="indianred")
    plt.plot(data['Date'], data['High'], label="High", color="mediumseagreen")
    plt.ylabel('Price (in USD)')
    plt.xlabel("Time")
    plt.title(f"High vs Low of {ticker}")
    plt.legend()
    plt.tight_layout()
    high_low_plot = get_plot_as_base64()

    # EMA plot
    plt.figure(figsize=(8, 4))
    plt.plot(data['Date'], data['EMA-50'], label="EMA for 50 days")
    plt.plot(data['Date'], data['EMA-200'], label="EMA for 200 days")
    plt.plot(data['Date'], data['Adj Close'], label="Close")
    plt.title(f'Exponential Moving Average for {ticker}')
    plt.ylabel('Price (in USD)')
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    ema_plot = get_plot_as_base64()

    # Polynomial Regression
    x = data[['Open', 'High', 'Low', 'Volume', 'EMA-50', 'EMA-200']]
    y = data['Close']
    poly_features = PolynomialFeatures(degree=2)
    x_poly = poly_features.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)
    poly_model = LinearRegression()
    poly_model.fit(X_train, y_train)
    pred = poly_model.predict(X_test)

    plt.figure(figsize=(8, 4))
    plt.plot(y_test.values, label="Actual Price")
    plt.plot(pred, label="Predicted Price")
    plt.xlabel("Data Points")
    plt.ylabel("Price")
    plt.title(f"Real vs Predicted Prices for {ticker}")
    plt.legend()
    plt.tight_layout()
    prediction_plot = get_plot_as_base64()

    return high_low_plot, ema_plot, prediction_plot

def get_plot_as_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    form = StockForm()
    stocks = list(MongoDB.find())
    template_dir = os.path.abspath(os.path.dirname(__file__))
    return render_template(os.path.join(template_dir, 'templates', 'index.html'), form=form, stocks=stocks)
def init_db():
    if 'stock_tracker' not in client.list_database_names():
        db = client['stock_tracker']
        db.create_collection('shares')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)