import os
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.offline as pyo
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def fetch_stock_data(symbol, years=1):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None, "No data available for this stock"
        
        return df, None
    except Exception as e:
        logging.error(f"Stock data fetch error: {e}")
        return None, str(e)

def generate_stock_chart(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=f'Stock Price', xaxis_rangeslider_visible=False)
    return pyo.plot(fig, output_type='div')

def predict_stock_price(df, days_to_predict=30):
    try:
        # Remove timezone information
        df.index = df.index.tz_localize(None)
        
        # Prepare data for Prophet
        prophet_df = df[['Close']].reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Ensure data is sufficient and valid
        if len(prophet_df) < 30:
            return None, "Insufficient historical data for prediction"
        
        if prophet_df['y'].isnull().any():
            return None, "Contains invalid price data"
        
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=days_to_predict)
        forecast = model.predict(future)
        
        return forecast, None
    except Exception as e:
        logging.error(f"Prediction detailed error: {e}")
        return None, f"Prediction failed: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('stock', '').upper()
        years = int(request.form.get('years', 1))
        days_to_predict = int(request.form.get('days_to_predict', 30))
        
        df, error = fetch_stock_data(symbol, years)
        
        if error:
            return render_template('error.html', message=error)
        
        stock_chart = generate_stock_chart(df)
        forecast, prediction_error = predict_stock_price(df, days_to_predict)
        
        if prediction_error:
            return render_template('error.html', message=prediction_error)
        
        predicted_price = forecast['yhat'].iloc[-1]
        prediction_date = forecast['ds'].iloc[-1]
        
        return render_template('results.html', 
                               stock=symbol, 
                               stock_chart=stock_chart,
                               predicted_price=f"{predicted_price:.2f}",
                               prediction_date=prediction_date.strftime('%Y-%m-%d'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)