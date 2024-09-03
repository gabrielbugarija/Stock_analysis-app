import os
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from pymongo import MongoClient
import json
import plotly
import plotly.graph_objs as go
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB setup
client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
db = client.stock_app
stock_data = db.stock_data

def get_stock_data(symbol, start_date, end_date):
    existing_data = stock_data.find_one({
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date
    })

    if existing_data:
        try:
            df = pd.DataFrame(json.loads(existing_data['data']))
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
            return df
        except (ValueError, KeyError) as e:
            print(f"Error parsing stored data: {e}")
            return fetch_and_store_data(symbol, start_date, end_date)
    else:
        return fetch_and_store_data(symbol, start_date, end_date)

def fetch_and_store_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    if not data.empty:
        stock_data.insert_one({
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'data': data.reset_index().to_json(date_format='iso')
        })
    return data

def calculate_indicators(data):
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def predict_future_price(data, days_to_predict=365):
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days_to_predict)
    forecast = model.predict(future)
    return forecast

def create_stock_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], name='EMA 50'))
    fig.update_layout(
        title='Stock Price Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False
    )
    return plotly.io.to_json(fig)

def create_indicator_plots(data):
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
    macd_fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], name='Signal Line'))
    macd_fig.update_layout(title='MACD', xaxis_title='Date', yaxis_title='Value')
    
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    rsi_fig.update_layout(title='RSI', xaxis_title='Date', yaxis_title='Value')
    
    return plotly.io.to_json(macd_fig), plotly.io.to_json(rsi_fig)

def create_prediction_plot(data, forecast):
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'],
                             mode='lines',
                             name='Historical Close Price'))
    
    # Predicted data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                             mode='lines',
                             name='Predicted Close Price'))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend_title="Data Series",
        hovermode="x unified"
    )
    
    return plotly.io.to_json(fig)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['stock']
        years = int(request.form['years'])
        days_to_predict = int(request.form.get('days_to_predict', 365))

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')

        try:
            data = get_stock_data(symbol, start_date, end_date)
            
            if data.empty:
                return render_template('error.html', message=f"No data available for {symbol} in the specified date range.")

            data = calculate_indicators(data)
            stock_plot = create_stock_plot(data)
            macd_plot, rsi_plot = create_indicator_plots(data)
            
            forecast = predict_future_price(data, days_to_predict)
            prediction_plot = create_prediction_plot(data, forecast)
            
            final_predicted_price = f"{forecast['yhat'].iloc[-1]:.2f}"
            prediction_date = forecast['ds'].iloc[-1].strftime('%Y-%m-%d')

            return render_template('results.html', 
                                   stock=symbol, 
                                   years=years,
                                   stock_plot=stock_plot,
                                   macd_plot=macd_plot,
                                   rsi_plot=rsi_plot,
                                   prediction_plot=prediction_plot,
                                   final_predicted_price=final_predicted_price,
                                   prediction_date=prediction_date)

        except Exception as e:
            return render_template('error.html', message=f"An error occurred: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)