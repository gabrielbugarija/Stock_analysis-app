from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
import numpy as np
np.float_ = np.float64
from prophet import Prophet


app = Flask(__name__)

def predict_future_price(data, days_to_predict=365):
    # Prepare data for Prophet
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']

    # Create and fit the model
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=days_to_predict)
    
    # Make predictions
    forecast = model.predict(future)
    
    return forecast

def create_stock_ema_plot(data, ema_period=50):
    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Stock Price', color='blue')
    plt.plot(data.index, data['EMA'], label=f'EMA ({ema_period})', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'Stock Price and EMA({ema_period})')
    plt.legend()
    plt.tight_layout()

    stock_ema_plot = io.BytesIO()
    plt.savefig(stock_ema_plot, format='png')
    stock_ema_plot.seek(0)
    stock_ema_plot = base64.b64encode(stock_ema_plot.getvalue()).decode()
    plt.close()

    return stock_ema_plot

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        whichstock = request.form['stock']
        howmanyyears = int(request.form['years'])
        days_to_predict = int(request.form.get('days_to_predict', 365))
        ema_period = int(request.form.get('ema_period', 50))

        today = datetime.today()
        END_DATE = today.strftime('%Y-%m-%d')
        START_DATE = today.replace(year=today.year - howmanyyears).strftime('%Y-%m-%d')
        data = yf.download(whichstock, start=START_DATE, end=END_DATE)
        
        if data.empty:
            return render_template('error.html', message=f"No data available for {whichstock} in the specified date range.")

        try:
            # Create stock and EMA plot
            stock_ema_plot = create_stock_ema_plot(data, ema_period)

            # Predict future prices
            forecast = predict_future_price(data, days_to_predict)

            # Create prediction plot
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['Close'], label='Historical Price', color='blue')
            plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='red')
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.2)
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.title(f'Historical and Predicted Prices for {whichstock}')
            plt.legend()
            plt.tight_layout()

            prediction_plot = io.BytesIO()
            plt.savefig(prediction_plot, format='png')
            prediction_plot.seek(0)
            prediction_plot = base64.b64encode(prediction_plot.getvalue()).decode()
            plt.close()

            # Get the predicted price for the last day
            final_predicted_price = forecast['yhat'].iloc[-1]

            return render_template('results.html', stock=whichstock, years=howmanyyears,
                                   stock_ema_plot=stock_ema_plot,
                                   prediction_plot=prediction_plot,
                                   final_predicted_price=f"{final_predicted_price:.2f}",
                                   prediction_date=forecast['ds'].iloc[-1].strftime('%Y-%m-%d'))

        except Exception as e:
            return render_template('error.html', message=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)