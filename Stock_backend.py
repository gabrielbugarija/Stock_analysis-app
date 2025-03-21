from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ta

app = Flask(__name__)

class StockAnalyzer:
    def __init__(self, symbol, prediction_days=30):
        self.symbol = symbol
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch historical data using yfinance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        self.df = yf.download(self.symbol, start=start_date, end=end_date)
        return self.df

    def prepare_data(self):
        """Prepare data for LSTM model"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.df['Close'].values.reshape(-1, 1))
        
        # Prepare training data
        x_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            x_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train

    def create_model(self, x_train, y_train):
        """Create and train LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(units=50, return_sequences=False),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
        
        return model

    def predict_prices(self, model):
        """Generate price predictions"""
        # Prepare data for prediction
        last_60_days = self.df['Close'].values[-60:]
        scaled_data = self.scaler.transform(last_60_days.reshape(-1, 1))
        
        # Generate predictions
        future_predictions = []
        current_batch = scaled_data.reshape((1, 60, 1))
        
        for _ in range(self.prediction_days):
            future_price = model.predict(current_batch)[0]
            future_predictions.append(future_price)
            current_batch = np.append(current_batch[:, 1:, :], 
                                    [[future_price]], 
                                    axis=1)
        
        # Inverse transform predictions
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        return future_predictions.flatten()

    def calculate_technical_indicators(self):
        """Calculate MACD and RSI indicators"""
        # Calculate MACD
        macd = ta.trend.MACD(close=self.df['Close'])
        self.df['MACD'] = macd.macd()
        self.df['Signal'] = macd.macd_signal()
        
        # Calculate RSI
        self.df['RSI'] = ta.momentum.RSIIndicator(close=self.df['Close']).rsi()
        
        return self.df

def prepare_chart_data(df, predictions):
    """Prepare data for charts"""
    dates = df.index.strftime('%Y-%m-%d').tolist()
    future_dates = pd.date_range(
        start=df.index[-1] + timedelta(days=1),
        periods=len(predictions),
        freq='D'
    ).strftime('%Y-%m-%d').tolist()

    historical_data = df['Close'].round(2).tolist()
    predictions = predictions.round(2).tolist()

    # Prepare MACD data
    macd_data = {
        'macdLine': df['MACD'].round(3).fillna(0).tolist(),
        'signalLine': df['Signal'].round(3).fillna(0).tolist()
    }

    # Prepare RSI data
    rsi_data = df['RSI'].round(2).fillna(0).tolist()

    return {
        'dates': dates + future_dates,
        'prices': historical_data + [None] * len(predictions),
        'predictions': [None] * len(historical_data) + predictions,
        'macd_data': macd_data,
        'rsi_data': rsi_data
    }

@app.route('/stock/<symbol>')
def stock_analysis(symbol):
    # Initialize analyzer
    analyzer = StockAnalyzer(symbol)
    
    try:
        # Fetch and prepare data
        df = analyzer.fetch_data()
        x_train, y_train = analyzer.prepare_data()
        
        # Create and train model
        model = analyzer.create_model(x_train, y_train)
        
        # Generate predictions
        predictions = analyzer.predict_prices(model)
        
        # Calculate technical indicators
        df = analyzer.calculate_technical_indicators()
        
        # Prepare chart data
        chart_data = prepare_chart_data(df, predictions)
        
        # Prepare template variables
        template_data = {
            'stock': symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'prediction_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'final_predicted_price': f"{predictions[-1]:.2f}",
            **chart_data
        }
        
        return render_template('results.html', **template_data)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)