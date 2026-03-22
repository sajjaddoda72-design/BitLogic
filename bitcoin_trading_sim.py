import numpy as np
import pandas as pd

def simulate_bitcoin_prices(days=60, start_price=60000.0, volatility=0.03, drift=0.0005):
    """
    Simulates Bitcoin price data using Geometric Brownian Motion.
    volatility is daily (0.03 = 3%)
    drift is daily expected return
    """
    np.random.seed(1)
    daily_returns = np.random.normal(drift, volatility, days)
    price_path = start_price * np.exp(np.cumsum(daily_returns))
    
    dates = pd.date_range(start='2023-01-01', periods=days)
    df = pd.DataFrame({'Price': price_path}, index=dates)
    return df

def calculate_moving_averages(df):
    """
    Calculates 7-day and 30-day Simple Moving Averages.
    """
    df['SMA7'] = df['Price'].rolling(window=7).mean()
    df['SMA30'] = df['Price'].rolling(window=30).mean()
    return df

def simulate_trading(df, initial_balance=10000.0):
    """
    Implements a Golden Cross trading strategy.
    Buy when SMA7 > SMA30, Sell when SMA7 < SMA30.
    """
    balance = initial_balance
    btc_held = 0.0
    ledger = []
    
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        price = row['Price']
        sma7 = row['SMA7']
        sma30 = row['SMA30']
        
        # We need both SMAs to be available
        if pd.isna(sma7) or pd.isna(sma30):
            ledger.append({'Date': index, 'Price': price, 'SMA7': sma7, 'SMA30': sma30, 'Action': 'Wait', 'Balance': balance, 'BTC': btc_held, 'Total Value': balance + btc_held * price})
            continue
            
        action = 'Hold'
        # Golden Cross Buy Signal: SMA7 crosses above SMA30
        if sma7 > sma30 and btc_held == 0:
            btc_held = balance / price
            balance = 0
            action = 'BUY'
        # Death Cross Sell Signal: SMA7 crosses below SMA30
        elif sma7 < sma30 and btc_held > 0:
            balance = btc_held * price
            btc_held = 0
            action = 'SELL'
            
        total_value = balance + (btc_held * price)
        ledger.append({'Date': index, 'Price': price, 'SMA7': sma7, 'SMA30': sma30, 'Action': action, 'Balance': balance, 'BTC': btc_held, 'Total Value': total_value})
    
    ledger_df = pd.DataFrame(ledger)
    return ledger_df

def print_performance(ledger_df, initial_balance=10000.0):
    """
    Prints the daily ledger and final portfolio performance.
    """
    print("Daily Trading Ledger:")
    print("-" * 110)
    pd.set_option('display.max_rows', None)
    # Format the floats for better readability
    formatted_df = ledger_df.copy()
    formatted_df['Price'] = formatted_df['Price'].map('{:,.2f}'.format)
    formatted_df['SMA7'] = formatted_df['SMA7'].map(lambda x: '{:,.2f}'.format(x) if pd.notnull(x) else 'NaN')
    formatted_df['SMA30'] = formatted_df['SMA30'].map(lambda x: '{:,.2f}'.format(x) if pd.notnull(x) else 'NaN')
    formatted_df['Balance'] = formatted_df['Balance'].map('{:,.2f}'.format)
    formatted_df['BTC'] = formatted_df['BTC'].map('{:,.4f}'.format)
    formatted_df['Total Value'] = formatted_df['Total Value'].map('{:,.2f}'.format)
    
    print(formatted_df[['Date', 'Price', 'SMA7', 'SMA30', 'Action', 'Balance', 'BTC', 'Total Value']].to_string(index=False))
    print("-" * 110)
    
    final_value = ledger_df['Total Value'].iloc[-1]
    profit = final_value - initial_balance
    roi = (profit / initial_balance) * 100
    
    print(f"Final Portfolio Performance:")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Value:     ${final_value:,.2f}")
    print(f"Total Profit:    ${profit:,.2f}")
    print(f"Total ROI:       {roi:.2f}%")

if __name__ == "__main__":
    initial_bal = 10000.0
    df = simulate_bitcoin_prices(days=60)
    df = calculate_moving_averages(df)
    ledger_df = simulate_trading(df, initial_balance=initial_bal)
    print_performance(ledger_df, initial_balance=initial_bal)
