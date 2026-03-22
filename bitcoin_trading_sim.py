"""Bitcoin trading simulator using Golden Cross / Death Cross strategy."""

from __future__ import annotations

from typing import Any

import math

import numpy as np
import pandas as pd


def simulate_bitcoin_prices(
    days: int = 60,
    start_price: float = 60000.0,
    volatility: float = 0.03,
    drift: float = 0.0005,
) -> pd.DataFrame:
    """Simulate Bitcoin price data using Geometric Brownian Motion.

    Args:
        days: Number of trading days to simulate.
        start_price: Starting BTC price in USD.
        volatility: Daily volatility (0.03 = 3%).
        drift: Daily expected return.

    Returns:
        DataFrame with a 'Price' column indexed by date.
    """
    rng = np.random.default_rng(seed=1)
    daily_returns = rng.normal(drift, volatility, days)
    price_path = start_price * np.exp(np.cumsum(daily_returns))

    dates = pd.date_range(start="2023-01-01", periods=days)
    return pd.DataFrame({"Price": price_path}, index=dates)


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA7 and SMA30 columns to the price DataFrame."""
    df["SMA7"] = df["Price"].rolling(window=7).mean()
    df["SMA30"] = df["Price"].rolling(window=30).mean()
    return df


def simulate_trading(
    df: pd.DataFrame,
    initial_balance: float = 10000.0,
    leverage: float = 1.0,
) -> pd.DataFrame:
    """Run a Golden Cross / Death Cross trading strategy with leverage.

    Buy when SMA7 > SMA30, Sell when SMA7 < SMA30.
    Leverage multiplies profits **and** losses on each trade.

    Args:
        df: DataFrame with Price, SMA7, SMA30 columns.
        initial_balance: Starting cash in USD.
        leverage: Leverage multiplier (e.g. 10 means 10x).

    Returns:
        DataFrame ledger with one row per day.
    """
    balance = initial_balance
    btc_held = 0.0
    entry_price = 0.0  # track the price at which we entered a position
    ledger: list[dict[str, Any]] = []

    for i in range(len(df)):
        row = df.iloc[i]
        date_idx = df.index[i]
        price: float = float(row["Price"])
        sma7_raw = row["SMA7"]
        sma30_raw = row["SMA30"]
        date_obj = getattr(date_idx, "date", None)
        date_str = str(date_obj()) if callable(date_obj) else str(date_idx)

        if pd.isna(sma7_raw) or pd.isna(sma30_raw):
            ledger.append(
                {
                    "Date": date_str,
                    "Price": price,
                    "SMA7": None,
                    "SMA30": None,
                    "Action": "Wait",
                    "Balance": balance,
                    "BTC": btc_held,
                    "Total Value": balance + btc_held * price,
                }
            )
            continue

        sma7_val = float(sma7_raw)
        sma30_val = float(sma30_raw)
        action = "Hold"

        # Golden Cross Buy Signal
        if sma7_val > sma30_val and btc_held == 0:
            # With leverage we can control more BTC than our balance allows
            btc_held = (balance * leverage) / price
            entry_price = price
            balance = 0.0
            action = "BUY"

        # Death Cross Sell Signal
        elif sma7_val < sma30_val and btc_held > 0:
            # Calculate leveraged P&L
            raw_value = btc_held * price
            notional = btc_held * entry_price  # what we "borrowed"
            pnl = raw_value - notional  # leveraged P&L already baked in via btc_held
            # Our actual equity was notional / leverage
            equity = notional / leverage
            balance = equity + pnl
            # Clamp to zero (liquidation scenario)
            if balance < 0:
                balance = 0.0
            btc_held = 0.0
            entry_price = 0.0
            action = "SELL"

        # Portfolio value: if holding, show leveraged equity
        if btc_held > 0:
            raw_value = btc_held * price
            notional = btc_held * entry_price
            pnl = raw_value - notional
            equity = notional / leverage
            total_value = equity + pnl
            if total_value < 0:
                total_value = 0.0
        else:
            total_value = balance

        ledger.append(
            {
                "Date": date_str,
                "Price": price,
                "SMA7": sma7_val,
                "SMA30": sma30_val,
                "Action": action,
                "Balance": round(balance, 2),
                "BTC": round(btc_held, 6),
                "Total Value": round(total_value, 2),
            }
        )

    return pd.DataFrame(ledger)


def run_simulation(
    days: int = 60,
    volatility: float = 0.03,
    leverage: float = 1.0,
    initial_balance: float = 10000.0,
) -> dict[str, Any]:
    """High-level entry point that returns JSON-serializable results.

    Returns:
        Dictionary with keys: summary, chart_data, signals.
    """
    df = simulate_bitcoin_prices(days=days, volatility=volatility)
    df = calculate_moving_averages(df)
    ledger_df = simulate_trading(df, initial_balance=initial_balance, leverage=leverage)

    final_value = float(ledger_df["Total Value"].iloc[-1])
    profit = final_value - initial_balance
    roi = (profit / initial_balance) * 100 if initial_balance > 0 else 0.0

    # Chart data: dates, prices, SMA lines
    chart_data = {
        "dates": ledger_df["Date"].tolist(),
        "prices": [round(p, 2) for p in ledger_df["Price"].tolist()],
        "sma7": [
            round(v, 2) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
            for v in ledger_df["SMA7"].tolist()
        ],
        "sma30": [
            round(v, 2) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
            for v in ledger_df["SMA30"].tolist()
        ],
    }

    # Only BUY/SELL signals for the ledger table
    signals_df = ledger_df[ledger_df["Action"].isin(["BUY", "SELL"])]
    signals_subset = pd.DataFrame(
        signals_df[["Date", "Price", "Action", "Balance", "Total Value"]]
    )
    signals: list[dict[str, Any]] = signals_subset.to_dict(orient="records")  # type: ignore[assignment]

    summary = {
        "initial_balance": initial_balance,
        "final_value": round(final_value, 2),
        "profit": round(profit, 2),
        "roi": round(roi, 2),
        "leverage": leverage,
        "days": days,
        "volatility": volatility,
    }

    return {"summary": summary, "chart_data": chart_data, "signals": signals}


def print_performance(ledger_df: pd.DataFrame, initial_balance: float = 10000.0) -> None:
    """Print the daily ledger and final portfolio performance."""
    print("Daily Trading Ledger:")
    print("-" * 110)
    pd.set_option("display.max_rows", None)
    formatted_df = ledger_df.copy()
    formatted_df["Price"] = formatted_df["Price"].map("{:,.2f}".format)
    formatted_df["SMA7"] = formatted_df["SMA7"].map(
        lambda x: "{:,.2f}".format(x) if x is not None else "NaN"
    )
    formatted_df["SMA30"] = formatted_df["SMA30"].map(
        lambda x: "{:,.2f}".format(x) if x is not None else "NaN"
    )
    formatted_df["Balance"] = formatted_df["Balance"].map("{:,.2f}".format)
    formatted_df["BTC"] = formatted_df["BTC"].map("{:,.4f}".format)
    formatted_df["Total Value"] = formatted_df["Total Value"].map("{:,.2f}".format)

    print(
        formatted_df[
            ["Date", "Price", "SMA7", "SMA30", "Action", "Balance", "BTC", "Total Value"]
        ].to_string(index=False)
    )
    print("-" * 110)

    final_value = float(ledger_df["Total Value"].iloc[-1])
    profit = final_value - initial_balance
    roi = (profit / initial_balance) * 100

    print("Final Portfolio Performance:")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Value:     ${final_value:,.2f}")
    print(f"Total Profit:    ${profit:,.2f}")
    print(f"Total ROI:       {roi:.2f}%")


if __name__ == "__main__":
    initial_bal = 10000.0
    df = simulate_bitcoin_prices(days=60)
    df = calculate_moving_averages(df)
    ledger_df = simulate_trading(df, initial_balance=initial_bal, leverage=1.0)
    print_performance(ledger_df, initial_balance=initial_bal)
