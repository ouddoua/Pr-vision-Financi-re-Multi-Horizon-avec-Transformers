"""Backtest long/cash des prédictions LSTM à 5 jours.

Depuis la racine du projet : python -m src.backtest
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import FEATURES, L, TARGETS, LSTMModel, make_windows


HORIZON = 5
DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0


def simulate_backtest(
    dates,
    close_prices,
    predictions,
    horizon=HORIZON,
    transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS,
    slippage_bps=DEFAULT_SLIPPAGE_BPS,
    threshold=0.0,
):
    """Simulate non-overlapping long/cash trades from close to close."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    close_prices = np.asarray(close_prices, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if horizon < 1:
        raise ValueError("horizon doit être supérieur ou égal à 1")
    if len(dates) != len(close_prices) or len(predictions) > len(dates) - horizon:
        raise ValueError(
            "dates et close_prices doivent avoir la même longueur, et dates doit "
            "inclure les cours futurs nécessaires aux sorties"
        )
    if not np.isfinite(close_prices).all() or (close_prices <= 0).any():
        raise ValueError("Les prix de clôture doivent être finis et strictement positifs")
    if not np.isfinite(predictions).all():
        raise ValueError("Les prédictions doivent être finies")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("Les coûts et le slippage ne peuvent pas être négatifs")

    n_periods = len(predictions) // horizon
    if n_periods == 0:
        raise ValueError("Pas assez de données pour un trade complet")

    cost_per_side = (transaction_cost_bps + slippage_bps) / 10_000
    rows = []
    equity = 1.0
    previous_position = 0.0

    for period in range(n_periods):
        start = period * horizon
        end = start + horizon
        signal = predictions[start]
        position = float(signal > threshold)
        gross_return = close_prices[end] / close_prices[start] - 1.0
        turnover = abs(position - previous_position)
        transaction_cost = cost_per_side * turnover
        net_return = (1.0 + position * gross_return) * (1.0 - cost_per_side) ** turnover - 1.0
        equity *= 1.0 + net_return

        rows.append(
            {
                "date": dates.iloc[start],
                "exit_date": dates.iloc[end],
                "prediction": signal,
                "position": int(position),
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "equity": equity,
            }
        )
        previous_position = position

    # Close any remaining long position at the final price.
    exit_cost = cost_per_side * previous_position
    if exit_cost:
        equity *= 1.0 - cost_per_side
        rows[-1]["transaction_cost"] += exit_cost
        rows[-1]["net_return"] = (1.0 + rows[-1]["net_return"]) * (1.0 - cost_per_side) - 1.0
        rows[-1]["equity"] = equity

    trades = pd.DataFrame(rows)
    period_returns = trades["net_return"].to_numpy()
    equity_curve = np.concatenate(([1.0], trades["equity"].to_numpy()))
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_peak - 1.0
    standard_deviation = period_returns.std(ddof=1) if len(period_returns) > 1 else 0.0
    sharpe = (
        period_returns.mean() / standard_deviation * np.sqrt(252 / horizon)
        if standard_deviation > 0
        else 0.0
    )

    metrics = {
        "cumulative_return": equity - 1.0,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdowns.min(),
        "number_of_periods": len(trades),
        "number_of_long_periods": int(trades["position"].sum()),
    }
    return trades, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--model", default="data/lstm_model.pt")
    parser.add_argument("--output", default="results/backtest_lstm_5d.csv")
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    test = pd.read_csv(args.test_csv, parse_dates=["Date"])
    if "Close" not in test.columns or "Date" not in test.columns:
        raise ValueError("Le CSV de test doit contenir les colonnes Date et Close")

    X_test, _ = make_windows(test, L, FEATURES, TARGETS)
    model = LSTMModel(n_features=X_test.shape[2])
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(X_test, dtype=torch.float32)).numpy()[:, 1]

    # make_windows labels row i with a window ending at i-1; execute at row i's close.
    decision_rows = np.arange(L, len(test) - HORIZON)
    usable = min(len(predictions), len(decision_rows))
    predictions = predictions[:usable]

    trades, metrics = simulate_backtest(
        dates=test["Date"].iloc[L:].reset_index(drop=True),
        close_prices=test["Close"].to_numpy()[L:],
        predictions=predictions,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        threshold=args.threshold,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output, index=False)

    print("Backtest LSTM 5 jours (long/cash)")
    print(f"Rendement cumulé : {metrics['cumulative_return']:.2%}")
    print(f"Sharpe annualisé : {metrics['sharpe_ratio']:.3f}")
    print(f"Max drawdown     : {metrics['max_drawdown']:.2%}")
    print(f"Périodes investies : {metrics['number_of_long_periods']}/{metrics['number_of_periods']}")
    print(f"Résultats        : {output}")


if __name__ == "__main__":
    main()