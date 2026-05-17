import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from typing import Callable


class BacktestEngine:
    """
    분기 리밸런싱 백테스팅 엔진

    사용법:
        engine = BacktestEngine(initial_cash=10_000_000)
        result = engine.run(
            screener_fn=my_screener,
            start="2022-01-01",
            end="2024-12-31",
        )
    """

    def __init__(self, initial_cash: float = 10_000_000):
        self.initial_cash = initial_cash

    def run(
        self,
        screener_fn: Callable[[str], pd.DataFrame],
        start: str,
        end: str,
        rebalance_period: str = "QE",
        top_n: int = 10,
        fee_rate: float = 0.00015,
        tax_rate: float = 0.0018,
    ) -> dict:
        rebalance_dates = pd.date_range(start, end, freq=rebalance_period)
        portfolio = {}
        cash = self.initial_cash
        portfolio_values = []

        for date in rebalance_dates:
            date_str = date.strftime("%Y-%m-%d")

            # 스크리너 실행 (리밸런싱 날짜 기준 모멘텀 계산)
            selected = screener_fn(date_str)
            if selected.empty:
                continue
            tickers = selected["ticker"].tolist()[:top_n]

            # 기존 포지션 청산
            total_value = cash
            for ticker, shares in portfolio.items():
                price = self._get_price(ticker, date_str)
                if price:
                    proceeds = shares * price * (1 - tax_rate)
                    total_value += proceeds
            portfolio = {}
            cash = total_value

            # 동일비중 매수
            per_stock = cash / len(tickers) if tickers else 0
            for ticker in tickers:
                price = self._get_price(ticker, date_str)
                if price and price > 0:
                    shares = int(per_stock * (1 - fee_rate) / price)
                    cost = shares * price * (1 + fee_rate)
                    if cost <= cash:
                        portfolio[ticker] = shares
                        cash -= cost

            # 포트폴리오 가치 기록
            pv = cash + sum(
                shares * (self._get_price(ticker, date_str) or 0)
                for ticker, shares in portfolio.items()
            )
            portfolio_values.append({"date": date, "value": pv})

        return self._calculate_metrics(portfolio_values)

    def _get_price(self, ticker: str, date: str) -> float | None:
        try:
            start = (pd.to_datetime(date) - pd.DateOffset(days=5)).strftime("%Y-%m-%d")
            df = fdr.DataReader(ticker, start, date)
            if df.empty:
                return None
            return float(df["Close"].iloc[-1])
        except Exception:
            return None

    def _calculate_metrics(self, portfolio_values: list) -> dict:
        if not portfolio_values:
            return {}

        df = pd.DataFrame(portfolio_values).set_index("date")
        df["return"] = df["value"].pct_change()

        total_return = (df["value"].iloc[-1] / self.initial_cash - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365
        cagr = ((df["value"].iloc[-1] / self.initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

        rolling_max = df["value"].cummax()
        drawdown = (df["value"] - rolling_max) / rolling_max
        mdd = drawdown.min() * 100

        rf = 0.03 / 4  # 분기 무위험수익률
        excess_returns = df["return"].dropna() - rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(4) if excess_returns.std() > 0 else 0

        return {
            "portfolio_values": df,
            "total_return": round(total_return, 2),
            "cagr": round(cagr, 2),
            "mdd": round(mdd, 2),
            "sharpe": round(sharpe, 2),
            "initial_cash": self.initial_cash,
            "final_value": round(df["value"].iloc[-1], 0),
        }
