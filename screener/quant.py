import pandas as pd
import numpy as np
import FinanceDataReader as fdr


def calculate_momentum(ticker: str, end_date: str, months: int = 12) -> float:
    """
    12-1 모멘텀: 12개월 수익률에서 최근 1개월 제외
    과거 연구에서 가장 안정적인 모멘텀 팩터
    """
    end = pd.to_datetime(end_date)
    start = end - pd.DateOffset(months=months)
    skip = end - pd.DateOffset(months=1)

    try:
        df = fdr.DataReader(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df.empty or len(df) < 20:
            return np.nan

        price_start = df["Close"].iloc[0]
        df_skip = df[df.index <= skip]
        price_skip = df_skip["Close"].iloc[-1] if not df_skip.empty else df["Close"].iloc[-1]

        return (price_skip - price_start) / price_start
    except Exception:
        return np.nan


def score_universe(universe: pd.DataFrame, rebalance_date: str, top_n: int = 10) -> pd.DataFrame:
    """
    팩터 스코어링 후 상위 N종목 선별

    팩터:
      - 12-1 모멘텀 (가격 기반)
      - 시가총액 (대형주 선호 — 유동성 확보)
    """
    results = []
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        mom = calculate_momentum(ticker, rebalance_date)
        results.append({
            "ticker": ticker,
            "name": row["name"],
            "marcap": row.get("marcap", 0),
            "momentum": mom,
        })

    df = pd.DataFrame(results).dropna(subset=["momentum"])
    if df.empty:
        return pd.DataFrame()

    df["score_momentum"] = _rank_normalize(df["momentum"])
    df["score_marcap"] = _rank_normalize(df["marcap"])

    # 모멘텀 70% + 시가총액 30%
    df["total_score"] = df["score_momentum"] * 0.7 + df["score_marcap"] * 0.3

    return df.nlargest(top_n, "total_score").reset_index(drop=True)


def _rank_normalize(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)
