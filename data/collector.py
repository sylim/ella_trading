import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta


def get_krx_universe(market: str = "KOSPI", top_n_by_marcap: int = 100) -> pd.DataFrame:
    """
    KOSPI/KOSDAQ 종목 리스트 반환
    시가총액 상위 top_n_by_marcap 종목만 반환 (데이터 수집 속도 최적화)
    """
    df = fdr.StockListing(market)
    df = df.rename(columns={"Code": "ticker", "Name": "name", "Marcap": "marcap"})
    df = df[["ticker", "name", "marcap"]].dropna()
    df["marcap"] = pd.to_numeric(df["marcap"], errors="coerce")
    df = df.sort_values("marcap", ascending=False).head(top_n_by_marcap)
    return df.reset_index(drop=True)


def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """일별 OHLCV 데이터 반환"""
    try:
        df = fdr.DataReader(ticker, start, end)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def get_recent_business_day(offset_days: int = 0) -> str:
    """가장 최근 영업일 반환 (주말 제외)"""
    day = datetime.today() - timedelta(days=offset_days)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.strftime("%Y-%m-%d")
