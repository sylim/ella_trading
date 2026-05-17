"""
주식 포트폴리오 분석 대시보드 — 모바일 최적화
매일 오전 8:30 KST 자동 갱신
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta, timezone

st.set_page_config(
    page_title="SY 트레이딩 대시보드",
    page_icon="📈",
    layout="wide",
)

PORTFOLIO = {
    "삼성전기":     "009150",
    "삼성전자":     "005930",
    "SK하이닉스":   "000660",
    "TIGER AI전력": "0117V0",
    "달바글로벌":   "483650",
    "HD현대중공업":  "329180",
    "NAVER":       "035420",
}

START  = "2024-01-01"
COLORS = px.colors.qualitative.Bold
KST    = timezone(timedelta(hours=9))


# ── 캐시 키 (매일 8:30 KST 갱신) ───────────────────────
def get_cache_key() -> str:
    now = datetime.now(KST)
    cutoff = now.replace(hour=8, minute=30, second=0, microsecond=0)
    base = now if now >= cutoff else now - timedelta(days=1)
    return base.strftime("%Y-%m-%d")


# ── 데이터 수집 ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_naver_chart(ticker: str, count: int = 400, cache_key: str = "") -> pd.DataFrame:
    url = f"https://fchart.stock.naver.com/sise.nhn?symbol={ticker}&timeframe=day&count={count}&requestType=0"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.text)
        rows = []
        for item in root.findall(".//item"):
            parts = item.get("data", "").split("|")
            if len(parts) >= 5:
                rows.append({"Date": parts[0], "Open": parts[1], "High": parts[2],
                             "Low": parts[3], "Close": parts[4],
                             "Volume": parts[5] if len(parts) > 5 else 0})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df.apply(pd.to_numeric, errors="coerce").sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_prices(cache_key: str = "") -> pd.DataFrame:
    closes = {}
    for name, ticker in PORTFOLIO.items():
        try:
            df = fdr.DataReader(ticker, START)
            if not df.empty:
                closes[name] = df["Close"]
                continue
        except Exception:
            pass
        df = get_naver_chart(ticker, cache_key=cache_key)
        if not df.empty:
            closes[name] = df[df.index >= START]["Close"]
    return pd.DataFrame(closes).ffill().dropna(how="all")


@st.cache_data(show_spinner=False)
def load_ohlcv(ticker: str, cache_key: str = "") -> pd.DataFrame:
    try:
        df = fdr.DataReader(ticker, START)
        if not df.empty:
            return df[df.index >= START]
    except Exception:
        pass
    df = get_naver_chart(ticker, cache_key=cache_key)
    return df[df.index >= START] if not df.empty else pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_valuation(ticker: str, cache_key: str = "") -> dict:
    try:
        r = requests.get(
            f"https://m.stock.naver.com/api/stock/{ticker}/integration",
            timeout=5, headers={"User-Agent": "Mozilla/5.0"}
        )
        result = {"PER": 0.0, "PBR": 0.0, "ROE": 0.0, "EPS": 0.0}
        for item in r.json().get("totalInfos", []):
            code = item.get("code", "")
            raw  = item.get("value", "0").replace("배","").replace(",","").replace("원","").replace("%","").strip()
            try:
                val = float(raw)
            except ValueError:
                val = 0.0
            if code == "per":   result["PER"] = val
            elif code == "pbr": result["PBR"] = val
            elif code == "roe": result["ROE"] = val
            elif code == "eps": result["EPS"] = val
        return result
    except Exception:
        return {"PER": 0.0, "PBR": 0.0, "ROE": 0.0, "EPS": 0.0}


# ── 기술 지표 ───────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_momentum(series: pd.Series, days: int) -> float:
    s = series.dropna()
    return (s.iloc[-1] / s.iloc[-days] - 1) * 100 if len(s) >= days else np.nan


def calc_sharpe(series: pd.Series) -> float:
    r  = series.pct_change().dropna()
    ex = r - 0.03 / 252
    return round((ex.mean() / ex.std()) * np.sqrt(252), 2) if ex.std() > 0 else 0.0


def signal_badge(rsi, pbr, mom3) -> list[tuple[str,str]]:
    """(텍스트, 설명) 리스트 반환"""
    badges = []
    if not np.isnan(rsi):
        if rsi >= 70:
            badges.append(("🔴 RSI 과매수", f"RSI {rsi:.0f} — 단기 과열, 신규 매수 신중히"))
        elif rsi <= 30:
            badges.append(("🟢 RSI 과매도", f"RSI {rsi:.0f} — 단기 낙폭 과대, 매수 기회 탐색"))
        else:
            badges.append(("🔵 RSI 중립", f"RSI {rsi:.0f}"))
    if pbr > 0:
        if pbr < 1:
            badges.append(("🟢 PBR 저평가", f"PBR {pbr:.2f} — 장부가 이하, 구조적 저평가"))
        elif pbr > 3:
            badges.append(("🟠 PBR 고평가", f"PBR {pbr:.2f} — 고성장 기대 반영, 실적 미달 시 급락 위험"))
    if not np.isnan(mom3):
        if mom3 > 15:
            badges.append(("⬆ 모멘텀 강", f"3개월 수익률 {mom3:+.1f}%"))
        elif mom3 < -15:
            badges.append(("⬇ 모멘텀 약", f"3개월 수익률 {mom3:+.1f}%"))
    return badges


# ── 개별 종목 탭 ────────────────────────────────────────
def render_stock_tab(name: str, ticker: str, prices: pd.DataFrame,
                     valuation: dict, ck: str):
    s    = prices[name].dropna()
    ohlcv = load_ohlcv(ticker, cache_key=ck)
    rsi_s = calc_rsi(s)
    rsi   = rsi_s.iloc[-1] if len(rsi_s) > 0 else np.nan
    mom1  = calc_momentum(s, 21)
    mom3  = calc_momentum(s, 63)
    mom6  = calc_momentum(s, 126)
    mom12 = calc_momentum(s, 252)
    sharpe = calc_sharpe(s)
    pbr   = valuation.get("PBR", 0)
    per   = valuation.get("PER", 0)
    cur   = s.iloc[-1]
    chg1d = (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) >= 2 else 0

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", f"{cur:,.0f}원", f"{chg1d:+.1f}% (1일)")
    c2.metric("RSI(14)", f"{rsi:.1f}" if not np.isnan(rsi) else "-",
              "과매수" if rsi >= 70 else ("과매도" if rsi <= 30 else "중립"))
    c3.metric("PBR", f"{pbr:.2f}배" if pbr > 0 else "-",
              "고평가" if pbr > 3 else ("저평가" if 0 < pbr < 1 else ""))
    c4.metric("샤프지수", f"{sharpe:.2f}")

    # 시그널 배지
    badges = signal_badge(rsi, pbr, mom3)
    if badges:
        st.markdown("**📌 시그널**")
        for badge, desc in badges:
            st.info(f"**{badge}** — {desc}")

    st.divider()

    # 모멘텀 요약
    st.markdown("**📊 수익률 요약**")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("1개월", f"{mom1:+.1f}%" if not np.isnan(mom1) else "-")
    mc2.metric("3개월", f"{mom3:+.1f}%" if not np.isnan(mom3) else "-")
    mc3.metric("6개월", f"{mom6:+.1f}%" if not np.isnan(mom6) else "-")
    mc4.metric("12개월", f"{mom12:+.1f}%" if not np.isnan(mom12) else "-")

    st.divider()

    # 캔들 차트 + 이동평균 + 거래량 + RSI
    if not ohlcv.empty:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.20, 0.25],
            vertical_spacing=0.03,
        )
        # 캔들
        fig.add_trace(go.Candlestick(
            x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
            low=ohlcv["Low"], close=ohlcv["Close"], name=name,
            increasing_line_color="#e74c3c", decreasing_line_color="#3498db",
        ), row=1, col=1)
        # 이동평균
        for w, col_ma in [(20,"#f39c12"), (60,"#9b59b6"), (120,"#1abc9c")]:
            ma = ohlcv["Close"].rolling(w).mean()
            fig.add_trace(go.Scatter(
                x=ohlcv.index, y=ma, name=f"MA{w}",
                line=dict(color=col_ma, width=1.2), opacity=0.8,
            ), row=1, col=1)
        # 거래량
        vol_c = ["#e74c3c" if c >= o else "#3498db"
                 for c, o in zip(ohlcv["Close"], ohlcv["Open"])]
        fig.add_trace(go.Bar(
            x=ohlcv.index, y=ohlcv["Volume"],
            marker_color=vol_c, name="거래량", showlegend=False,
        ), row=2, col=1)
        # RSI
        rsi_full = calc_rsi(ohlcv["Close"])
        fig.add_trace(go.Scatter(
            x=rsi_full.index, y=rsi_full.values,
            name="RSI", line=dict(color="#e67e22", width=1.5),
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.1, line_width=0, row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.1, line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        fig.update_layout(
            height=600, template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.05, font=dict(size=10)),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        fig.update_yaxes(title_text="가격",   row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0,100], row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # 밸류에이션 상세
    st.markdown("**💰 밸류에이션**")
    vc1, vc2 = st.columns(2)
    vc1.metric("PBR", f"{pbr:.2f}배" if pbr > 0 else "-",
               help="주가 / 주당순자산. 1 미만이면 장부가 이하 → 저평가 가능성. 3 초과면 고성장 기대 반영.")
    vc2.metric("PER", f"{per:.1f}배" if per > 0 else "-",
               help="주가 / 주당순이익. 15 미만이면 이익 대비 저평가. 25 초과면 고평가 주의.")


# ── 전체 비교 탭 ────────────────────────────────────────
def render_overview_tab(names, prices, mom_df, rsi_latest, sharpe_data, valuations):
    st.subheader("누적 수익률 비교")
    norm = prices / prices.ffill().iloc[0] * 100
    fig = go.Figure()
    for i, name in enumerate(names):
        s = norm[name].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=400, template="plotly_dark", hovermode="x unified",
                      yaxis_title="지수 (시작=100)",
                      legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("종목별 종합 현황")
    rows = []
    for name in names:
        v     = valuations.get(name, {})
        rsi   = rsi_latest.get(name, np.nan)
        mom3  = mom_df.loc[name, "3M"] if name in mom_df.index else np.nan
        badges = [b[0] for b in signal_badge(rsi, v.get("PBR",0), mom3)]
        rows.append({
            "종목":     name,
            "현재가":   f"{prices[name].dropna().iloc[-1]:,.0f}",
            "3M수익률": f"{mom3:+.1f}%" if not np.isnan(mom3) else "-",
            "RSI":     f"{rsi:.0f}" if not np.isnan(rsi) else "-",
            "PBR":     f"{v.get('PBR',0):.2f}" if v.get('PBR',0) > 0 else "-",
            "PER":     f"{v.get('PER',0):.1f}" if v.get('PER',0) > 0 else "-",
            "샤프":    f"{sharpe_data.get(name,0):.2f}",
            "시그널":  " | ".join(badges) if badges else "⚪ 중립",
        })
    st.dataframe(pd.DataFrame(rows).set_index("종목"), use_container_width=True)


# ── 메인 ────────────────────────────────────────────────
def main():
    st.title("📈 SY 트레이딩 대시보드")
    ck      = get_cache_key()
    now_kst = datetime.now(KST)
    st.caption(f"기준일: {ck} · 매일 오전 8:30 KST 갱신 · 현재: {now_kst.strftime('%H:%M')} KST")

    with st.spinner("데이터 수집 중..."):
        prices     = load_prices(cache_key=ck)
        valuations = {name: get_valuation(ticker, cache_key=ck)
                      for name, ticker in PORTFOLIO.items() if name in prices.columns}

    names = list(prices.columns)

    mom_data, sharpe_data, rsi_latest = {}, {}, {}
    for name in names:
        s = prices[name].dropna()
        mom_data[name]    = {"1M": calc_momentum(s,21), "3M": calc_momentum(s,63),
                             "6M": calc_momentum(s,126), "12M": calc_momentum(s,252)}
        sharpe_data[name] = calc_sharpe(s)
        rsi_s             = calc_rsi(s)
        rsi_latest[name]  = rsi_s.iloc[-1] if len(rsi_s) > 0 else np.nan
    mom_df = pd.DataFrame(mom_data).T

    # 탭: 전체 개요 + 종목별
    tab_labels = ["📊 전체 비교"] + [f"{name}" for name in names]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview_tab(names, prices, mom_df, rsi_latest, sharpe_data, valuations)

    for i, name in enumerate(names):
        with tabs[i + 1]:
            render_stock_tab(
                name, PORTFOLIO[name], prices,
                valuations.get(name, {}), ck
            )

    st.divider()
    st.caption("⚠️ 투자 참고용입니다. 투자 결정은 본인 책임입니다.")


if __name__ == "__main__":
    main()
