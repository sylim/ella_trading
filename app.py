"""
주식 포트폴리오 분석 대시보드
매일 자동 갱신 — Streamlit Cloud 배포용
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

# ── 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="대시보드",
    page_icon="📈",
    layout="wide",
)

PORTFOLIO = {
    "삼성전기":    "009150",
    "삼성전자":    "005930",
    "SK하이닉스":  "000660",
    "TIGER AI전력": "0117V0",
    "달바글로벌":  "483650",
    "HD현대중공업": "329180",
    "NAVER":      "035420",
}

START = "2024-01-01"
COLORS = px.colors.qualitative.Bold
KST = timezone(timedelta(hours=9))


def get_cache_key() -> str:
    """매일 오전 8:30 KST 기준으로 캐시 키 갱신"""
    now = datetime.now(KST)
    cutoff = now.replace(hour=8, minute=30, second=0, microsecond=0)
    base = now if now >= cutoff else now - timedelta(days=1)
    return base.strftime("%Y-%m-%d")


# ── 데이터 수집 (TTL 6시간 캐시) ─────────────────────────
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


@st.cache_data(ttl=21600, show_spinner=False)
def load_prices() -> pd.DataFrame:
    closes = {}
    for name, ticker in PORTFOLIO.items():
        try:
            df = fdr.DataReader(ticker, START)
            if not df.empty:
                closes[name] = df["Close"]
                continue
        except Exception:
            pass
        df = get_naver_chart(ticker)
        if not df.empty:
            closes[name] = df[df.index >= START]["Close"]
    return pd.DataFrame(closes).ffill().dropna(how="all")


@st.cache_data(ttl=21600, show_spinner=False)
def get_valuation(ticker: str) -> dict:
    try:
        r = requests.get(
            f"https://m.stock.naver.com/api/stock/{ticker}/integration",
            timeout=5, headers={"User-Agent": "Mozilla/5.0"}
        )
        result = {"PER": 0.0, "PBR": 0.0, "ROE": 0.0, "EPS": 0.0, "현재가": 0.0}
        data = r.json()
        for item in data.get("totalInfos", []):
            code = item.get("code", "")
            raw = item.get("value", "0").replace("배","").replace(",","").replace("원","").replace("%","").strip()
            try:
                val = float(raw)
            except ValueError:
                val = 0.0
            if code == "per":   result["PER"] = val
            elif code == "pbr": result["PBR"] = val
            elif code == "roe": result["ROE"] = val
            elif code == "eps": result["EPS"] = val
        # 현재가
        price_raw = data.get("closePrice", "0").replace(",", "")
        try:
            result["현재가"] = float(price_raw)
        except Exception:
            pass
        return result
    except Exception:
        return {"PER": 0.0, "PBR": 0.0, "ROE": 0.0, "EPS": 0.0, "현재가": 0.0}


# ── 기술 지표 ───────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_momentum(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days:
        return np.nan
    return (s.iloc[-1] / s.iloc[-days] - 1) * 100


def calc_sharpe(series: pd.Series, rf: float = 0.03) -> float:
    r = series.pct_change().dropna()
    ex = r - rf / 252
    return round((ex.mean() / ex.std()) * np.sqrt(252), 2) if ex.std() > 0 else 0.0


def get_signal(rsi, pbr, mom3) -> tuple[str, str]:
    signals, color = [], "gray"
    if not np.isnan(rsi):
        if rsi >= 70:
            signals.append("🔴 RSI 과매수")
            color = "red"
        elif rsi <= 30:
            signals.append("🟢 RSI 과매도(매수기회)")
            color = "green"
    if pbr > 0:
        if pbr < 1:   signals.append("🟢 PBR 저평가")
        elif pbr > 3: signals.append("🟠 PBR 고평가")
    if not np.isnan(mom3):
        if mom3 > 15:   signals.append("⬆ 모멘텀 강")
        elif mom3 < -15: signals.append("⬇ 모멘텀 약")
    return (" | ".join(signals) if signals else "⚪ 중립"), color


# ── 메인 UI ─────────────────────────────────────────────
def main():
    st.title("📈주식 대시보드3")
    st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} KST · 6시간마다 자동 갱신")

    with st.spinner("데이터 수집 중..."):
        prices = load_prices()
        valuations = {name: get_valuation(ticker) for name, ticker in PORTFOLIO.items()
                      if name in prices.columns}

    names = list(prices.columns)
    val_df = pd.DataFrame(valuations).T

    # 모멘텀 & 지표
    mom_data, sharpe_data, rsi_latest = {}, {}, {}
    for name in names:
        s = prices[name].dropna()
        mom_data[name] = {"1M": calc_momentum(s,21), "3M": calc_momentum(s,63),
                          "6M": calc_momentum(s,126), "12M": calc_momentum(s,252)}
        sharpe_data[name] = calc_sharpe(s)
        rsi_s = calc_rsi(s)
        rsi_latest[name] = rsi_s.iloc[-1] if len(rsi_s) > 0 else np.nan
    mom_df = pd.DataFrame(mom_data).T

    # ── 상단 KPI 카드 ─────────────────────────────────────
    st.subheader("종목별 현황")
    cols = st.columns(len(names))
    for i, name in enumerate(names):
        ticker = PORTFOLIO[name]
        s = prices[name].dropna()
        cur = s.iloc[-1]
        chg_1d = (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) >= 2 else 0
        chg_1m = calc_momentum(s, 21)
        rsi = rsi_latest.get(name, np.nan)
        signal, _ = get_signal(rsi, val_df.loc[name, "PBR"] if name in val_df.index else 0,
                               mom_df.loc[name, "3M"] if name in mom_df.index else np.nan)
        with cols[i]:
            st.metric(
                label=name,
                value=f"{cur:,.0f}",
                delta=f"{chg_1d:+.1f}% (1일)",
            )
            st.caption(f"1M: {chg_1m:+.1f}% | RSI: {rsi:.0f}")
            st.caption(signal)

    st.divider()

    # ── 탭 레이아웃 ───────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 누적 수익률", "🌡️ 모멘텀", "💰 밸류에이션", "📡 RSI 시그널", "🕯️ 개별 차트"
    ])

    # ── Tab 1: 누적 수익률 ─────────────────────────────────
    with tab1:
        st.subheader("누적 수익률 비교 (시작=100)")
        norm = prices / prices.ffill().iloc[0] * 100
        fig = go.Figure()
        for i, name in enumerate(names):
            s = norm[name].dropna()
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, name=name,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>"
            ))
        fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=500, hovermode="x unified",
                          yaxis_title="지수 (시작=100)", template="plotly_dark",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: 모멘텀 히트맵 ──────────────────────────────
    with tab2:
        st.subheader("기간별 모멘텀 히트맵")
        mom_plot = mom_df[["1M","3M","6M","12M"]].rename(
            columns={"1M":"1개월","3M":"3개월","6M":"6개월","12M":"12개월"})

        fig = go.Figure(go.Heatmap(
            z=mom_plot.values.astype(float),
            x=mom_plot.columns.tolist(),
            y=mom_plot.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "N/A"
                   for v in row] for row in mom_plot.values],
            texttemplate="%{text}",
            textfont=dict(size=13, color="black"),
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        ))
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # 모멘텀 순위 바
        st.subheader("3개월 모멘텀 순위")
        mom3_s = mom_df["3M"].sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(
            y=mom3_s.index, x=mom3_s.values, orientation="h",
            marker_color=["#d73027" if v < 0 else "#1a9641" for v in mom3_s],
            text=[f"{v:.1f}%" for v in mom3_s], textposition="outside",
        ))
        fig2.add_vline(x=0, line_color="white", line_width=1)
        fig2.update_layout(height=350, template="plotly_dark", xaxis_title="수익률 (%)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: 밸류에이션 ─────────────────────────────────
    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("PBR (낮을수록 저평가)")
            pbr_s = val_df["PBR"].replace(0, np.nan).dropna().sort_values()
            if not pbr_s.empty:
                fig = go.Figure(go.Bar(
                    y=pbr_s.index, x=pbr_s.values, orientation="h",
                    marker_color=["#d73027" if v > 3 else "#4575b4" for v in pbr_s],
                    text=[f"{v:.2f}배" for v in pbr_s], textposition="outside",
                ))
                fig.add_vline(x=1, line_dash="dash", line_color="green",
                              annotation_text="PBR=1 (장부가)", annotation_position="top right")
                fig.add_vline(x=3, line_dash="dash", line_color="red",
                              annotation_text="PBR=3 (고평가)", annotation_position="top right")
                fig.update_layout(height=350, template="plotly_dark", xaxis_title="PBR")
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("PER (낮을수록 저평가)")
            per_s = val_df["PER"].replace(0, np.nan).dropna().sort_values()
            if not per_s.empty:
                fig = go.Figure(go.Bar(
                    y=per_s.index, x=per_s.values, orientation="h",
                    marker_color=["#d73027" if v > 25 else "#4575b4" for v in per_s],
                    text=[f"{v:.1f}배" for v in per_s], textposition="outside",
                ))
                fig.add_vline(x=15, line_dash="dash", line_color="green",
                              annotation_text="PER=15 (적정)", annotation_position="top right")
                fig.add_vline(x=25, line_dash="dash", line_color="red",
                              annotation_text="PER=25 (고평가)", annotation_position="top right")
                fig.update_layout(height=350, template="plotly_dark", xaxis_title="PER")
                st.plotly_chart(fig, use_container_width=True)

        # 종합 밸류 테이블
        st.subheader("밸류에이션 + 샤프지수 종합")
        table_data = []
        for name in names:
            v = valuations.get(name, {})
            rsi = rsi_latest.get(name, np.nan)
            mom3 = mom_df.loc[name, "3M"] if name in mom_df.index else np.nan
            sig, _ = get_signal(rsi, v.get("PBR", 0), mom3)
            table_data.append({
                "종목": name,
                "현재가": f"{prices[name].dropna().iloc[-1]:,.0f}",
                "PBR": f"{v.get('PBR',0):.2f}배" if v.get('PBR',0) > 0 else "-",
                "PER": f"{v.get('PER',0):.1f}배" if v.get('PER',0) > 0 else "-",
                "EPS": f"{v.get('EPS',0):,.0f}원" if v.get('EPS',0) > 0 else "-",
                "3M수익률": f"{mom3:+.1f}%" if not np.isnan(mom3) else "-",
                "샤프지수": f"{sharpe_data.get(name,0):.2f}",
                "RSI": f"{rsi:.1f}" if not np.isnan(rsi) else "-",
                "시그널": sig,
            })
        st.dataframe(pd.DataFrame(table_data).set_index("종목"), use_container_width=True, height=320)

    # ── Tab 4: RSI 시그널 ─────────────────────────────────
    with tab4:
        st.subheader("RSI(14) 매매 시그널")
        rsi_s = pd.Series(rsi_latest).reindex(names)
        colors_rsi = ["#d73027" if v >= 70 else "#1a9641" if v <= 30 else "#4575b4"
                      for v in rsi_s.fillna(50)]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rsi_s.index, y=rsi_s.values,
            marker_color=colors_rsi,
            text=[f"{v:.1f}" for v in rsi_s.values],
            textposition="outside",
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="70 — 과매수 (매도 고려)", annotation_position="top right")
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      annotation_text="30 — 과매도 (매수 기회)", annotation_position="bottom right")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(height=450, template="plotly_dark",
                          yaxis=dict(range=[0, 105], title="RSI"))
        st.plotly_chart(fig, use_container_width=True)

        # RSI 추이 라인
        st.subheader("RSI 추이 (최근 60거래일)")
        fig2 = go.Figure()
        for i, name in enumerate(names):
            s = prices[name].dropna()
            rsi_series = calc_rsi(s).tail(60)
            fig2.add_trace(go.Scatter(
                x=rsi_series.index, y=rsi_series.values, name=name,
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            ))
        fig2.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.08, line_width=0)
        fig2.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.08, line_width=0)
        fig2.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.6)
        fig2.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.6)
        fig2.update_layout(height=400, template="plotly_dark", hovermode="x unified",
                           yaxis=dict(range=[0,100], title="RSI"),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 5: 개별 캔들 차트 ────────────────────────────
    with tab5:
        st.subheader("개별 종목 차트")
        selected = st.selectbox("종목 선택", names)
        ticker = PORTFOLIO[selected]

        # 캔들 데이터
        try:
            ohlcv = fdr.DataReader(ticker, START)
        except Exception:
            ohlcv = get_naver_chart(ticker)
        ohlcv = ohlcv[ohlcv.index >= START].copy()

        if not ohlcv.empty:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                row_heights=[0.55, 0.25, 0.20],
                                vertical_spacing=0.03)

            # 캔들스틱
            fig.add_trace(go.Candlestick(
                x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
                low=ohlcv["Low"], close=ohlcv["Close"],
                name=selected, increasing_line_color="#e74c3c",
                decreasing_line_color="#3498db",
            ), row=1, col=1)

            # 이동평균선
            for window, color_ma in [(20,"#f39c12"),(60,"#9b59b6"),(120,"#1abc9c")]:
                ma = ohlcv["Close"].rolling(window).mean()
                fig.add_trace(go.Scatter(
                    x=ohlcv.index, y=ma, name=f"MA{window}",
                    line=dict(color=color_ma, width=1.2), opacity=0.8
                ), row=1, col=1)

            # 거래량
            vol_colors = ["#e74c3c" if c >= o else "#3498db"
                          for c, o in zip(ohlcv["Close"], ohlcv["Open"])]
            fig.add_trace(go.Bar(
                x=ohlcv.index, y=ohlcv["Volume"],
                marker_color=vol_colors, name="거래량", showlegend=False,
            ), row=2, col=1)

            # RSI
            rsi_line = calc_rsi(ohlcv["Close"])
            fig.add_trace(go.Scatter(
                x=rsi_line.index, y=rsi_line.values,
                name="RSI", line=dict(color="#e67e22", width=1.5)
            ), row=3, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1,
                          line_width=0, row=3, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1,
                          line_width=0, row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          opacity=0.5, row=3, col=1)

            fig.update_layout(
                height=700, template="plotly_dark",
                title=f"{selected} ({ticker})",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.05),
            )
            fig.update_yaxes(title_text="가격", row=1, col=1)
            fig.update_yaxes(title_text="거래량", row=2, col=1)
            fig.update_yaxes(title_text="RSI", range=[0,100], row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("⚠️ 이 대시보드는 투자 참고용입니다. 투자 결정은 본인 책임입니다.")


if __name__ == "__main__":
    main()
