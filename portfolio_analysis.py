"""
보유 종목 종합 분석 대시보드
- 누적 수익률 비교
- 모멘텀 히트맵 (1/3/6/12개월)
- RSI 매매 시그널
- PBR/PER 밸류에이션 (Naver 데이터)
- 종목별 샤프지수
"""

import warnings
warnings.filterwarnings("ignore")

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import requests
import os
from xml.etree import ElementTree as ET

# ── 보유 종목 ──────────────────────────────────────────
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


# ── 한국어 폰트 ─────────────────────────────────────────
def set_korean_font():
    for p in ["/System/Library/Fonts/Supplemental/AppleGothic.ttf",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            return

set_korean_font()
plt.rcParams["axes.unicode_minus"] = False


# ── 가격 데이터 수집 ────────────────────────────────────
def get_naver_chart(ticker: str, count: int = 400) -> pd.DataFrame:
    """Naver 차트 API — FDR에서 지원하지 않는 종목 대응"""
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


def load_prices() -> pd.DataFrame:
    closes = {}
    for name, ticker in PORTFOLIO.items():
        # FDR 시도
        try:
            df = fdr.DataReader(ticker, START)
            if not df.empty:
                closes[name] = df["Close"]
                print(f"  ✓ {name} ({ticker}): {len(df)}일 [FDR]")
                continue
        except Exception:
            pass

        # Naver 차트 API 대체
        df = get_naver_chart(ticker)
        if not df.empty:
            df = df[df.index >= START]
            closes[name] = df["Close"]
            print(f"  ✓ {name} ({ticker}): {len(df)}일 [Naver]")
        else:
            print(f"  ✗ {name}: 데이터 없음")

    return pd.DataFrame(closes).dropna(how="all")


# ── Naver 밸류에이션 ───────────────────────────────────
def get_valuation(ticker: str) -> dict:
    try:
        r = requests.get(
            f"https://m.stock.naver.com/api/stock/{ticker}/integration",
            timeout=5, headers={"User-Agent": "Mozilla/5.0"}
        )
        result = {"PER": 0.0, "PBR": 0.0, "ROE": 0.0, "EPS": 0.0}
        for item in r.json().get("totalInfos", []):
            code = item.get("code", "")
            raw = item.get("value", "0").replace("배","").replace(",","").replace("원","").replace("%","").strip()
            try:
                val = float(raw)
            except ValueError:
                val = 0.0
            if code == "per":
                result["PER"] = val
            elif code == "pbr":
                result["PBR"] = val
            elif code == "roe":
                result["ROE"] = val
            elif code == "eps":
                result["EPS"] = val
        return result
    except Exception:
        return {"PER": 0.0, "PBR": 0.0, "ROE": 0.0, "EPS": 0.0}


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


def calc_sharpe(series: pd.Series, rf_annual: float = 0.03) -> float:
    r = series.pct_change().dropna()
    rf_d = rf_annual / 252
    ex = r - rf_d
    return round((ex.mean() / ex.std()) * np.sqrt(252), 2) if ex.std() > 0 else 0.0


# ── 메인 ────────────────────────────────────────────────
def run():
    print("보유 종목 가격 데이터 수집 중...")
    prices = load_prices()
    names = list(prices.columns)

    print("\n밸류에이션 수집 중 (Naver)...")
    valuations = {}
    for name, ticker in PORTFOLIO.items():
        if name in names:
            v = get_valuation(ticker)
            valuations[name] = v
            print(f"  {name}: PER={v['PER']}, PBR={v['PBR']}, ROE={v['ROE']}%")

    # 모멘텀 & 샤프
    mom_data, sharpe_data, rsi_current = {}, {}, {}
    for name in names:
        s = prices[name].dropna()
        mom_data[name] = {"1M": calc_momentum(s,21), "3M": calc_momentum(s,63),
                          "6M": calc_momentum(s,126), "12M": calc_momentum(s,252)}
        sharpe_data[name] = calc_sharpe(s)
        rsi_s = calc_rsi(s)
        rsi_current[name] = rsi_s.iloc[-1] if not rsi_s.empty else np.nan

    mom_df = pd.DataFrame(mom_data).T
    val_df = pd.DataFrame(valuations).T if valuations else pd.DataFrame()

    # ── 차트 ────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 26))
    fig.suptitle("보유 종목 종합 분석 대시보드", fontsize=18, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # ① 누적 수익률 비교
    ax1 = fig.add_subplot(gs[0, :])
    norm = prices / prices.ffill().iloc[0] * 100
    for i, name in enumerate(names):
        s = norm[name].dropna()
        ax1.plot(s.index, s.values, label=name, color=colors[i], linewidth=2)
    ax1.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("① 누적 수익률 비교 (시작=100)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("지수 (시작=100)")
    ax1.legend(loc="upper left", fontsize=9, ncol=4)
    ax1.grid(True, alpha=0.3)

    # ② 모멘텀 히트맵
    ax2 = fig.add_subplot(gs[1, 0])
    mom_plot = mom_df[["1M", "3M", "6M", "12M"]]
    im = ax2.imshow(mom_plot.values.astype(float), cmap="RdYlGn", aspect="auto", vmin=-40, vmax=40)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(["1개월","3개월","6개월","12개월"])
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(mom_plot.index)
    for i in range(len(names)):
        for j in range(4):
            val = mom_plot.iloc[i, j]
            txt = f"{val:.1f}%" if not np.isnan(val) else "N/A"
            color = "white" if abs(val) > 20 else "black"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    ax2.set_title("② 기간별 모멘텀 (%)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # ③ 샤프지수
    ax3 = fig.add_subplot(gs[1, 1])
    sharpe_s = pd.Series(sharpe_data).sort_values()
    bar_c = ["#d73027" if v < 0 else "#4575b4" for v in sharpe_s]
    bars = ax3.barh(sharpe_s.index, sharpe_s.values, color=bar_c, edgecolor="white")
    ax3.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, sharpe_s.values):
        ha = "left" if val >= 0 else "right"
        ax3.text(val + (0.05 if val >= 0 else -0.05),
                 bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", ha=ha, fontsize=9)
    ax3.set_title("③ 샤프지수 (연환산, Rf=3%)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("샤프지수"); ax3.grid(True, alpha=0.3, axis="x")

    # ④ PBR 밸류에이션
    ax4 = fig.add_subplot(gs[2, 0])
    if not val_df.empty:
        pbr_s = val_df["PBR"].replace(0, np.nan).dropna().sort_values()
        if not pbr_s.empty:
            bar_c4 = ["#d73027" if v > 3 else "#4575b4" for v in pbr_s]
            ax4.barh(pbr_s.index, pbr_s.values, color=bar_c4, edgecolor="white")
            ax4.axvline(1.0, color="green", linestyle="--", linewidth=1.5, label="PBR=1 (장부가)")
            ax4.axvline(3.0, color="red", linestyle="--", linewidth=1.5, label="PBR=3 (고평가 경계)")
            for i, (name, val) in enumerate(pbr_s.items()):
                ax4.text(val+0.1, i, f"{val:.2f}배", va="center", fontsize=9)
            ax4.set_title("④ PBR — 낮을수록 저평가\n(파랑:적정 / 빨강:고평가)", fontsize=12, fontweight="bold")
            ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis="x")

    # ⑤ PER 밸류에이션
    ax5 = fig.add_subplot(gs[2, 1])
    if not val_df.empty:
        per_s = val_df["PER"].replace(0, np.nan).dropna().sort_values()
        if not per_s.empty:
            bar_c5 = ["#d73027" if v > 25 else "#4575b4" for v in per_s]
            ax5.barh(per_s.index, per_s.values, color=bar_c5, edgecolor="white")
            ax5.axvline(15, color="green", linestyle="--", linewidth=1.5, label="PER=15 (적정)")
            ax5.axvline(25, color="red", linestyle="--", linewidth=1.5, label="PER=25 (고평가 경계)")
            for i, (name, val) in enumerate(per_s.items()):
                ax5.text(val+0.3, i, f"{val:.1f}배", va="center", fontsize=9)
            ax5.set_title("⑤ PER — 낮을수록 저평가\n(파랑:적정 / 빨강:고평가)", fontsize=12, fontweight="bold")
            ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3, axis="x")

    # ⑥ RSI 신호
    ax6 = fig.add_subplot(gs[3, :])
    rsi_s = pd.Series(rsi_current).reindex(names)
    rsi_colors = ["#d73027" if v >= 70 else "#1a9641" if v <= 30 else "#4575b4"
                  for v in rsi_s.fillna(50)]
    bars6 = ax6.bar(rsi_s.index, rsi_s.values, color=rsi_colors, edgecolor="white", width=0.5)
    ax6.axhline(70, color="red", linestyle="--", linewidth=1.5, label="RSI 70 (과매수 — 매도 고려)")
    ax6.axhline(30, color="green", linestyle="--", linewidth=1.5, label="RSI 30 (과매도 — 매수 기회)")
    ax6.axhline(50, color="gray", linestyle=":", linewidth=0.8)
    for bar, val in zip(bars6, rsi_s.values):
        if not np.isnan(val):
            signal = " ▲매수" if val <= 30 else (" ▼매도" if val >= 70 else "")
            ax6.text(bar.get_x()+bar.get_width()/2, val+1.5,
                     f"{val:.1f}{signal}", ha="center", fontsize=9, fontweight="bold")
    ax6.set_ylim(0, 100)
    ax6.set_title("⑥ RSI(14) 매매 시그널", fontsize=13, fontweight="bold")
    ax6.set_ylabel("RSI"); ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3, axis="y")

    plt.savefig("portfolio_dashboard.png", dpi=150, bbox_inches="tight")
    print("\n차트 저장 완료: portfolio_dashboard.png")

    # ── 텍스트 요약 ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("   종목별 종합 시그널 요약")
    print("=" * 65)
    print(f"{'종목':<14} {'3M모멘텀':>8} {'PBR':>6} {'PER':>7} {'RSI':>6} {'샤프':>6}  시그널")
    print("-" * 65)
    for name in names:
        mom3  = mom_df.loc[name, "3M"] if name in mom_df.index else float("nan")
        pbr   = val_df.loc[name, "PBR"] if (not val_df.empty and name in val_df.index) else 0
        per   = val_df.loc[name, "PER"] if (not val_df.empty and name in val_df.index) else 0
        rsi   = rsi_current.get(name, float("nan"))
        sharpe = sharpe_data.get(name, 0)

        # 시그널 판단
        signals = []
        if not np.isnan(rsi):
            if rsi >= 70: signals.append("RSI과매수")
            elif rsi <= 30: signals.append("RSI과매도")
        if pbr > 0:
            if pbr < 1: signals.append("PBR저평가")
            elif pbr > 3: signals.append("PBR고평가")
        if not np.isnan(mom3):
            if mom3 > 15: signals.append("모멘텀강")
            elif mom3 < -15: signals.append("모멘텀약")
        sig_str = " | ".join(signals) if signals else "중립"

        print(f"{name:<14} {mom3:>7.1f}%  {pbr:>5.2f}  {per:>6.1f}  {rsi:>5.1f}  {sharpe:>5.2f}  {sig_str}")
    print("=" * 65)
    print("\n[해석 기준]")
    print("  RSI ≥ 70  → 과매수, 단기 매도/관망 고려")
    print("  RSI ≤ 30  → 과매도, 매수 기회 탐색")
    print("  PBR < 1   → 장부가 이하 (구조적 저평가 가능성)")
    print("  PBR > 3   → 성장 프리미엄 또는 고평가 주의")
    print("  PER < 15  → 이익 대비 저평가")
    print("  PER > 25  → 고성장 기대 반영, 실적 미달 시 급락 위험")


if __name__ == "__main__":
    run()
