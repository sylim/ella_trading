import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


def print_summary(result: dict):
    print("=" * 40)
    print("   백테스팅 결과 요약")
    print("=" * 40)
    print(f"초기 자본    : {result['initial_cash']:>15,.0f} 원")
    print(f"최종 자산    : {result['final_value']:>15,.0f} 원")
    print(f"총 수익률    : {result['total_return']:>14.2f} %")
    print(f"연환산 수익률: {result['cagr']:>14.2f} %")
    print(f"최대 낙폭    : {result['mdd']:>14.2f} %")
    print(f"샤프지수     : {result['sharpe']:>14.2f}")
    print("=" * 40)


def plot_portfolio(result: dict, save_path: str = None):
    df = result["portfolio_values"]

    _set_korean_font()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # 누적 수익률
    cumret = (df["value"] / df["value"].iloc[0] - 1) * 100
    ax1.plot(cumret.index, cumret.values, color="#1f77b4", linewidth=2)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.fill_between(cumret.index, cumret.values, 0,
                     where=(cumret.values >= 0), alpha=0.1, color="blue")
    ax1.fill_between(cumret.index, cumret.values, 0,
                     where=(cumret.values < 0), alpha=0.1, color="red")
    ax1.set_title("누적 수익률 (%)", fontsize=14)
    ax1.set_ylabel("%")
    ax1.grid(True, alpha=0.3)

    # 드로우다운
    rolling_max = df["value"].cummax()
    drawdown = (df["value"] - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax2.set_title("드로우다운 (%)", fontsize=12)
    ax2.set_ylabel("%")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"차트 저장: {save_path}")
    else:
        plt.show()


def _set_korean_font():
    font_candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            return
    plt.rcParams["font.family"] = "sans-serif"
