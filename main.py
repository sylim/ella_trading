from data.collector import get_krx_universe
from screener.quant import score_universe
from backtest.engine import BacktestEngine
from backtest.report import print_summary, plot_portfolio


def run():
    print("KOSPI 유니버스 수집 중 (시가총액 상위 100)...")
    universe = get_krx_universe(market="KOSPI", top_n_by_marcap=100)
    print(f"총 {len(universe)}개 종목 선정")

    def screener_fn(rebalance_date: str):
        print(f"  스크리닝: {rebalance_date}")
        return score_universe(universe, rebalance_date, top_n=10)

    print("\n백테스팅 실행 중... (2022~2024, 분기 리밸런싱)")
    engine = BacktestEngine(initial_cash=10_000_000)
    result = engine.run(
        screener_fn=screener_fn,
        start="2022-01-01",
        end="2024-12-31",
        rebalance_period="QE",  # 분기 리밸런싱 (속도 최적화)
        top_n=10,
    )

    if result:
        print()
        print_summary(result)
        plot_portfolio(result, save_path="backtest_result.png")
    else:
        print("백테스팅 결과 없음 — 데이터를 확인하세요.")


if __name__ == "__main__":
    run()
