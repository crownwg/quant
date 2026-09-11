"""A 股低频选股策略回测 —— 命令行入口。

用法示例
--------
# 1. 经典 120 日动量，月度调仓，次日开盘成交
python -m quant.main --codes "600031,000858,002557" --start 20200101 --end 20240909 --use-open

# 2. 反转策略（跌得多的反弹），周度调仓，选 2 只
python -m quant.main --codes "600031,000858,002557" --strategy reversal --rebalance W --top-n 2

# 3. 多因子：动量 1.0 + 低波动 0.5，用排名法合成
python -m quant.main --codes "600031,000858,002557" \
    --factors "momentum=1,low_volatility=0.5" --standardize rank

# 4. 开启全部实盘约束：涨跌停 + 停牌 + 流动性过滤
python -m quant.main --codes "600031,000858,002557" --min-volume 5000000 --capital 1000000

# 5. 股票池自动建池（不用手敲代码）：沪深300 动量，选前 20
python -m quant.main --pool hs300 --start 20200101 --end 20240909 --use-open --top-n 20

# 6. 多池合并 + 概念板块：沪深300 与消费板块并集，反转策略
python -m quant.main --pool "hs300,concept:白酒" --strategy reversal --top-n 15

# 7. 手动 + 池混合：在沪深300里叠加两只自定义标的
python -m quant.main --codes 600031,000858 --pool zz500 --top-n 10

输出文件
--------
  equity_curve.csv     净值曲线（画图用）
  metrics.csv          评价指标
  rebalance_plan.csv   模拟调仓清单（可对着下单）
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import factors, filters
from .backtest import run
from .data import load_panel, load_index
from .rebalance import build_plan, summarize
from .report import build_report
from .strategy import factor_weights
from . import universe
from . import rolling
from . import attribution
from . import grid
from . import compare
from . import combine


# ----------------------------------------------------------- 因子构造

def _f_momentum(a, prices, volume):
    return factors.momentum(prices, a.lookback, a.skip_recent)


def _f_reversal(a, prices, volume):
    return factors.reversal(prices, a.reversal_lookback)


def _f_ma_trend(a, prices, volume):
    return factors.ma_trend(prices, a.ma_short, a.ma_long)


def _f_ma_breakout(a, prices, volume):
    return factors.ma_breakout(prices, a.ma_window)


def _f_low_volatility(a, prices, volume):
    return factors.low_volatility(prices, a.vol_lookback)


def _f_volume_trend(a, prices, volume):
    return factors.volume_trend(volume, a.vol_short, a.vol_long)


FACTOR_BUILDERS = {
    "momentum": _f_momentum,
    "reversal": _f_reversal,
    "ma_trend": _f_ma_trend,
    "ma_breakout": _f_ma_breakout,
    "low_volatility": _f_low_volatility,
    "volume_trend": _f_volume_trend,
}


def parse_factor_spec(text: str) -> dict[str, float]:
    """解析 "momentum=1,low_volatility=0.5" 形式的因子权重。"""
    spec: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, weight = item.partition("=")
        name = name.strip()
        if name not in FACTOR_BUILDERS:
            raise ValueError(f"未知因子 {name!r}，可选: {sorted(FACTOR_BUILDERS)}")
        spec[name] = spec.get(name, 0.0) + (float(weight) if weight.strip() else 1.0)
    if not spec:
        raise ValueError("--factors 为空")
    return spec


def build_score(args, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """根据参数构造打分表：单因子直接用，多因子合成。"""
    if args.factors:
        spec = parse_factor_spec(args.factors)
        built = {name: FACTOR_BUILDERS[name](args, prices, volume) for name in spec}
        return factors.combine(built, spec, args.standardize)
    if args.strategy not in FACTOR_BUILDERS:
        raise ValueError(f"未知策略 {args.strategy!r}，可选: {sorted(FACTOR_BUILDERS)}")
    return FACTOR_BUILDERS[args.strategy](args, prices, volume)


# ------------------------------------------------------------- 输出

PCT_METRICS = {
    "total_return", "annual_return", "max_drawdown", "annual_volatility", "win_rate",
    "average_daily_turnover", "annual_turnover",
    "total_commission", "total_stamp_tax", "total_cost", "cost_drag_annual",
    "excess_return",
}
INT_METRICS = {"max_drawdown_days", "trading_days"}

GROUPS = [
    ("收益", ["total_return", "annual_return", "final_equity"]),
    ("风险", ["max_drawdown", "max_drawdown_days", "annual_volatility"]),
    ("风险调整收益", ["sharpe", "sortino", "calmar"]),
    ("交易特征", ["win_rate", "profit_loss_ratio", "average_daily_turnover", "annual_turnover"]),
    ("交易成本（占本金比例）", ["total_commission", "total_stamp_tax", "total_cost", "cost_drag_annual"]),
    ("相对基准", ["excess_return", "information_ratio"]),
    ("样本", ["trading_days", "years"]),
]

LABELS = {
    "total_return": "总收益",
    "annual_return": "年化收益",
    "final_equity": "期末净值",
    "max_drawdown": "最大回撤",
    "max_drawdown_days": "最长回撤天数",
    "annual_volatility": "年化波动率",
    "sharpe": "夏普比率",
    "sortino": "索提诺比率",
    "calmar": "卡玛比率",
    "win_rate": "日胜率",
    "profit_loss_ratio": "盈亏比",
    "average_daily_turnover": "日均双边换手",
    "annual_turnover": "年化双边换手",
    "total_commission": "佣金总额",
    "total_stamp_tax": "印花税总额",
    "total_cost": "成本总额",
    "cost_drag_annual": "成本年化拖累",
    "excess_return": "超额收益",
    "information_ratio": "信息比率",
    "trading_days": "交易日数",
    "years": "回测年数",
}


def fmt(key: str, value) -> str:
    if key in PCT_METRICS:
        return f"{value:.4%}"
    if key in INT_METRICS:
        return f"{int(value)}"
    return f"{value:.4f}"


def pool_display(codes: list[str], head: int = 12) -> str:
    """股票池显示：超过 head 只时截断为「前 N 只 等」。"""
    if len(codes) <= head:
        return f"{len(codes)}只 {','.join(codes)}"
    return f"{len(codes)}只 {','.join(codes[:head])} 等"


def _build_today_plan(plan: pd.DataFrame, prices: pd.DataFrame,
                      weights: pd.DataFrame, args) -> pd.DataFrame:
    """生成「今日调仓清单」，支持两种模式：

    1. 只指定 --today：从完整 plan 里筛选该日调仓动作。
    2. 指定 --holdings (CSV: code,shares)：按今日目标权重 × 实际持仓市值
       → 目标股数，与实际持仓对比，输出「该买/该卖多少股」。

    返回精简版 DataFrame：code / action / current_shares / target_shares /
    delta_shares / price / amount。
    """
    today_ts = pd.Timestamp(args.today) if args.today else prices.index[-1]
    if today_ts not in weights.index:
        avail = weights.index[weights.index <= today_ts]
        if avail.empty:
            return pd.DataFrame()
        today_ts = avail[-1]

    holdings_df = None
    if args.holdings:
        holdings_df = pd.read_csv(args.holdings)
        if "code" not in holdings_df.columns or "shares" not in holdings_df.columns:
            raise ValueError("--holdings CSV 必须包含 code, shares 两列")
        holdings_df["code"] = holdings_df["code"].astype(str).str.zfill(6)

    target_w = weights.loc[today_ts]
    target_w = target_w[target_w > 1e-6]
    if target_w.empty and holdings_df is None:
        return pd.DataFrame()

    exec_price = prices.loc[today_ts]

    # 当前总市值（holdings 给的）+ 现金作底
    if holdings_df is not None:
        cur_value = 0.0
        for row in holdings_df.itertuples():
            p = exec_price.get(row.code, np.nan)
            if not pd.isna(p) and p > 0:
                cur_value += int(row.shares) * float(p)
        total_capital = cur_value if cur_value > 0 else args.capital
    else:
        total_capital = args.capital

    # 模式 1: 只指定 --today → 从 plan 里直接筛
    if args.today and not args.holdings:
        out = plan[plan["date"] == today_ts.date()].copy()
        if out.empty:
            print(f"  ({today_ts.date()} 不是调仓日，无清单)")
            return out
        # 重命名为精简字段
        out = out[["code", "action", "delta_weight", "price", "shares", "amount"]].rename(
            columns={"shares": "delta_shares"}
        )
        out["current_shares"] = 0
        out["target_shares"] = out["delta_shares"].where(
            out["delta_shares"] > 0, 0
        ).astype(int)
        return out.reset_index(drop=True)

    # 模式 2/3: 用 holdings 算 delta
    rows = []
    codes = set(target_w.index)
    if holdings_df is not None:
        codes |= set(holdings_df["code"])
    for code in sorted(codes):
        w = float(target_w.get(code, 0.0))
        target_amount = w * total_capital
        price = float(exec_price.get(code, np.nan))
        if pd.isna(price) or price <= 0:
            continue
        target_shares = int(target_amount / price / 100) * 100
        cur_shares = 0
        if holdings_df is not None:
            m = holdings_df[holdings_df["code"] == code]
            if not m.empty:
                cur_shares = int(m["shares"].iloc[0])
        delta_shares = target_shares - cur_shares
        if abs(delta_shares) < 100:
            continue
        if delta_shares > 0:
            action = "建仓" if cur_shares == 0 else "加仓"
        else:
            action = "清仓" if target_shares == 0 else "减仓"
        amount = abs(delta_shares) * price
        rows.append({
            "code": code,
            "action": action,
            "current_shares": cur_shares,
            "target_shares": target_shares,
            "delta_shares": delta_shares,
            "price": round(price, 4),
            "amount": round(amount, 2),
        })
    return pd.DataFrame(rows).sort_values(["action", "code"]).reset_index(drop=True)


def print_metrics(metrics: dict) -> None:
    for title, keys in GROUPS:
        print(f"\n【{title}】")
        for key in keys:
            if key in metrics:
                print(f"  {LABELS.get(key, key):<14} {fmt(key, metrics[key])}")


# ------------------------------------------------------------- 主流程

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A 股低频选股策略回测")
    p.add_argument("--codes", default="",
                   help="逗号分隔的股票代码，如 600031,000858,002557；与 --pool 二选一或合并")
    p.add_argument("--pool", default="",
                   help="股票池：指数池名(hs300/zz500/cyb/消费/白酒/蓝筹...)、6位指数代码(000300)、"
                        "或 concept:白酒 概念板块；可逗号合并，如 hs300,消费")
    p.add_argument("--start", default="20200101")
    p.add_argument("--end", default="20251231")

    # 选股
    p.add_argument("--strategy", default="momentum",
                   choices=sorted(FACTOR_BUILDERS), help="单因子策略，默认 momentum")
    p.add_argument("--factors", default="",
                   help="多因子合成，如 momentum=1,low_volatility=0.5；给定时覆盖 --strategy")
    p.add_argument("--standardize", default="zscore", choices=["zscore", "rank"],
                   help="多因子标准化方式：zscore 保留分布信息，rank 抗极端值")
    p.add_argument("--top-n", type=int, default=10, help="每次调仓选前几名")
    p.add_argument("--rebalance", default="M", help="调仓频率：M 月 / W 周 / Q 季 / 2M 双月")
    p.add_argument("--min-names", type=int, default=1,
                   help="有效候选少于该数量时当天空仓，避免被迫满仓垃圾票")
    p.add_argument("--buffer", type=int, default=0,
                   help="换手缓冲：老持仓排名仍在 top_n+buffer 内就继续持有，抑制频繁对倒")

    # 因子参数
    p.add_argument("--lookback", type=int, default=120, help="动量因子回看天数")
    p.add_argument("--skip-recent", type=int, default=0, help="动量跳过最近 N 日（12-1 动量）")
    p.add_argument("--reversal-lookback", type=int, default=20, help="反转因子回看天数")
    p.add_argument("--ma-short", type=int, default=20)
    p.add_argument("--ma-long", type=int, default=60)
    p.add_argument("--ma-window", type=int, default=60, help="均线突破因子窗口")
    p.add_argument("--vol-lookback", type=int, default=60, help="低波动因子窗口")
    p.add_argument("--vol-short", type=int, default=5)
    p.add_argument("--vol-long", type=int, default=60)

    # 成交与过滤
    p.add_argument("--use-open", action="store_true",
                   help="次日开盘价成交（消除前视偏差，推荐常开）")
    p.add_argument("--min-volume", type=float, default=0.0,
                   help="流动性门槛：近 20 日均成交量（股）低于该值则不可交易")
    p.add_argument("--no-limit-filter", action="store_true", help="关闭涨跌停过滤")
    p.add_argument("--no-suspend-filter", action="store_true", help="关闭停牌过滤")
    p.add_argument("--st-codes", default="", help="ST 股代码，逗号分隔（涨跌停幅度按 5%% 计）")

    # 成本与本金
    p.add_argument("--capital", type=float, default=1_000_000.0, help="组合本金（调仓清单用）")
    p.add_argument("--fee", type=float, default=0.0003, help="佣金费率（双边）")
    p.add_argument("--stamp-tax", type=float, default=0.0005, help="印花税率（仅卖出）")

    # 输出
    p.add_argument("--out-prefix", default="", help="输出文件名前缀")
    p.add_argument("--save-daily", action="store_true", help="额外保存每日明细 daily_detail.csv")
    p.add_argument("--benchmark", default="sh000300",
                   help="基准指数 symbol，如 sh000300 沪深300；传空串 --benchmark '' 关闭对比")
    p.add_argument("--rolling", action="store_true",
                   help="启用滚动 / walk-forward 样本外回测，检验策略稳健性（输出 rolling_*.csv）")
    p.add_argument("--roll-window", type=int, default=504,
                   help="滚动回测每段长度（交易日），默认 504≈2年；step 等于 window（不重叠）")

    # 风险归因
    p.add_argument("--attribution", action="store_true",
                   help="风险归因：按市场状态(牛/熊/震荡)拆分策略收益，看策略靠什么行情吃饭（需配合基准指数）")
    p.add_argument("--regime-window", type=int, default=60,
                   help="市场状态判定窗口（交易日），默认 60≈3个月；过去 N 日基准收益决定牛熊")
    p.add_argument("--regime-band", type=float, default=0.05,
                   help="牛熊阈值：基准过去 N 日收益 ≥ +band 为牛、≤ -band 为熊，默认 0.05")

    # 参数优化（网格搜索）
    p.add_argument("--grid", action="store_true",
                   help="参数优化：在 (lookback × top_n × buffer) 网格上回测，找稳健最优参数")
    p.add_argument("--grid-lookbacks", default="60,90,120,160",
                   help="网格 lookback 候选，逗号分隔，默认 60,90,120,160")
    p.add_argument("--grid-topn", default="10,15,20",
                   help="网格 top_n 候选，逗号分隔，默认 10,15,20")
    p.add_argument("--grid-buffers", default="0,2",
                   help="网格 buffer 候选，逗号分隔，默认 0,2")

    # 多策略对比
    p.add_argument("--compare", action="store_true",
                   help="多策略对比：同一池子横向跑多个预定义策略，挑稳健组合")
    p.add_argument("--compare-strategies", default="",
                   help="对比策略列表，逗号分隔，如 momentum_120,reversal_20；为空用默认 7 个")

    # 多池组合配置
    p.add_argument("--combine", default="",
                   help="多池组合：'消费=reversal_20,蓝筹=low_vol_60,医药=ma_trend' 池子=策略映射")
    p.add_argument("--combine-weights", default="",
                   help="池子间权重，逗号分隔，按 --combine 顺序；缺省等权")

    # 每日再平衡清单（实盘化）
    p.add_argument("--today", default="",
                   help="只看这一天的调仓清单（YYYYMMDD），输出精简版 today_plan.csv；省略则输出完整周期")
    p.add_argument("--holdings", default="",
                   help="当前持仓 CSV（含 code,shares 两列），用于计算「目标 vs 实际」的 delta 清单")
    return p


def main() -> None:
    args = make_parser().parse_args()

    manual = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
    pool_codes = universe.build_universe(args.pool) if args.pool.strip() else []
    # --combine 自带池子，单独模式不需要 --codes/--pool
    if not manual and not pool_codes and not args.combine:
        raise SystemExit("必须提供 --codes 或 --pool（至少其一）；或使用 --combine 直接指定池子")
    # 合并去重保序
    seen: set[str] = set()
    codes: list[str] = []
    for c in manual + pool_codes:
        if c not in seen:
            seen.add(c)
            codes.append(c)
    print(f"股票池合计 {len(codes)} 只" + (f"（手动 {len(manual)} + 池 {len(pool_codes)}）" if pool_codes else ""), flush=True)

    # 预热期：因子需要历史数据才能计算，往前多取一段，否则开头几个月空仓
    warm_days = max(args.lookback, args.ma_long, args.vol_lookback, args.vol_long) * 2 + 30
    fetch_start = (pd.Timestamp(args.start) - pd.Timedelta(days=warm_days)).strftime("%Y%m%d")

    # 大池首次下载较慢：池子大时降低单只间隔并允许跳过失败标的
    sleep = 0.3 if len(codes) > 20 else 1.0
    on_error = "skip" if (pool_codes or len(codes) > 20) else "raise"
    prices_all = pd.DataFrame(); open_all = pd.DataFrame(); volume_all = pd.DataFrame()
    if codes:  # --combine 模式下 codes 为空，跳过面板加载（池子各自在 combine 里加载）
        panel = load_panel(codes, fetch_start, args.end, sleep=sleep, on_error=on_error)
        prices_all = panel["close"]
        open_all = panel["open"]
        volume_all = panel["volume"]
        if len(prices_all.columns) < len(codes):
            skipped = len(codes) - len(prices_all.columns)
            print(f"⚠️ {skipped} 只代码数据不可用已跳过，实际参与回测 {len(prices_all.columns)} 只")
        if prices_all.empty:
            raise SystemExit("没有任何可用数据，回测终止")

    score = build_score(args, prices_all, volume_all)

    # ---- 交易可行性约束 ----
    st_codes = [c.strip() for c in args.st_codes.split(",") if c.strip()]
    limit_pct = filters.limit_pct_by_code(codes, st_codes)
    illiquid = filters.illiquid_mask(volume_all, args.min_volume)
    can_buy, can_sell = filters.tradability(
        prices_all, volume_all, limit_pct=limit_pct, illiquid=illiquid,
        enable_limit=not args.no_limit_filter,
        enable_suspend=not args.no_suspend_filter,
    )

    # ---- 参数优化：网格搜索（在加载面板后、单次回测前分叉）----
    if args.grid:
        lookbacks = grid.parse_ints(args.grid_lookbacks)
        top_ns = grid.parse_ints(args.grid_topn)
        buffers = grid.parse_ints(args.grid_buffers)
        bench = None
        if args.benchmark:
            try:
                bench = load_index(args.benchmark, args.start, args.end)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 基准获取失败，网格不计算超额: {exc}")
        ncomb = len(lookbacks) * len(top_ns) * len(buffers)
        print(f"\n【参数优化】网格 {len(lookbacks)}×{len(top_ns)}×{len(buffers)} = {ncomb} 组，池子 {len(codes)} 只")
        start_ts = pd.Timestamp(args.start)
        res = grid.grid_search(prices_all, open_all, can_buy, can_sell, args,
                               lookbacks, top_ns, buffers, start_ts, benchmark_curve=bench)
        res.to_csv(f"{args.out_prefix}grid_results.csv", index=False)
        sub = (f"池子 {len(codes)}只 · 区间 {args.start}~{args.end} · "
               f"网格 {len(lookbacks)}×{len(top_ns)}×{len(buffers)} · 基准 {args.benchmark or '无'}")
        grid.build_grid_report(res, f"{args.out_prefix}grid_report.html",
                               title="参数优化网格搜索", subtitle=sub)
        print("\n【Top 按夏普】")
        for _, r in res.sort_values("sharpe", ascending=False).head(10).iterrows():
            er = f" 超额 {r['excess_return']:.1%} IR {r['information_ratio']:.2f}" if "excess_return" in r else ""
            print(f"  lb={int(r['lookback'])} n={int(r['top_n'])} buf={int(r['buffer'])}"
                  f" | 收益 {r['total_return']:.1%} 夏普 {r['sharpe']:.2f}"
                  f" 回撤 {r['max_drawdown']:.1%} 最差年 {r['worst_year_return']:.1%}"
                  f" 正年 {r['positive_year_ratio']:.0%}{er}")
        print(f"\n网格结果已保存: {args.out_prefix}grid_results.csv / {args.out_prefix}grid_report.html")
        return

    # ---- 多策略对比：在加载面板后、单次回测前分叉 ----
    if args.compare:
        strategies = compare.parse_strategy_spec(args.compare_strategies)
        print(f"\n【多策略对比】共 {len(strategies)} 个策略：{strategies}")
        # start_ts 在 grid 分支里临时定义过，这里再算一次供 compare 用
        start_ts = pd.Timestamp(args.start)
        # 加载基准（如果提供），用于计算每个策略的超额/信息比率
        bench = None
        if args.benchmark:
            try:
                idx_full = load_index(args.benchmark, fetch_start, args.end)
                idx_close = idx_full[idx_full.index >= start_ts]
                bench = (idx_close / idx_close.iloc[0]).reindex(
                    prices_all.loc[prices_all.index >= start_ts].index
                ).ffill().bfill()
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 基准获取失败，对比不含超额: {exc}")
                bench = None
        results_df, equity_dict, annual_dict, params_dict = compare.run_compare(
            prices_all, open_all, volume_all, can_buy, can_sell,
            strategies, args, benchmark_curve=bench,
        )
        results_df.to_csv(f"{args.out_prefix}compare_results.csv", index=False)
        sub = (f"池子 {len(codes)}只 · 区间 {args.start}~{args.end} · "
               f"{len(strategies)}策略 · 基准 {args.benchmark or '无'}")
        compare.build_compare_report(
            results_df, equity_dict, annual_dict, params_dict,
            f"{args.out_prefix}compare_report.html",
            title="多策略横向对比", subtitle=sub,
        )
        print("\n【Top 按夏普】")
        for _, r in results_df.sort_values("sharpe", ascending=False).iterrows():
            er = f" 超额 {r['excess_return']:.1%} IR {r['information_ratio']:.2f}" if bench is not None else ""
            print(f"  {r['strategy']:18s} 收益 {r['total_return']:6.1%}"
                  f" 夏普 {r['sharpe']:5.2f} 回撤 {r['max_drawdown']:6.1%}{er}")
        print(f"\n对比结果已保存: {args.out_prefix}compare_results.csv"
              f" / {args.out_prefix}compare_report.html")
        return

    # ---- 多池组合配置（在 grid/compare 之后、单次回测之前分叉）----
    if args.combine:
        start_ts = pd.Timestamp(args.start)
        spec = combine.parse_combine_spec(args.combine)
        # 加载基准（如果提供）—— --combine 模式下 prices_all 为空，直接用 idx_full 自己的索引
        bench = None
        if args.benchmark:
            try:
                idx_full = load_index(args.benchmark, fetch_start, args.end)
                idx_close = idx_full[idx_full.index >= start_ts]
                if not idx_close.empty:
                    bench = (idx_close / idx_close.iloc[0]).reindex(idx_close.index).ffill().bfill()
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 基准获取失败，组合不含超额: {exc}")
                bench = None
        pool_results, _, pool_equity, corr, combo_eq, combo_m, weights = combine.run_combine(
            spec, args.combine_weights, args, fetch_start, benchmark_curve=bench,
        )
        pool_results.to_csv(f"{args.out_prefix}combine_results.csv", index=False)
        # 相关性矩阵单独存 CSV（heatmap 用）
        corr.to_csv(f"{args.out_prefix}combine_correlation.csv")
        combo_eq.rename("equity").to_csv(f"{args.out_prefix}combine_equity.csv")
        weights_str = "/".join(f"{w:.0%}" for w in weights)
        sub = (f"区间 {args.start}~{args.end} · "
 f"{len(spec)} 池 · 权重 {weights_str} · 基准 {args.benchmark or '无'}")
        combine.build_combine_report(
            pool_results, pool_equity, corr, combo_eq, combo_m, weights,
            f"{args.out_prefix}combine_report.html",
            title="多池组合配置", subtitle=sub,
        )
        print(f"\n【组合整体】总收益 {combo_m['total_return']:.1%} 年化 {combo_m['annual_return']:.1%}"
              f" 夏普 {combo_m['sharpe']:.2f} 回撤 {combo_m['max_drawdown']:.1%}")
        if bench is not None:
            print(f"  超额 {combo_m.get('excess_return', 0):.1%} IR {combo_m.get('information_ratio', 0):.2f}")
        print(f"\n组合结果已保存: {args.out_prefix}combine_results.csv"
              f" / {args.out_prefix}combine_correlation.csv"
              f" / {args.out_prefix}combine_equity.csv"
              f" / {args.out_prefix}combine_report.html")
        return

    weights_all = factor_weights(score, top_n=args.top_n, freq=args.rebalance,
                                 min_names=args.min_names, buffer=args.buffer,
                                 can_buy=can_buy, can_sell=can_sell)

    # 截掉预热期，只在用户指定区间上评价
    start_ts = pd.Timestamp(args.start)
    keep = prices_all.index >= start_ts
    prices = prices_all.loc[keep]
    open_prices = open_all.loc[keep]
    weights = weights_all.loc[keep]
    can_buy_r, can_sell_r = can_buy.loc[keep], can_sell.loc[keep]

    if prices.empty:
        raise SystemExit(f"区间 {args.start}~{args.end} 内没有数据")

    equity, metrics, detail = run(prices, weights,
                                  open_prices=open_prices if args.use_open else None,
                                  fee=args.fee, stamp_tax=args.stamp_tax)

    # ---- 分年度稳健性分解（默认输出）----
    annual = rolling.annual_breakdown(equity, detail)
    annual.to_csv(f"{args.out_prefix}annual_metrics.csv", index=False)

    # ---- 滚动 / walk-forward 样本外回测 ----
    rolling_equity = None
    rolling_table = None
    if args.rolling:
        n = len(prices_all)
        windows = rolling.rolling_windows(n, window=args.roll_window, step=args.roll_window)
        # 仅保留窗口起始日期 >= 用户回测起点，避免用到预热期之外的无效段
        windows = [(s, e) for (s, e) in windows if prices_all.index[s] >= start_ts]
        if windows:
            roll_results = rolling.run_rolling(
                score, prices_all, open_all, can_buy, can_sell, args, windows)
            rolling_equity = rolling.concat_equity(roll_results)
            rolling_table = pd.DataFrame([
                {"start": r[0].date(), "end": r[1].date(), **r[3]} for r in roll_results
            ])
            rolling_equity.rename("equity").to_csv(f"{args.out_prefix}rolling_curve.csv")
            rolling_table.to_csv(f"{args.out_prefix}rolling_metrics.csv", index=False)
            print(f"\n【滚动回测】{len(roll_results)} 段，每段 {args.roll_window} 交易日")
            for r in roll_results:
                m = r[3]
                print(f"  {r[0].date()} ~ {r[1].date()}: 收益 {m['total_return']:.2%} | 夏普 {m['sharpe']:.2f} | 回撤 {m['max_drawdown']:.2%}")
        else:
            print("⚠️ 样本区间不足以做滚动回测（请拉长 --start 或缩短 --roll-window）")

    # ---- 基准对比（买入持有指数）----
    benchmark_curve = None
    bench_dd = None
    idx_full = None
    if args.benchmark:
        try:
            # 用扩展历史（含预热期）加载，便于风险归因准确判定早期市场状态
            idx_full = load_index(args.benchmark, fetch_start, args.end)
            idx_close = idx_full[idx_full.index >= start_ts]
            # 基准净值：指数收盘价 / 首日收盘价，对齐到策略交易日，前后填充缺口
            idx_equity = (idx_close / idx_close.iloc[0]).reindex(equity.index).ffill().bfill()
            benchmark_curve = idx_equity
            bench_dd = idx_equity / idx_equity.cummax() - 1
            # 超额指标
            excess = equity / idx_equity
            excess_daily = excess.pct_change().fillna(0.0)
            metrics["excess_return"] = float(excess.iloc[-1] - 1)
            metrics["information_ratio"] = (
                float(excess_daily.mean() / excess_daily.std() * (252 ** 0.5))
                if excess_daily.std() else 0.0
            )
            pd.DataFrame({"date": equity.index, "benchmark": idx_equity.values}).to_csv(
                f"{args.out_prefix}benchmark_curve.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 基准数据获取失败，报告不含基准对比: {exc}")

    # ---- 风险归因：按市场状态拆分收益（需可用基准）----
    attribution_table = None
    if args.attribution:
        if not args.benchmark or idx_full is None or benchmark_curve is None:
            print("⚠️ 风险归因需要可用的基准指数（--benchmark），已跳过")
        else:
            regime_full = attribution.market_regime(
                idx_full, window=args.regime_window, band=args.regime_band)
            # 对齐到回测区间，缺失（预热期外）填为震荡
            regime_eval = (regime_full[regime_full.index >= start_ts]
                           .reindex(equity.index).ffill().fillna("side"))
            attribution_table = attribution.regime_attribution(
                equity, detail, benchmark_curve, regime_eval, periods_per_year=252)
            attribution_table.to_csv(f"{args.out_prefix}attribution.csv", index=False)
            print(f"\n【风险归因】市场状态由基准 {args.benchmark} 判定"
                  f"（窗口 {args.regime_window} 日，阈值 ±{args.regime_band:.0%}）")
            for r in attribution_table.to_dict("records"):
                sr = float(r["strategy_return"]); br = float(r["benchmark_return"])
                er = float(r["excess_return"]); sh = float(r["sharpe"])
                ct = float(r["contribution_pct"]); dr = float(r["day_ratio"])
                print(f"  {r['regime_label']}: 天数 {int(r['days'])} ({dr:.0%})"
                      f" | 策略 {sr:.1%} | 基准 {br:.1%}"
                      f" | 超额 {er:.1%} | 夏普 {sh:.2f}"
                      f" | 贡献 {ct:.0f}%")

    # 决策到下单的延迟：开盘价成交要多等一根 K 线（详见 backtest.run 的注释）
    exec_shift = 1 if args.use_open else 0

    # ---- 调仓清单 ----
    exec_prices = open_prices if args.use_open else prices
    plan = build_plan(weights, exec_prices, capital=args.capital,
                      fee=args.fee, stamp_tax=args.stamp_tax,
                      can_buy=can_buy_r, can_sell=can_sell_r,
                      exec_shift=exec_shift)
    plan_stats = summarize(plan)

    # ---- 输出 ----
    print("=" * 56)
    print("回测配置")
    print("=" * 56)
    if args.factors:
        print(f"  策略        : 多因子 {args.factors} ({args.standardize})")
    else:
        print(f"  策略        : {args.strategy}")
    print(f"  股票池      : {pool_display(codes)}")
    print(f"  区间        : {args.start} ~ {args.end}（预热 {warm_days} 天）")
    print(f"  调仓频率    : 每{args.rebalance}  选前 {args.top_n} 名")
    print(f"  成交价      : {'次日开盘价' if args.use_open else '当日收盘价'}")
    filters_on = []
    if not args.no_limit_filter:
        filters_on.append("涨跌停")
    if not args.no_suspend_filter:
        filters_on.append("停牌")
    if args.min_volume > 0:
        filters_on.append(f"流动性>{args.min_volume:,.0f}股")
    print(f"  启用过滤    : {' + '.join(filters_on) if filters_on else '无'}")

    print_metrics(metrics)

    if plan_stats.get("order_count"):
        print("\n【调仓统计】")
        print(f"  调仓次数    : {plan_stats['rebalance_count']}")
        print(f"  下单笔数    : {plan_stats['order_count']}"
              f"（买 {plan_stats['buy_count']} / 卖 {plan_stats['sell_count']}）")
        if plan_stats["frozen_count"]:
            print(f"  未成交笔数  : {plan_stats['frozen_count']}（涨跌停/停牌导致）")
        print(f"  累计买入额  : {plan_stats['total_buy_amount']:,.2f}")
        print(f"  累计卖出额  : {plan_stats['total_sell_amount']:,.2f}")
        print(f"  累计费用    : 佣金 {plan_stats['total_commission']:,.2f}"
              f" + 印花税 {plan_stats['total_stamp_tax']:,.2f}")

    prefix = args.out_prefix
    equity.rename("equity").to_csv(f"{prefix}equity_curve.csv")
    pd.DataFrame(
        [{"metric": k, "label": LABELS.get(k, k), "value": v} for k, v in metrics.items()]
    ).to_csv(f"{prefix}metrics.csv", index=False)
    plan.to_csv(f"{prefix}rebalance_plan.csv", index=False)
    if args.save_daily:
        detail.to_csv(f"{prefix}daily_detail.csv")

    # ---- 每日再平衡清单（实盘化）----
    if args.today or args.holdings:
        today_plan = _build_today_plan(plan, prices, weights, args)
        out = f"{args.out_prefix}today_plan.csv"
        today_plan.to_csv(out, index=False)
        print(f"\n【今日调仓清单】日期 {args.today or '自动'} → {out}")
        if not today_plan.empty:
            for _, r in today_plan.head(30).iterrows():
                print(f"  {r['action']:8s} {r['code']} {int(r['shares']):>6} 股"
                      f" @ {r['price']:.2f} = {r['amount']:>12,.2f} 元")
            if len(today_plan) > 30:
                print(f"  ... 还有 {len(today_plan) - 30} 条")
        return

    # ---- ECharts HTML 报告 ----
    drawdown = equity / equity.cummax() - 1
    metrics_groups = []
    for title, keys in GROUPS:
        items = [(LABELS.get(k, k), fmt(k, metrics[k])) for k in keys if k in metrics]
        if items:
            metrics_groups.append({"title": title, "items": items})

    plan_cards: dict[str, str] = {}
    if plan_stats.get("order_count"):
        plan_cards["调仓次数"] = str(plan_stats["rebalance_count"])
        plan_cards["下单笔数"] = f"买{plan_stats['buy_count']}/卖{plan_stats['sell_count']}"
        if plan_stats.get("frozen_count"):
            plan_cards["未成交"] = str(plan_stats["frozen_count"])
        plan_cards["累计买入"] = f"{plan_stats['total_buy_amount']:,.0f}"
        plan_cards["累计卖出"] = f"{plan_stats['total_sell_amount']:,.0f}"
        plan_cards["费用合计"] = f"{plan_stats['total_commission'] + plan_stats['total_stamp_tax']:,.2f}"

    report_data = {
        "title": "A股低频选股策略回测报告",
        "subtitle": (f"策略 {args.factors or args.strategy} · 股票池 {len(codes)}只 · "
                     f"区间 {args.start}~{args.end} · 基准 {args.benchmark or '无'}"),
        "config": [
            ("策略", args.factors or args.strategy),
            ("股票池", pool_display(codes)),
            ("区间", f"{args.start} ~ {args.end}"),
            ("调仓频率", f"每{args.rebalance} 选前{args.top_n}"),
            ("成交价", "次日开盘价" if args.use_open else "当日收盘价"),
            ("基准", args.benchmark or "无"),
        ],
        "metrics_groups": metrics_groups,
        "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
        "equity": [round(float(v), 4) for v in equity.values],
        "benchmark": [round(float(v), 4) for v in benchmark_curve.values] if benchmark_curve is not None else None,
        "drawdown": [round(float(v), 4) for v in drawdown.values],
        "bench_dd": [round(float(v), 4) for v in bench_dd.values] if bench_dd is not None else None,
        "plan": plan_cards,
        "annual": [
            {"year": int(r["year"]), "return": round(float(r["return"]), 4),
             "sharpe": round(float(r["sharpe"]), 4), "max_drawdown": round(float(r["max_drawdown"]), 4)}
            for r in annual.to_dict("records")
        ],
        "rolling_dates": [d.strftime("%Y-%m-%d") for d in rolling_equity.index] if rolling_equity is not None else None,
        "rolling_equity": [round(float(v), 4) for v in rolling_equity.values] if rolling_equity is not None else None,
        "attribution": [
            {"label": r["regime_label"],
             "strategy_return": round(float(r["strategy_return"]), 4),
             "benchmark_return": round(float(r["benchmark_return"]), 4),
             "excess_return": round(float(r["excess_return"]), 4),
             "sharpe": round(float(r["sharpe"]), 4),
             "win_rate": round(float(r["win_rate"]), 4),
             "contribution_pct": round(float(r["contribution_pct"]), 2),
             "days": int(r["days"]),
             "day_ratio": round(float(r["day_ratio"]), 4)}
            for r in (attribution_table.to_dict("records") if attribution_table is not None else [])
        ],
    }
    build_report(report_data, f"{prefix}report.html")

    print(f"\n净值曲线已保存到 {prefix}equity_curve.csv")
    print(f"评价指标已保存到 {prefix}metrics.csv")
    print(f"分年度表现已保存到 {prefix}annual_metrics.csv")
    print(f"调仓清单已保存到 {prefix}rebalance_plan.csv（共 {len(plan)} 条）")
    if args.rolling:
        print(f"滚动回测曲线已保存到 {prefix}rolling_curve.csv")
        print(f"滚动回测绩效已保存到 {prefix}rolling_metrics.csv")
    if args.attribution and attribution_table is not None:
        print(f"风险归因已保存到 {prefix}attribution.csv")
    if args.save_daily:
        print(f"每日明细已保存到 {prefix}daily_detail.csv")
    print(f"回测报告已保存到 {prefix}report.html")


if __name__ == "__main__":
    main()
