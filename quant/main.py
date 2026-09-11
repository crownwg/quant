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
from .strategy import factor_weights, weights_from_args
from . import universe
from . import rolling
from . import attribution
from . import grid
from . import compare
from . import combine
from . import timing
from . import neutralize
from . import factor_eval
from .backtest import cost_kwargs
from .strategy import apply_exposure


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
    "total_commission", "total_stamp_tax", "total_slippage", "total_impact",
    "total_cost", "cost_drag_annual",
    "excess_return",
    "avg_exposure", "in_market_ratio",
}
INT_METRICS = {"max_drawdown_days", "trading_days", "exposure_switches"}

GROUPS = [
    ("收益", ["total_return", "annual_return", "final_equity"]),
    ("风险", ["max_drawdown", "max_drawdown_days", "annual_volatility"]),
    ("风险调整收益", ["sharpe", "sortino", "calmar"]),
    ("交易特征", ["win_rate", "profit_loss_ratio", "average_daily_turnover", "annual_turnover"]),
    ("交易成本（占本金比例）", ["total_commission", "total_stamp_tax", "total_slippage",
                            "total_impact", "total_cost", "cost_drag_annual"]),
    ("相对基准", ["excess_return", "information_ratio"]),
    ("择时仓位（启用后才有）", ["avg_exposure", "in_market_ratio", "exposure_switches"]),
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
    "total_slippage": "滑点成本",
    "total_impact": "冲击成本",
    "total_cost": "成本总额",
    "cost_drag_annual": "成本年化拖累",
    "excess_return": "超额收益",
    "information_ratio": "信息比率",
    "avg_exposure": "平均仓位",
    "in_market_ratio": "在场交易日占比（仓位>50%）",
    "exposure_switches": "仓位切换次数",
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


def _norm_index_symbol(sym: str) -> str:
    """把裸指数代码补上交易所前缀（实现见 timing.norm_index_symbol，两处共用一份规则）。"""
    return timing.norm_index_symbol(sym)


def _resolve_proxy(args, prices_all: pd.DataFrame, fetch_start: str):
    """解析择时代理序列，返回 (proxy, label)；proxy 为 None 表示改用池子等权组合。

    --timing-proxy 决定拿什么当「市场」：
      benchmark : --benchmark 指定的指数
      pool      : 所交易池子的等权组合（衡量「我持有的这类资产的自身趋势」）
      auto      : 有 --benchmark 就用基准，否则用池子等权组合
      其他任意值 : 当作指数代码，如 000932（中证消费）→ 自动补前缀

    结果缓存在 args._proxy_cache：网格/样本外要按几十组参数分别构造敞口，
    若每次都重新联网取指数，启动时间会拖到不可用。
    """
    cached = getattr(args, "_proxy_cache", None)
    if cached is not None:
        return cached

    spec = (getattr(args, "timing_proxy", "auto") or "auto").strip()
    sym = ""
    if spec in ("benchmark", "auto"):
        if args.benchmark:
            sym = args.benchmark
    elif spec != "pool":
        sym = spec

    proxy, label = None, "池子等权组合"
    if sym:
        sym = _norm_index_symbol(sym)
        try:
            proxy = load_index(sym, fetch_start, args.end)
            proxy.index = pd.to_datetime(proxy.index)
            label = f"指数 {sym}"
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 择时代理指数 {sym} 获取失败，回退到池子等权组合: {exc}")
            proxy = None
    args._proxy_cache = (proxy, label)
    return proxy, label


def resolve_exposure(args, prices_all: pd.DataFrame, fetch_start: str):
    """构造择时敞口序列（0~1），未启用择时时返回 None。

    选基准还是池子，取决于你想对冲的是「市场 beta」还是「这个行业的 beta」。
    交易消费池时，用中证消费指数（sh000932）通常比沪深300更贴切。
    """
    timing_on = (getattr(args, "timing", "off") or "off") != "off"
    vol_on = float(getattr(args, "vol_target", 0.0) or 0.0) > 0
    if not timing_on and not vol_on:
        return None

    proxy, label = _resolve_proxy(args, prices_all, fetch_start)
    return timing.build_exposure(args, proxy=proxy, prices=prices_all,
                                 proxy_label=label, verbose=True)


def _parse_timing_grid(args):
    """解析 --grid-timing / --grid-timing-lookbacks / --grid-timing-bands。

    返回 (modes, lookbacks, bands, search_timing)。search_timing=False 时网格只扫
    选股参数，各组合沿用 --timing 那一条敞口（旧行为，保持兼容）。
    模式列表里自动补上 off：不带上「不择时」这个对照组，搜索就永远得不出
    「不如不择时」的结论。
    """
    raw = str(getattr(args, "grid_timing", "") or "")
    modes = [m.strip().lower() for m in raw.split(",") if m.strip()]
    bad = [m for m in modes if m not in timing.TIMING_MODES]
    if bad:
        raise SystemExit(f"--grid-timing 含未知模式 {bad}，可选 {list(timing.TIMING_MODES)}")
    if not modes:
        return [], [], [], False
    lbs = grid.parse_ints(str(getattr(args, "grid_timing_lookbacks", "60,120") or "60,120"))
    bands = grid.parse_floats(str(getattr(args, "grid_timing_bands", "0") or "0"))
    if "off" not in modes:
        modes.insert(0, "off")
    return modes, lbs, bands, True


def _grid_timing_proxy(args, prices_all, fetch_start, search_timing: bool):
    """网格/样本外扫择时时需要一条「整段」代理序列；不扫则返回 None（连指数都不去取）。"""
    if not search_timing:
        return None
    proxy, label = _resolve_proxy(args, prices_all, fetch_start)
    print(f"  择时代理    : {label}")
    return proxy


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
    skipped: list[str] = []
    codes = set(target_w.index)
    if holdings_df is not None:
        codes |= set(holdings_df["code"])
    for code in sorted(codes):
        w = float(target_w.get(code, 0.0))
        target_amount = w * total_capital
        price = float(exec_price.get(code, np.nan))
        if pd.isna(price) or price <= 0:
            # 最常见的原因：持仓里有当前股票池之外的股票，价格表里没有它。
            # 必须显式告知，否则用户会把「不在清单里」误读成「这只不用调整」。
            if holdings_df is not None:
                m = holdings_df[holdings_df["code"] == code]
                if not m.empty and int(m["shares"].iloc[0]) > 0:
                    skipped.append(code)
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
    if skipped:
        print(f"\n  ⚠️ 以下持仓因取不到价格而无法计算，需单独处理："
              f"{'、'.join(skipped)}")
        print("     （它们不在当前股票池的价格表内，不代表不需要调整）")
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
                        "concept:白酒 概念板块、all 全市场(含退市)；可逗号合并，如 hs300,消费")
    p.add_argument("--as-of", default="",
                   help="股票池时点日期(YYYYMMDD)：按该日期的成分股建池，消除幸存者偏差。"
                        "默认取 --start；需要本地已积累该日期的成分股快照")
    p.add_argument("--min-listed-days", type=int, default=0,
                   help="剔除上市未满 N 个自然日的次新股（0=关闭）。建议 250（约一年），"
                        "避开新股连续一字板与无涨跌停限制期")
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

    # 权重方案与集中度
    p.add_argument("--weighting", default="equal",
                   choices=["equal", "inv_vol", "score", "rank"],
                   help="权重分配：equal 等权 / inv_vol 波动率倒数(近似风险平价) / "
                        "score 按分数幅度加权 / rank 按排名线性加权")
    p.add_argument("--max-weight", type=float, default=0.0,
                   help="单票权重上限，如 0.2；0=不限。超出部分按比例分配给未触顶标的，"
                        "全部触顶则留现金（不强行满仓）")
    p.add_argument("--neutralize", default="",
                   help="因子中性化，可组合：industry（行业内去均值）/ size（市值分组去均值），"
                        "如 --neutralize industry,size。用于剥离行业/小市值 beta，"
                        "避免把 beta 收益误当成因子 alpha")

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
    p.add_argument("--min-amount", type=float, default=0.0,
                   help="流动性门槛（金额口径，元）：近 20 日均成交额低于该值则不可交易。"
                        "比 --min-volume 更合理——1000 万股对 3 元股是 3000 万、对 300 元股是 30 亿")
    p.add_argument("--max-participation", type=float, default=0.0,
                   help="容量约束：单票权重上限 = 参与率 × 20日均成交额 / 本金。"
                        "0=不限；建议 0.05~0.10。一次性吃掉某票当日 30%% 成交额时，"
                        "回测里假设的成交价根本拿不到")
    p.add_argument("--no-limit-filter", action="store_true", help="关闭涨跌停过滤")
    p.add_argument("--no-suspend-filter", action="store_true", help="关闭停牌过滤")
    p.add_argument("--st-codes", default="", help="ST 股代码，逗号分隔（涨跌停幅度按 5%% 计）")

    # 成本与本金
    p.add_argument("--capital", type=float, default=1_000_000.0, help="组合本金（调仓清单用）")
    p.add_argument("--fee", type=float, default=0.0003, help="佣金费率（双边）")
    p.add_argument("--stamp-tax", type=float, default=0.0005, help="印花税率（仅卖出）")
    p.add_argument("--slippage", type=float, default=0.0005,
                   help="滑点率（单边，按成交额），默认 5bp。月频换手 10 只票时滑点通常"
                        "吃掉 1~3%%/年，设为 0 会系统性高估收益")
    p.add_argument("--min-commission", type=float, default=5.0,
                   help="单笔最低佣金（元），默认 5。低本金时这项比费率本身更重要")
    p.add_argument("--impact-coef", type=float, default=0.0,
                   help="冲击成本系数（平方根模型）：impact = coef × sqrt(成交额/日均成交额)。"
                        "0=关闭；0.1 意味着吃掉 100%% ADV 时额外 10%% 冲击")

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

    # 择时与仓位管理
    p.add_argument("--timing", default="off", choices=list(timing.TIMING_MODES),
                   help="趋势择时：off 关闭 / ma 价格对均线 / momentum 过去N日收益为正 / "
                        "dual 均线+斜率双确认。输出 0~1 的仓位系数，剩余部分留现金")
    p.add_argument("--timing-lookback", type=int, default=120,
                   help="趋势择时窗口（交易日），默认 120")
    p.add_argument("--timing-band", type=float, default=0.0,
                   help="滞回带：空仓转满仓需高于均线×(1+band)、满仓转空仓需低于×(1-band)，"
                        "中间维持原状态。0.02~0.05 能显著减少均线附近的来回打脸")
    p.add_argument("--timing-min-exposure", type=float, default=0.0,
                   help="出场时的最低仓位，默认 0（完全空仓）。设 0.3 表示看空也只减到三成")
    p.add_argument("--timing-max-exposure", type=float, default=1.0,
                   help="最高仓位，默认 1.0（不加杠杆）")
    p.add_argument("--timing-ma-slope", type=int, default=0,
                   help="dual 模式判断「均线向上」的回看天数，0=自动取 timing-lookback/10")
    p.add_argument("--timing-smooth", type=int, default=0,
                   help="对仓位序列做 N 日移动平均，降低仓位自身的换手（会引入轻微滞后）")
    p.add_argument("--timing-update", default="rebalance", choices=["rebalance", "daily"],
                   help="仓位更新频率：rebalance（默认）只在调仓日调整仓位、与选股同步；"
                        "daily 逐日调整。逐日会让月频策略退化成日频对倒"
                        "（实测 17000+ 笔订单把收益吃光），仅用于观察代价")
    p.add_argument("--timing-proxy", default="auto",
                   help="择时代理：auto（有基准用基准，否则用池子等权组合）/ benchmark / "
                        "pool（池子等权组合，衡量所持资产的自身趋势）/ 或直接给指数代码如 000932")
    p.add_argument("--vol-target", type=float, default=0.0,
                   help="波动率目标：仓位 = 目标年化波动率 / 已实现波动率。0=关闭。"
                        "如 0.15 表示把组合波动压到 15%% 附近")
    p.add_argument("--vol-target-lookback", type=int, default=60,
                   help="已实现波动率的滚动窗口（交易日），默认 60")
    p.add_argument("--vol-floor", type=float, default=0.2,
                   help="波动率目标的仓位下限，默认 0.2")
    p.add_argument("--vol-cap", type=float, default=1.0,
                   help="波动率目标的仓位上限，默认 1.0（不加杠杆）")

    # 参数优化（网格搜索）
    p.add_argument("--grid", action="store_true",
                   help="参数优化：在 (lookback × top_n × buffer) 网格上回测，找稳健最优参数")
    p.add_argument("--grid-lookbacks", default="60,90,120,160",
                   help="网格 lookback 候选，逗号分隔，默认 60,90,120,160")
    p.add_argument("--grid-topn", default="10,15,20",
                   help="网格 top_n 候选，逗号分隔，默认 10,15,20")
    p.add_argument("--grid-buffers", default="0,2",
                   help="网格 buffer 候选，逗号分隔，默认 0,2")
    p.add_argument("--grid-timing", default="",
                   help="把择时模式也纳入网格/样本外搜索，逗号分隔，如 off,ma,momentum。"
                        "留空=不扫择时（沿用 --timing 那一条）。"
                        "会自动补上 off（不择时），否则无法回答「择时到底该不该用」")
    p.add_argument("--grid-timing-lookbacks", default="60,120",
                   help="择时窗口候选，逗号分隔，默认 60,120（仅对 --grid-timing 里非 off 的模式生效）")
    p.add_argument("--grid-timing-bands", default="0,0.02",
                   help="择时滞回带候选，逗号分隔，默认 0,0.02")

    # 因子有效性检验
    p.add_argument("--factor-eval", action="store_true",
                   help="因子有效性检验：输出 IC / IR / 正IC占比 / 因子衰减 / 分组收益。"
                        "直接回答「分数靠前的股票是否真的跑赢」——净值曲线回答不了这个问题")

    # 多策略对比
    p.add_argument("--compare", action="store_true",
                   help="多策略对比：同一池子横向跑多个预定义策略，挑稳健组合")
    p.add_argument("--compare-strategies", default="",
                   help="对比策略列表，逗号分隔，如 momentum_120,reversal_20；为空用默认 7 个")

    # 样本外验证（walk-forward）
    p.add_argument("--walk-forward", action="store_true",
                   help="样本外验证：滚动「训练窗选参 → 紧邻测试窗纯验证」，"
                        "并与「全样本最优参数」「默认参数」对比，识别参数过拟合")
    p.add_argument("--fw-train", type=float, default=2.0, help="walk-forward 训练窗长度（年），默认 2")
    p.add_argument("--fw-test", type=float, default=0.5, help="walk-forward 测试窗长度（年），默认 0.5")
    p.add_argument("--wf-ensemble-k", default="3,5,10,0",
                   help="参数集成的集成度 K（逗号分隔），只在训练期平滑分前 K 组内等权平均；"
                        "0 表示全部候选，1 退化为「按平滑分选参」。默认 3,5,10,0 —— "
                        "多个 K 并列给出，是为了不把「K 取多少」变成新一轮事后择优")
    p.add_argument("--fw-metric", default="sharpe", choices=["sharpe", "calmar", "total_return"],
                   help="训练窗内选参依据，默认 sharpe")

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
    as_of = args.as_of.strip() or args.start
    pool_codes: list[str] = []
    pool_report: list[dict] = []
    if args.pool.strip():
        pool_codes, pool_report = universe.build_universe_report(args.pool, as_of=as_of)
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

    # 幸存者偏差告警：任何子池没能做到时点正确，都必须显式提示，绝不静默放行
    biased = [r for r in pool_report if not r["pit"]]
    if biased:
        print("\n" + "!" * 68)
        print("⚠️  幸存者偏差告警：以下池子未能按历史时点取成分股")
        for r in biased:
            print(f"  · {r['pool']}（{r['n_codes']} 只）：{r['message']}")
        print("!" * 68 + "\n", flush=True)

    # 预热期：因子需要历史数据才能计算，往前多取一段，否则开头几个月空仓。
    # 择时的滚动窗口（均线 / 已实现波动率）同样需要预热，否则回测开头会被迫空仓。
    factor_warm = max(args.lookback, args.ma_long, args.vol_lookback, args.vol_long)
    # 网格/样本外若同时扫择时窗口，预热期必须覆盖候选里最长的那个窗口，
    # 否则那些「长窗口」组合在开头会因为均线历史不足而被迫空仓，被误判成更差。
    grid_tlbs: list[int] = []
    if str(getattr(args, "grid_timing", "") or "").strip():
        try:
            grid_tlbs = grid.parse_ints(str(getattr(args, "grid_timing_lookbacks", "") or "0"))
        except ValueError:
            grid_tlbs = []
    timing_warm = max([int(getattr(args, "timing_lookback", 0) or 0),
                       int(getattr(args, "vol_target_lookback", 0) or 0),
                       int(getattr(args, "timing_smooth", 0) or 0)] + grid_tlbs)
    warm_days = max(factor_warm, timing_warm) * 2 + 30
    # 次新股过滤按「上市自然日」判断，面板必须比回测起点再往前 min_listed_days，
    # 否则老股票的可用历史会被误判成"刚上市"而被整体剔除。
    if args.min_listed_days > 0:
        warm_days = max(warm_days, args.min_listed_days + 30)
    fetch_start = (pd.Timestamp(args.start) - pd.Timedelta(days=warm_days)).strftime("%Y%m%d")

    # 大池首次下载较慢：池子大时降低单只间隔并允许跳过失败标的
    sleep = 0.3 if len(codes) > 20 else 1.0
    on_error = "skip" if (pool_codes or len(codes) > 20) else "raise"
    prices_all = pd.DataFrame(); open_all = pd.DataFrame(); volume_all = pd.DataFrame()
    high_all = pd.DataFrame(); low_all = pd.DataFrame()
    amount_all = pd.DataFrame(); share_all = pd.DataFrame()
    if codes:  # --combine 模式下 codes 为空，跳过面板加载（池子各自在 combine 里加载）
        panel = load_panel(codes, fetch_start, args.end, sleep=sleep, on_error=on_error)
        prices_all = panel["close"]
        open_all = panel["open"]
        volume_all = panel["volume"]
        high_all = panel.get("high", pd.DataFrame())
        low_all = panel.get("low", pd.DataFrame())
        amount_all = panel.get("amount", pd.DataFrame())
        share_all = panel.get("outstanding_share", pd.DataFrame())
        if len(prices_all.columns) < len(codes):
            skipped = len(codes) - len(prices_all.columns)
            print(f"⚠️ {skipped} 只代码数据不可用已跳过，实际参与回测 {len(prices_all.columns)} 只")
        if prices_all.empty:
            raise SystemExit("没有任何可用数据，回测终止")

    # ---- 日均成交额 ADV：容量约束与冲击成本的共同分母 ----
    adv_all = filters.adv_notional(amount=amount_all if not amount_all.empty else None,
                                   volume=volume_all, close=prices_all)
    if adv_all is None or adv_all.empty:
        adv_all = None
        has_adv = False
    else:
        has_adv = bool(adv_all.notna().any().any())
    # 挂到 args 上：backtest.cost_kwargs 会取它做冲击成本，各入口口径一致
    args.adv_panel = adv_all

    score = build_score(args, prices_all, volume_all)

    # ---- 因子中性化：剥离行业 / 市值 beta ----
    if args.neutralize.strip():
        mktcap_all = neutralize.mktcap_panel(
            prices_all, amount=amount_all if not amount_all.empty else None,
            volume=volume_all, outstanding_share=share_all if not share_all.empty else None)
        score = neutralize.apply_neutralize(score, args.neutralize,
                                            mktcap=mktcap_all, verbose=True)

    # ---- 次新股过滤：上市未满 min_listed_days 的标的不参与选股 ----
    if args.min_listed_days > 0:
        age_ok = filters.listing_age_mask(prices_all, args.min_listed_days)
        blocked = int((~age_ok).sum().sum())
        score = score.where(age_ok)
        print(f"  次新股过滤  : 剔除上市不足 {args.min_listed_days} 自然日的标的"
              f"（累计屏蔽 {blocked} 个 股票×交易日）")

    # ---- 交易可行性约束 ----
    st_codes = [c.strip() for c in args.st_codes.split(",") if c.strip()]
    limit_pct = filters.limit_pct_by_code(codes, st_codes)
    illiquid = filters.illiquid_mask(volume_all, args.min_volume)
    # 金额口径的流动性门槛与股数口径取并集（两条都需要满足）
    illiquid_amt = filters.illiquid_mask_by_amount(
        amount_all if not amount_all.empty else None, args.min_amount)
    if illiquid_amt is not None:
        illiquid = illiquid_amt if illiquid is None else (illiquid | illiquid_amt)

    # ---- 容量约束：单票权重上限 = 参与率 × ADV / 本金 ----
    weight_cap_all = filters.capacity_cap(adv_all, args.capital, args.max_participation)
    if weight_cap_all is not None:
        finite = weight_cap_all.notna()
        if finite.any().any():
            median_cap = float(weight_cap_all.stack().median())
            n_tight = int((weight_cap_all.min(axis=0) < 1.0 / max(args.top_n, 1)).sum())
            if median_cap >= 1.0:
                print(f"  容量约束    : 参与率 {args.max_participation:.0%} · 本金 "
                      f"{args.capital:,.0f} 元 → 容量充裕，不构成约束"
                      f"（单票上限中位数 {median_cap:.0%} 已超过满仓）")
            else:
                print(f"  容量约束    : 参与率 {args.max_participation:.0%} · 本金 "
                      f"{args.capital:,.0f} 元 → 单票上限中位数 {median_cap:.2%}"
                      f"（{n_tight} 只标的的容量低于等权仓位 {1.0/max(args.top_n,1):.2%}）")

    limit_stats: dict = {}
    can_buy, can_sell = filters.tradability(
        prices_all, volume_all, limit_pct=limit_pct, illiquid=illiquid,
        enable_limit=not args.no_limit_filter,
        enable_suspend=not args.no_suspend_filter,
        open_=open_all if args.use_open else None,
        high=high_all if not high_all.empty else None,
        low=low_all if not low_all.empty else None,
        exec_at_open=args.use_open,
        stats=limit_stats,
    )
    if not args.no_limit_filter and limit_stats:
        print(f"  涨跌停判定  : 判定价={limit_stats.get('held_price', 'close')}"
              f" | 涨停 一字{limit_stats.get('limit_up_oneword', 0)}/封板{limit_stats.get('limit_up_close', 0)}"
              f" | 跌停 一字{limit_stats.get('limit_down_oneword', 0)}/封板{limit_stats.get('limit_down_close', 0)}"
              f" | 涨跌停致禁买{limit_stats.get('limit_blocked_buy', 0)}"
              f"/禁卖{limit_stats.get('limit_blocked_sell', 0)}"
              f"（单位：股票×交易日）")
        parts = []
        if "suspend_blocked" in limit_stats:
            parts.append(f"停牌 {limit_stats['suspend_blocked']}")
        if "illiquid_blocked" in limit_stats:
            parts.append(f"流动性 {limit_stats['illiquid_blocked']}")
        if parts:
            print(f"  其他不可交易: {' + '.join(parts)}"
                  f" | 合计禁买{limit_stats.get('blocked_buy', 0)}/禁卖{limit_stats.get('blocked_sell', 0)}")

    # ---- 择时仓位：决定「什么时候在场、在场放多少」----
    # 必须放在参数扫描分支之前：网格 / 对比 / 滚动都要用同一份敞口，
    # 否则会出现「主回测算择时、网格搜索没算」的口径分裂。
    exposure = None
    if codes:
        exposure = resolve_exposure(args, prices_all, fetch_start)

    # ---- 因子有效性检验：直接回答「因子有没有用」，与回测相互独立----
    if args.factor_eval:
        print("\n【因子有效性检验】横截面 Rank IC（未来 20 日收益）")
        # 待检验因子集合：--factors 指定则只验这些；否则验当前策略的因子
        # 外加两个对照因子，便于横向比较「当前因子是否真的有信息」
        to_eval: dict[str, "pd.DataFrame"] = {}
        if args.factors:
            for name, w in parse_factor_spec(args.factors).items():
                to_eval[name] = FACTOR_BUILDERS[name](args, prices_all, volume_all)
        else:
            to_eval[args.strategy] = FACTOR_BUILDERS[args.strategy](args, prices_all, volume_all)
            for name in ("momentum", "reversal", "low_volatility"):
                if name not in to_eval:
                    to_eval[name] = FACTOR_BUILDERS[name](args, prices_all, volume_all)
        if args.neutralize.strip():
            mktcap_all = neutralize.mktcap_panel(
                prices_all, amount=amount_all if not amount_all.empty else None,
                volume=volume_all,
                outstanding_share=share_all if not share_all.empty else None)
            for name in list(to_eval):
                to_eval[name] = neutralize.apply_neutralize(
                    to_eval[name], args.neutralize, mktcap=mktcap_all, verbose=False)

        results = factor_eval.evaluate_many(to_eval, prices_all)
        factor_eval.print_summary(results)
        out_html = f"{args.out_prefix}factor_eval.html"
        factor_eval.build_ic_report(
            results, out_html, title="因子有效性检验",
            subtitle=(f"股票池 {len(prices_all.columns)}只 · 区间 {args.start}~{args.end} · "
                      f"中性化 {args.neutralize or '无'}"))
        print(f"\n因子检验报告已保存: {out_html}")
        return

    # ---- 参数优化：网格搜索（在加载面板后、单次回测前分叉）----
    if args.grid:
        lookbacks = grid.parse_ints(args.grid_lookbacks)
        top_ns = grid.parse_ints(args.grid_topn)
        buffers = grid.parse_ints(args.grid_buffers)
        t_modes, t_lbs, t_bands, search_timing = _parse_timing_grid(args)
        proxy = _grid_timing_proxy(args, prices_all, fetch_start, search_timing)
        bench = None
        if args.benchmark:
            try:
                bench = load_index(args.benchmark, args.start, args.end)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ 基准获取失败，网格不计算超额: {exc}")
        nbase = len(lookbacks) * len(top_ns) * len(buffers)
        ncomb = len(grid.build_combos(lookbacks, top_ns, buffers, t_modes, t_lbs, t_bands))
        tdesc = (f" × 择时 {','.join(t_modes)}×窗口{len(t_lbs)}×带{len(t_bands)}"
                 if search_timing else "")
        print(f"\n【参数优化】网格 {len(lookbacks)}×{len(top_ns)}×{len(buffers)} = {nbase} 组"
              f"{tdesc} → 合计 {ncomb} 组，池子 {len(codes)} 只")
        start_ts = pd.Timestamp(args.start)
        res = grid.grid_search(prices_all, open_all, can_buy, can_sell, args,
                               lookbacks, top_ns, buffers, start_ts, benchmark_curve=bench,
                               weight_cap=weight_cap_all, exposure=exposure,
                               timing_modes=t_modes, timing_lookbacks=t_lbs,
                               timing_bands=t_bands, proxy=proxy)
        res.to_csv(f"{args.out_prefix}grid_results.csv", index=False)
        sub = (f"池子 {len(codes)}只 · 区间 {args.start}~{args.end} · "
               f"网格 {len(lookbacks)}×{len(top_ns)}×{len(buffers)}"
               + (f" × 择时({','.join(t_modes)})" if search_timing else "")
               + f" = {ncomb} 组 · 基准 {args.benchmark or '无'}")
        grid.build_grid_report(res, f"{args.out_prefix}grid_report.html",
                               title="参数优化网格搜索", subtitle=sub)
        print("\n【Top 按夏普】")
        for _, r in res.sort_values("sharpe", ascending=False).head(10).iterrows():
            er = f" 超额 {r['excess_return']:.1%} IR {r['information_ratio']:.2f}" if "excess_return" in r else ""
            lbl = grid.combo_label({"lookback": int(r["lookback"]), "top_n": int(r["top_n"]),
                                    "buffer": int(r["buffer"]),
                                    "timing": r.get("timing", "off"),
                                    "timing_lookback": r.get("timing_lookback", 0),
                                    "timing_band": r.get("timing_band", 0.0)})
            print(f"  {lbl:<34s} | 收益 {r['total_return']:.1%} 夏普 {r['sharpe']:.2f}"
                  f" 回撤 {r['max_drawdown']:.1%} 最差年 {r['worst_year_return']:.1%}"
                  f" 正年 {r['positive_year_ratio']:.0%}{er}")
        if search_timing and "timing" in res.columns:
            print("\n【按择时模式分组（组内平均）】")
            for tm, sres in res.groupby("timing", sort=False):
                print(f"  {str(tm):<9s} n={len(sres):<3d} 平均收益 {sres['total_return'].mean():>7.1%}"
                      f" 平均夏普 {sres['sharpe'].mean():>5.2f}"
                      f" 平均回撤 {sres['max_drawdown'].mean():>7.1%}")
            print("  ——若 off 组的平均夏普最高，说明这一轮里择时参数整体在减分")
        print(f"\n网格结果已保存: {args.out_prefix}grid_results.csv / {args.out_prefix}grid_report.html")
        return

    # ---- 样本外验证：walk-forward（识别参数过拟合）----
    if args.walk_forward:
        lookbacks = grid.parse_ints(args.grid_lookbacks)
        top_ns = grid.parse_ints(args.grid_topn)
        buffers = grid.parse_ints(args.grid_buffers)
        t_modes, t_lbs, t_bands, search_timing = _parse_timing_grid(args)
        proxy = _grid_timing_proxy(args, prices_all, fetch_start, search_timing)
        ens_ks = grid.parse_ensemble_ks(args.wf_ensemble_k)
        start_ts = pd.Timestamp(args.start)
        ncomb = len(grid.build_combos(lookbacks, top_ns, buffers, t_modes, t_lbs, t_bands))
        print(f"\n【样本外验证】walk-forward · 训练 {args.fw_train} 年 / 测试 {args.fw_test} 年"
              f" · 网格 {len(lookbacks)}×{len(top_ns)}×{len(buffers)}"
              + (f" × 择时 {','.join(t_modes)}" if search_timing else "")
              + f" = {ncomb} 组"
              + f" · 集成 K={','.join('all' if k is None else str(k) for k in ens_ks)}")
        res = grid.walk_forward_search(
            prices_all, open_all, can_buy, can_sell, args,
            lookbacks, top_ns, buffers, start_ts,
            train_years=args.fw_train, test_years=args.fw_test,
            weight_cap=weight_cap_all, select_metric=args.fw_metric,
            exposure=exposure,
            timing_modes=t_modes, timing_lookbacks=t_lbs,
            timing_bands=t_bands, proxy=proxy,
            ensemble_ks=ens_ks)
        res["folds"].to_csv(f"{args.out_prefix}wf_folds.csv", index=False)
        res["summary"].to_csv(f"{args.out_prefix}wf_summary.csv", index=False)
        grid.build_wf_report(
            res, f"{args.out_prefix}wf_report.html",
            title="样本外验证（Walk-Forward）",
            subtitle=(f"池子 {len(codes)}只 · 区间 {args.start}~{args.end} · "
                      f"训练 {res['train_n']}日/测试 {res['test_n']}日 · "
                      f"搜索空间 {res['n_combos']} 组 · "
                      f"全样本最优 {res['full_best_label']}"))
        print("\n【样本外表现对比】")
        sm_ = res["summary"].set_index("key")
        base_ret = float(sm_.loc["walk_forward", "total_return"])
        for _, r in res["summary"].iterrows():
            is_adaptive = str(r.get("key")) == "smooth" or str(r.get("key")).startswith("ens")
            delta = (f"  vs argmax {r['total_return'] - base_ret:>+7.1%}"
                     if is_adaptive else " " * 18)
            print(f"  {r['strategy']:<28s} 总收益 {r['total_return']:>7.1%}"
                  f" 年化 {r['annual_return']:>6.1%} 平均夏普 {r['avg_sharpe']:>5.2f}"
                  f" 最差折回撤 {r['worst_fold_drawdown']:>7.1%}"
                  f" 正收益折 {r['positive_folds']:.0%}{delta}")
        gap = float(sm_.loc["full_sample_best", "total_return"]) - base_ret
        print(f"\n  事后选参 vs 自适应选参（argmax）差距：{gap:+.1%}"
              + ("（事后选参高估，说明参数在拟合噪声）" if gap > 0.02 else "（差距不大）"))
        # 不做单点择优的臂里最好的那条：平滑 + 各集成度一起比
        no_sel_keys = [k for k in sm_.index if k == "smooth" or str(k).startswith("ens")]
        best_nosel = max(float(sm_.loc[k, "total_return"]) for k in no_sel_keys)
        d_nosel = best_nosel - base_ret
        print(f"  不踩尖峰（邻域平滑 / 参数集成）最好的一条 vs argmax：{d_nosel:+.1%}"
              + ("（正则说明 argmax 挑到的主要是噪声）" if d_nosel > 0.005 else "（选参方式差别不大）"))
        if res.get("k_note"):
            print(f"  集成度 K：{res['k_note']}")
            print("  ——K 的收益随 K 单调下降/上升才是信息（说明参数曲面有真实的上下结构）；"
                  "忽高忽低说明排序本身就是噪声。无论哪种，都不要事后挑最好看的 K。")
        if res.get("ens_note"):
            print(f"  {res['ens_note']}")
            print("  ——若最窄集成臂每折持有的模式高度一致（都是 ma 或都是 off），"
                  "那是结构结论；若每折都不一样，说明宽集成只是被动分散。")
        if res.get("timing_note"):
            print(f"  {res['timing_note']}")
            print("  ——频繁选中 off 说明择时参数没有稳定信息；频繁选中某个模式"
                  "才是「均线确实能识别下跌」的证据")
        print(f"\n样本外验证结果已保存: {args.out_prefix}wf_folds.csv"
              f" / {args.out_prefix}wf_summary.csv / {args.out_prefix}wf_report.html")
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
            strategies, args, benchmark_curve=bench, weight_cap=weight_cap_all,
            exposure=exposure,
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

    weights_all = weights_from_args(score, args, can_buy=can_buy, can_sell=can_sell,
                                    prices=prices_all, weight_cap=weight_cap_all,
                                    exposure=exposure)

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
                                  **cost_kwargs(args))

    # ---- 择时仓位统计（并入指标，并作为报告的一条曲线）----
    exposure_eval = None
    if exposure is not None:
        exposure_eval = exposure.reindex(equity.index).ffill().fillna(1.0).clip(lower=0.0)
        es = timing.exposure_summary(exposure_eval)
        metrics["avg_exposure"] = es["avg_exposure"]
        metrics["in_market_ratio"] = es["in_market_ratio"]
        metrics["exposure_switches"] = float(es["switches"])
        detail["exposure"] = exposure_eval

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
                score, prices_all, open_all, can_buy, can_sell, args, windows,
                weight_cap=weight_cap_all, exposure=exposure)
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
    weight_desc = {"equal": "等权", "inv_vol": "波动率倒数(风险平价)",
                   "score": "分数加权", "rank": "排名加权"}[args.weighting]
    print(f"  权重方案    : {weight_desc}"
          + (f" · 单票上限 {args.max_weight:.0%}" if args.max_weight > 0 else ""))
    if exposure is not None:
        tm_parts = []
        if args.timing != "off":
            tm_parts.append(f"趋势 {args.timing}({args.timing_lookback}日"
                            + (f",带{args.timing_band:.1%}" if args.timing_band else "") + ")")
        if args.vol_target > 0:
            tm_parts.append(f"波动率目标 {args.vol_target:.0%}(下限 {args.vol_floor:.0%})")
        upd = "仅调仓日更新" if args.timing_update == "rebalance" else "逐日更新"
        print(f"  择时仓位    : {' × '.join(tm_parts)}"
              + (f" · 最低仓位 {args.timing_min_exposure:.0%}" if args.timing_min_exposure > 0 else "")
              + f" · 代理 {args.timing_proxy} · {upd}")
    if args.neutralize.strip():
        print(f"  因子中性化  : {args.neutralize}")
    cost_desc = (f"佣金 {args.fee:.4%}(最低 {args.min_commission:g}元/笔)"
                 f" + 印花税 {args.stamp_tax:.4%} + 滑点 {args.slippage:.4%}")
    if args.impact_coef > 0:
        cost_desc += f" + 冲击(coef={args.impact_coef:g})"
    print(f"  成本模型    : {cost_desc}")

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
            # 列名是 delta_shares（本次变动股数）；旧代码写的 r['shares'] 不存在，
            # 一执行就 KeyError → 进程以退出码 1 结束，调用方（如 Web）会当成失败。
            has_qty = "delta_shares" in today_plan.columns
            for _, r in today_plan.head(30).iterrows():
                qty = abs(int(r["delta_shares"])) if has_qty else 0
                print(f"  {r['action']:8s} {r['code']} {qty:>6} 股"
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

    timing_desc = ""
    if exposure is not None:
        tp = []
        if args.timing != "off":
            tp.append(f"趋势{args.timing}({args.timing_lookback}日)")
        if args.vol_target > 0:
            tp.append(f"波动率目标{args.vol_target:.0%}")
        timing_desc = " + ".join(tp)

    report_data = {
        "title": "A股低频选股策略回测报告",
        "subtitle": (f"策略 {args.factors or args.strategy} · 股票池 {len(codes)}只 · "
                     f"区间 {args.start}~{args.end} · 基准 {args.benchmark or '无'}"
                     + (f" · 择时 {timing_desc}" if timing_desc else "")),
        "config": [
            ("策略", args.factors or args.strategy),
            ("股票池", pool_display(codes)),
            ("区间", f"{args.start} ~ {args.end}"),
            ("调仓频率", f"每{args.rebalance} 选前{args.top_n}"),
            ("成交价", "次日开盘价" if args.use_open else "当日收盘价"),
            ("基准", args.benchmark or "无"),
        ] + ([("择时仓位", timing_desc)] if timing_desc else []),
        "metrics_groups": metrics_groups,
        "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
        "equity": [round(float(v), 4) for v in equity.values],
        "benchmark": [round(float(v), 4) for v in benchmark_curve.values] if benchmark_curve is not None else None,
        "drawdown": [round(float(v), 4) for v in drawdown.values],
        "bench_dd": [round(float(v), 4) for v in bench_dd.values] if bench_dd is not None else None,
        "exposure": ([round(float(v), 4) for v in exposure_eval.values]
                     if exposure_eval is not None else None),
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
