"""多池组合配置：跨多个池子各自跑策略，整合成一个组合。

设计目标
--------
用户的实际投资往往跨多个板块（消费 + 蓝筹 + 医药 + 工程机械），需要：
  1. 每个池子用稳健策略跑回测 → 得到各池的净值曲线
  2. 计算池子间相关性矩阵
  3. 池子间配权（默认等权；可指定固定权重；可按近期动量打分动态调权）
  4. 整合成组合净值 = sum(weight_i × pool_equity_i)

输入格式：--combine "消费=reversal_20,蓝筹=low_vol_60,医药=ma_trend"
权重可：--combine-weights "0.5,0.3,0.2"（与 combine 顺序对齐，缺省则等权）

不在这里做的事
--------------
- 池子内的选股与权重交给 strategy.py；本模块只做"池子间"配权与合并。
- 池子间的相关性按日收益计算（更稳，比直接看净值更准）。
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np

from . import factors, filters, universe
from .backtest import run as backtest_run
from .data import load_panel
from .strategy import factor_weights


# 策略名 → factor builder（与 compare.py 共用一份）
def _b_momentum_120(a, p, v):  return factors.momentum(p, lookback=120)
def _b_momentum_60(a, p, v):   return factors.momentum(p, lookback=60)
def _b_reversal_20(a, p, v):   return factors.reversal(p, lookback=20)
def _b_reversal_5(a, p, v):    return factors.reversal(p, lookback=5)
def _b_low_vol_60(a, p, v):    return factors.low_volatility(p, lookback=60)
def _b_ma_trend(a, p, v):      return factors.ma_trend(p, short=20, long=60)
def _b_ma_breakout(a, p, v):   return factors.ma_breakout(p, window=60)
def _b_combine_mom_lv(a, p, v):
    return factors.combine(
        {"momentum": factors.momentum(p, lookback=120),
         "low_volatility": factors.low_volatility(p, lookback=60)},
        {"momentum": 1.0, "low_volatility": 0.5}, method="rank")
def _b_combine_mom_rev(a, p, v):
    return factors.combine(
        {"momentum": factors.momentum(p, lookback=120),
         "reversal": factors.reversal(p, lookback=20)},
        {"momentum": 1.0, "reversal": 0.5}, method="rank")


PREDEFINED_BUILDERS = {
    "momentum_120": _b_momentum_120,
    "momentum_60": _b_momentum_60,
    "reversal_20": _b_reversal_20,
    "reversal_5": _b_reversal_5,
    "low_vol_60": _b_low_vol_60,
    "ma_trend": _b_ma_trend,
    "ma_breakout": _b_ma_breakout,
    "combine_mom_lv": _b_combine_mom_lv,
    "combine_mom_rev": _b_combine_mom_rev,
}


# ------------------------------------------------------------- 解析

def parse_combine_spec(text: str) -> list[tuple[str, str]]:
    """解析 `--combine "消费=reversal_20,蓝筹=low_vol_60"`。

    返回 [(pool, strategy), ...] 列表。
    """
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"格式错误：{part!r} 应为 '池子=策略'，如 '消费=reversal_20'")
        pool, _, strategy = part.partition("=")
        pool, strategy = pool.strip(), strategy.strip()
        if strategy not in PREDEFINED_BUILDERS:
            raise ValueError(f"未知策略 {strategy!r}，可选: {sorted(PREDEFINED_BUILDERS)}")
        items.append((pool, strategy))
    if not items:
        raise ValueError("--combine 不能为空")
    return items


def parse_weights(text: str, n: int) -> list[float]:
    """解析 `--combine-weights "0.5,0.3,0.2"`，自动归一化。空则返回等权。"""
    if not text or not text.strip():
        return [1.0 / n] * n
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != n:
        raise ValueError(f"--combine-weights 数量 {len(parts)} ≠ 池子数 {n}")
    s = sum(parts)
    if s <= 0:
        raise ValueError("--combine-weights 总和必须 > 0")
    return [w / s for w in parts]


# ------------------------------------------------------------- 单池回测

def run_one_pool(pool: str, strategy: str, args, fetch_start: str, start_ts: pd.Timestamp):
    """跑一个 (pool, strategy) 的完整回测，返回 (equity, metrics, code_count)。

    复用 load_panel 与 feasibility filters，每个池子单独跑一次（开销较大但结果干净）。
    """
    codes = universe.build_universe(pool, as_of=getattr(args, "as_of", "") or args.start)
    if not codes:
        raise ValueError(f"池子 {pool!r} 无成分股")
    print(f"  [{pool:>6s} + {strategy:18s}] 池子 {len(codes)} 只", flush=True)
    panel = load_panel(codes, fetch_start, args.end, sleep=0.3, on_error="skip")
    prices_all = panel["close"]
    open_all = panel["open"]
    volume_all = panel["volume"]
    high_all = panel.get("high", pd.DataFrame())
    low_all = panel.get("low", pd.DataFrame())
    if prices_all.empty:
        raise RuntimeError(f"池子 {pool!r} 拉不到任何数据")
    # 可行性约束
    st_codes = [c.strip() for c in args.st_codes.split(",") if c.strip()]
    limit_pct = filters.limit_pct_by_code(codes, st_codes)
    illiquid = filters.illiquid_mask(volume_all, args.min_volume)
    can_buy, can_sell = filters.tradability(
        prices_all, volume_all, limit_pct=limit_pct, illiquid=illiquid,
        enable_limit=not args.no_limit_filter,
        enable_suspend=not args.no_suspend_filter,
        open_=open_all if args.use_open else None,
        high=high_all if not high_all.empty else None,
        low=low_all if not low_all.empty else None,
        exec_at_open=args.use_open,
    )
    score = PREDEFINED_BUILDERS[strategy](args, prices_all, volume_all)
    # 次新股过滤（与 main.py 口径一致）
    min_listed = getattr(args, "min_listed_days", 0)
    if min_listed > 0:
        score = score.where(filters.listing_age_mask(prices_all, min_listed))
    weights = factor_weights(score, top_n=args.top_n, freq=args.rebalance,
                             min_names=args.min_names, buffer=args.buffer,
                             can_buy=can_buy, can_sell=can_sell,
                             exec_shift=1 if args.use_open else 0)
    keep = prices_all.index >= start_ts
    p = prices_all.loc[keep]; o = open_all.loc[keep]; w = weights.loc[keep]
    equity, metrics, _ = run_one_backtest(p, w, o if args.use_open else None,
                                           fee=args.fee, stamp_tax=args.stamp_tax)
    return equity, metrics, len(prices_all.columns)


def run_one_backtest(prices, weights, open_prices, fee, stamp_tax):
    """简单封装 backtest.run，便于阅读。"""
    return backtest_run(prices, weights, open_prices=open_prices,
                        fee=fee, stamp_tax=stamp_tax)


# ------------------------------------------------------------- 组合整合

def combine_equity(equities: dict[str, pd.Series], weights: list[float]) -> pd.Series:
    """把多个池子的净值曲线按权重合成成一条组合净值。

    每条净值归一化到起点 1.0；按权重加权求和；总市值也归一化到 1.0。
    """
    names = list(equities.keys())
    common_idx = None
    for eq in equities.values():
        common_idx = eq.index if common_idx is None else common_idx.union(eq.index)
    common_idx = common_idx.sort_values()
    aligned = pd.DataFrame({n: equities[n].reindex(common_idx).ffill().bfill() for n in names})
    # 各池归一化到 1.0
    norm = aligned.div(aligned.iloc[0])
    # 加权
    w = pd.Series(weights, index=names)
    combo = (norm * w.values).sum(axis=1)
    # 整体归一化
    combo = combo / combo.iloc[0]
    return combo


def correlation_matrix(equities: dict[str, pd.Series]) -> pd.DataFrame:
    """按日收益计算池子间相关系数矩阵（更稳健）。"""
    names = list(equities.keys())
    common_idx = None
    for eq in equities.values():
        common_idx = eq.index if common_idx is None else common_idx.union(eq.index)
    common_idx = common_idx.sort_values()
    aligned = pd.DataFrame({n: equities[n].reindex(common_idx).ffill().bfill() for n in names})
    daily = aligned.pct_change().dropna()
    return daily.corr()


def portfolio_metrics(equity: pd.Series) -> dict:
    """简化的组合指标计算：总收益 / 年化 / 夏普 / 最大回撤 / 年化波动。"""
    rets = equity.pct_change().dropna()
    if len(rets) == 0:
        return {}
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    annual_return = float((1 + total_return) ** (1 / years) - 1)
    vol = float(rets.std() * (252 ** 0.5))
    sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() else 0.0
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "years": float(years),
        "trading_days": int(len(equity)),
    }


# ------------------------------------------------------------- 主流程

def run_combine(combine_spec, weights_arg, args, fetch_start, benchmark_curve=None):
    """跑多池组合：每个池子单独回测，整合 + 汇总指标。

    返回：
      pool_results    每池一行指标
      pool_metrics    {pool_name: metrics}
      equities        {pool_name: equity}
      correlation     相关性矩阵
      combo_equity    组合净值曲线
      combo_metrics   组合指标
      weights         池子间权重
    """
    start_ts = pd.Timestamp(args.start)
    weights = parse_weights(weights_arg, len(combine_spec))

    print(f"\n【多池组合】{len(combine_spec)} 个池子，权重 {weights}")
    pool_metrics: dict[str, dict] = {}
    pool_equity: dict[str, pd.Series] = {}
    pool_rows: list[dict] = []
    for (pool, strategy), w in zip(combine_spec, weights):
        eq, m, n_codes = run_one_pool(pool, strategy, args, fetch_start, start_ts)
        pool_metrics[pool] = m
        pool_equity[pool] = eq
        pool_rows.append({
            "pool": pool,
            "strategy": strategy,
            "n_codes": n_codes,
            "weight": w,
            **{k: m.get(k, 0.0) for k in (
                "total_return", "annual_return", "sharpe",
                "max_drawdown", "annual_volatility")},
        })

    combo_eq = combine_equity(pool_equity, weights)
    combo_m = portfolio_metrics(combo_eq)
    # 组合 vs 基准（如果有）
    if benchmark_curve is not None:
        bench = benchmark_curve.reindex(combo_eq.index).ffill().bfill()
        if len(bench) == len(combo_eq) and bench.iloc[0]:
            excess = combo_eq / bench
            combo_m["excess_return"] = float(excess.iloc[-1] - 1)
            ex_daily = excess.pct_change().fillna(0.0)
            combo_m["information_ratio"] = (
                float(ex_daily.mean() / ex_daily.std() * (252 ** 0.5))
                if ex_daily.std() else 0.0
            )

    corr = correlation_matrix(pool_equity)

    return (
        pd.DataFrame(pool_rows),
        pool_metrics, pool_equity, corr,
        combo_eq, combo_m, weights,
    )


# ------------------------------------------------------------- 报告

def build_combine_report(pool_results, pool_equity, corr, combo_eq, combo_m,
                         weights, out_html, title="多池组合", subtitle=""):
    """生成 ECharts HTML：净值对比 + 相关性矩阵 + 组合 vs 池子明细。"""
    data = {
        "pools": pool_results["pool"].tolist(),
        "strategies": pool_results["strategy"].tolist(),
        "weights": [round(float(w), 3) for w in pool_results["weight"]],
        "n_codes": [int(n) for n in pool_results["n_codes"]],
        "pool_total_return": [round(float(v), 4) for v in pool_results["total_return"]],
        "pool_sharpe": [round(float(v), 3) for v in pool_results["sharpe"]],
        "pool_max_dd": [round(float(v), 4) for v in pool_results["max_drawdown"]],
        "combo_total_return": round(float(combo_m.get("total_return", 0)), 4),
        "combo_annual_return": round(float(combo_m.get("annual_return", 0)), 4),
        "combo_sharpe": round(float(combo_m.get("sharpe", 0)), 3),
        "combo_max_dd": round(float(combo_m.get("max_drawdown", 0)), 4),
        "combo_vol": round(float(combo_m.get("annual_volatility", 0)), 4),
        "combo_excess": round(float(combo_m.get("excess_return", 0)), 4),
        "combo_ir": round(float(combo_m.get("information_ratio", 0)), 3),
    }

    # 净值曲线数据：所有池 + 组合
    common_idx = None
    for s in list(pool_equity.values()) + [combo_eq]:
        idx = s.index
        common_idx = idx if common_idx is None else common_idx.union(idx)
    common_idx = common_idx.sort_values()
    palette = ["#1677ff", "#52c41a", "#fa8c16", "#eb2f96",
               "#13c2c2", "#fadb14", "#722ed1", "#9254de"]

    equity_lines = []
    series_defs = [(p, s) for p, s in zip(pool_equity.keys(), pool_results["strategy"])]
    for i, (p, s) in enumerate(series_defs):
        eq = pool_equity[p].reindex(common_idx).ffill().bfill()
        equity_lines.append({
            "name": f"{p}({s})",
            "type": "line", "showSymbol": False, "smooth": True,
            "lineStyle": {"width": 1.5, "type": "dashed"},
            "itemStyle": {"color": palette[i % len(palette)]},
            "data": [round(float(v), 4) for v in eq.values],
        })
    # 组合线（粗实线）
    eq = combo_eq.reindex(common_idx).ffill().bfill()
    equity_lines.append({
        "name": "组合(权重加权)",
        "type": "line", "showSymbol": False, "smooth": True,
        "lineStyle": {"width": 3.5},
        "itemStyle": {"color": "#cf1322"},
        "data": [round(float(v), 4) for v in eq.values],
    })
    data["dates"] = [d.strftime("%Y-%m-%d") for d in common_idx]
    data["equity_lines"] = equity_lines

    # 相关性矩阵
    data["corr_codes"] = list(corr.columns)
    data["corr_values"] = [[round(float(v), 3) for v in row] for row in corr.values]

    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__TITLE__", title).replace("__SUBTITLE__", subtitle)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  body { margin: 0; padding: 24px; font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #f5f5f5; color: #1f1f1f; }
  h1 { margin: 0 0 8px; font-size: 22px; }
  .sub { color: #666; margin-bottom: 24px; font-size: 13px; }
  .group { background: #fff; border: 1px solid #e8e8e8; border-radius: 12px;
           padding: 16px 20px; margin-bottom: 16px; }
  h2 { margin: 0 0 12px; font-size: 16px; }
  .chart { width: 100%; height: 460px; }
  .chart-sm { width: 100%; height: 380px; }
  .note { color: #888; font-size: 12px; margin-top: 8px; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
             gap: 12px; margin-bottom: 12px; }
  .metric { background: #fafafa; border: 1px solid #eee; padding: 10px 12px; border-radius: 8px; }
  .metric .lbl { color: #666; font-size: 12px; margin-bottom: 4px; }
  .metric .val { font-size: 20px; font-weight: 600; }
  .pos { color: #cf1322; } .neg { color: #389e0d; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #f0f0f0; text-align: right; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  th { background: #fafafa; font-weight: 600; color: #333; }
</style></head><body>
<h1>__TITLE__</h1><div class="sub">__SUBTITLE__</div>

<div class="group">
  <h2>组合整体指标</h2>
  <div class="metrics" id="combo-metrics"></div>
</div>

<div class="group">
  <h2>净值曲线对比</h2>
  <div id="equity" class="chart"></div>
  <div class="note">虚线 = 各池子独立回测；红色实线 = 按指定权重加权的组合净值（强相关池子少时红利最明显）。</div>
</div>

<div class="group">
  <h2>池子间相关性矩阵（按日收益计算）</h2>
  <div id="corr" class="chart-sm"></div>
  <div class="note">值越接近 0 表示池子越独立，组合分散效果越好；接近 1 表示几乎同涨同跌。</div>
</div>

<div class="group">
  <h2>各池明细</h2>
  <table id="tbl"><thead><tr>
    <th>池子</th><th>策略</th><th>权重</th><th>代码数</th>
    <th>总收益</th><th>夏普</th><th>最大回撤</th>
  </tr></thead><tbody></tbody></table>
</div>

<script>
const DATA = __DATA__;
const PALETTE = ['#1677ff','#52c41a','#fa8c16','#eb2f96','#13c2c2','#fadb14','#722ed1','#9254de'];

// 组合指标卡片
const m = {
  '总收益': DATA.combo_total_return,
  '年化': DATA.combo_annual_return,
  '夏普': DATA.combo_sharpe,
  '最大回撤': DATA.combo_max_dd,
  '年化波动': DATA.combo_vol,
  '超额收益': DATA.combo_excess,
  '信息比率': DATA.combo_ir,
};
const wrap = document.getElementById('combo-metrics');
Object.entries(m).forEach(([k, v]) => {
  const d = document.createElement('div'); d.className = 'metric';
  const cls = (k === '夏普' || k === '信息比率') ? '' : (v >= 0 ? 'pos' : 'neg');
  const fmt = (k === '夏普' || k === '信息比率') ? v.toFixed(2) : (v*100).toFixed(2)+'%';
  d.innerHTML = `<div class="lbl">${k}</div><div class="val ${cls}">${fmt}</div>`;
  wrap.appendChild(d);
});

// 净值图
const ecEq = echarts.init(document.getElementById('equity'));
ecEq.setOption({
  tooltip: { trigger: 'axis' },
  legend: { top: 0 },
  grid: { left: 60, right: 30, top: 50, bottom: 50 },
  xAxis: { type: 'category', data: DATA.dates, axisLabel: { fontSize: 10 } },
  yAxis: { type: 'value', name: '净值' },
  series: DATA.equity_lines
});

// 相关性矩阵热力图
const ecCorr = echarts.init(document.getElementById('corr'));
const codes = DATA.corr_codes;
const heatData = [];
codes.forEach((x, i) => codes.forEach((y, j) => heatData.push([i, j, DATA.corr_values[i][j]])));
ecCorr.setOption({
  tooltip: { position: 'top', formatter: p => `${codes[p.value[0]]} ↔ ${codes[p.value[1]]}: ${p.value[2].toFixed(3)}` },
  grid: { left: 80, right: 30, top: 30, bottom: 80 },
  xAxis: { type: 'category', data: codes, axisLabel: { rotate: 30 } },
  yAxis: { type: 'category', data: codes },
  visualMap: { min: -1, max: 1, calculable: true,
               orient: 'horizontal', left: 'center', bottom: 0,
               inRange: { color: ['#389e0d','#fff','#cf1322'] } },
  series: [{ type: 'heatmap', data: heatData, label: { show: true, fontSize: 11 } }]
});

// 明细表
const tb = document.querySelector('#tbl tbody');
DATA.pools.forEach((p, i) => {
  const tr = document.createElement('tr');
  const pct = v => `<td class="${v>=0?'pos':'neg'}">${(v*100).toFixed(2)}%</td>`;
  tr.innerHTML = `
    <td><b>${p}</b></td><td style="text-align:left">${DATA.strategies[i]}</td>
    <td>${(DATA.weights[i]*100).toFixed(0)}%</td>
    <td>${DATA.n_codes ? DATA.n_codes[i] : '—'}</td>
    ${pct(DATA.pool_total_return[i])}
    <td>${DATA.pool_sharpe[i].toFixed(3)}</td>${pct(DATA.pool_max_dd[i])}`;
  tb.appendChild(tr);
});

window.addEventListener('resize', () => { ecEq.resize(); ecCorr.resize(); });
</script></body></html>
"""