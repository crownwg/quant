"""多策略对比框架：同一池子横向跑多个策略，挑稳健组合。

设计目标
--------
- 把现有 factors.py 的 7 个因子 / strategy.py 的 factor_weights / backtest.run 串成一个
  "批量回测 → 收集指标 → 输出对比报告"的工作流。
- 预定义 6-8 个代表策略（动量长短周期、反转、低波、均线、合成），覆盖常见风格。
- 复用归因的 annual_breakdown 给每个策略做年度分解，自动暴露"牛市 vs 熊市"差异。

不在这里做的事
--------------
- 选股约束（涨跌停/停牌/流动性）由外部传入 can_buy/can_sell（与单次回测完全一致）。
- 调仓频率 / top_n / buffer / 成本从 args 透传，避免重新造配置。
"""

from __future__ import annotations

import pandas as pd

from . import factors, rolling
from .backtest import run as backtest_run
from .strategy import factor_weights


# ------------------------------------------------------- 预定义策略池

def _b_momentum_120(a, p, v):
    return factors.momentum(p, lookback=120)


def _b_momentum_60(a, p, v):
    return factors.momentum(p, lookback=60)


def _b_reversal_20(a, p, v):
    return factors.reversal(p, lookback=20)


def _b_low_vol_60(a, p, v):
    return factors.low_volatility(p, lookback=60)


def _b_ma_trend(a, p, v):
    return factors.ma_trend(p, short=20, long=60)


def _b_ma_breakout(a, p, v):
    return factors.ma_breakout(p, window=60)


def _b_combine_mom_lv(a, p, v):
    return factors.combine(
        {"momentum": factors.momentum(p, lookback=120),
         "low_volatility": factors.low_volatility(p, lookback=60)},
        {"momentum": 1.0, "low_volatility": 0.5},
        method="rank",
    )


def _b_combine_mom_rev(a, p, v):
    return factors.combine(
        {"momentum": factors.momentum(p, lookback=120),
         "reversal": factors.reversal(p, lookback=20)},
        {"momentum": 1.0, "reversal": 0.5},
        method="rank",
    )


PREDEFINED_STRATEGIES = {
    "momentum_120":    {"description": "120日动量（经典低频动量）",      "builder": _b_momentum_120},
    "momentum_60":     {"description": "60日动量（短周期动量）",         "builder": _b_momentum_60},
    "reversal_20":     {"description": "20日反转（短期反弹）",          "builder": _b_reversal_20},
    "low_vol_60":      {"description": "60日低波动（低波动异象）",       "builder": _b_low_vol_60},
    "ma_trend":        {"description": "均线趋势（短20/长60）",        "builder": _b_ma_trend},
    "ma_breakout":     {"description": "60日均线突破",                "builder": _b_ma_breakout},
    "combine_mom_lv":  {"description": "动量+低波动合成（rank法）",     "builder": _b_combine_mom_lv},
    "combine_mom_rev": {"description": "动量+反转合成（对冲）",         "builder": _b_combine_mom_rev},
}

# 默认横向对比的 7 个：动量长短/反转/低波/均线趋势/均线突破/动量+低波动合成
DEFAULT_STRATEGIES = [
    "momentum_120", "momentum_60", "reversal_20",
    "low_vol_60", "ma_trend", "ma_breakout", "combine_mom_lv",
]


# ---------------------------------------------------------- 工具函数

def parse_strategy_spec(text: str) -> list[str]:
    """解析 `--compare-strategies "momentum_120,reversal_20"`。"""
    if not text or not text.strip():
        return list(DEFAULT_STRATEGIES)
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in PREDEFINED_STRATEGIES]
    if unknown:
        raise ValueError(
            f"未知策略 {unknown}，可选: {sorted(PREDEFINED_STRATEGIES)}"
        )
    return names


# ------------------------------------------------------- 主流程

def run_compare(prices, open_prices, volume, can_buy, can_sell,
                strategies, args, benchmark_curve=None):
    """逐策略跑回测，汇总指标 + 净值曲线 + 年度收益。

    返回
    ----
    results_df    : 每行一个策略的核心指标
    equity_dict   : {策略名: pd.Series(净值)}
    annual_dict   : {策略名: pd.DataFrame(年度分解)}
    params_dict   : {策略名: 策略描述}
    """
    rows: list[dict] = []
    equity_dict: dict[str, pd.Series] = {}
    annual_dict: dict[str, pd.DataFrame] = {}
    params_dict: dict[str, str] = {}

    start_ts = pd.Timestamp(args.start)

    for name in strategies:
        cfg = PREDEFINED_STRATEGIES[name]
        score = cfg["builder"](args, prices, volume)
        weights = factor_weights(score, top_n=args.top_n, freq=args.rebalance,
                                 min_names=args.min_names, buffer=args.buffer,
                                 can_buy=can_buy, can_sell=can_sell)
        # 切片到用户回测区间
        keep = prices.index >= start_ts
        p = prices.loc[keep]
        o = open_prices.loc[keep]
        w = weights.loc[keep]
        equity, metrics, detail = backtest_run(
            p, w,
            open_prices=o if args.use_open else None,
            fee=args.fee, stamp_tax=args.stamp_tax,
        )
        # 超额 vs 基准（如果提供）
        if benchmark_curve is not None:
            bench = benchmark_curve.reindex(equity.index).ffill().bfill()
            if len(bench) == len(equity) and bench.iloc[0]:
                excess = equity / bench
                metrics["excess_return"] = float(excess.iloc[-1] - 1)
                ex_daily = excess.pct_change().fillna(0.0)
                metrics["information_ratio"] = (
                    float(ex_daily.mean() / ex_daily.std() * (252 ** 0.5))
                    if ex_daily.std() else 0.0
                )
        # 年度分解
        annual = rolling.annual_breakdown(equity, detail)
        annual_dict[name] = annual
        equity_dict[name] = equity

        rows.append({
            "strategy": name,
            "description": cfg["description"],
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "annual_volatility": metrics["annual_volatility"],
            "annual_turnover": metrics.get("annual_turnover", 0.0),
            "excess_return": metrics.get("excess_return", 0.0),
            "information_ratio": metrics.get("information_ratio", 0.0),
        })
        params_dict[name] = cfg["description"]
        print(f"  [{name:18s}] 收益 {metrics['total_return']:.1%}"
              f" 夏普 {metrics['sharpe']:.2f} 回撤 {metrics['max_drawdown']:.1%}", flush=True)

    results_df = pd.DataFrame(rows)
    return results_df, equity_dict, annual_dict, params_dict


# ------------------------------------------------------- 报告生成

def build_compare_report(results_df, equity_dict, annual_dict, params_dict,
                         out_html, title="多策略对比", subtitle=""):
    """生成 ECharts HTML 报告：核心指标柱状图 + 净值叠加曲线 + 年度收益折线对比。"""
    import json

    # ---- 准备数据 ----
    strategies = results_df["strategy"].tolist()
    descs = [params_dict.get(s, s) for s in strategies]
    sharpe = [round(float(v), 3) for v in results_df["sharpe"]]
    total_ret = [round(float(v), 4) for v in results_df["total_return"]]
    annual_ret = [round(float(v), 4) for v in results_df["annual_return"]]
    max_dd = [round(float(v), 4) for v in results_df["max_drawdown"]]
    annual_vol = [round(float(v), 4) for v in results_df["annual_volatility"]]
    annual_to = [round(float(v), 4) for v in results_df["annual_turnover"]]
    excess = [round(float(v), 4) for v in results_df["excess_return"]]
    info_ratio = [round(float(v), 3) for v in results_df["information_ratio"]]

    # 净值曲线数据：每条线一个 series，对齐到共同时间轴
    common_index = None
    for s in strategies:
        idx = equity_dict[s].index
        common_index = idx if common_index is None else common_index.union(idx)
    common_index = common_index.sort_values()

    equity_lines = []
    palette = ["#722ed1", "#1677ff", "#52c41a", "#fa8c16",
               "#eb2f96", "#13c2c2", "#fadb14", "#9254de"]
    for i, s in enumerate(strategies):
        eq = equity_dict[s].reindex(common_index).ffill().bfill()
        equity_lines.append({
            "name": s,
            "type": "line",
            "showSymbol": False,
            "smooth": True,
            "lineStyle": {"width": 2},
            "itemStyle": {"color": palette[i % len(palette)]},
            "data": [round(float(v), 4) for v in eq.values],
        })

    # 年度收益折线对比：横轴是年份，每个策略一条线
    annual_lines = []
    years = None
    for s in strategies:
        y = annual_dict[s]["year"].tolist()
        years = y if years is None else sorted(set(years) | set(y))
    years = sorted(set(years))
    for i, s in enumerate(strategies):
        df = annual_dict[s].set_index("year")["return"]
        ys = [float(df.get(y, 0.0)) for y in years]
        annual_lines.append({
            "name": s,
            "type": "line",
            "showSymbol": True,
            "smooth": False,
            "lineStyle": {"width": 2},
            "itemStyle": {"color": palette[i % len(palette)]},
            "data": [round(v, 4) for v in ys],
        })

    data = {
        "strategies": strategies,
        "descs": descs,
        "sharpe": sharpe,
        "total_return": total_ret,
        "annual_return": annual_ret,
        "max_drawdown": max_dd,
        "annual_volatility": annual_vol,
        "annual_turnover": annual_to,
        "excess_return": excess,
        "information_ratio": info_ratio,
        "dates": [d.strftime("%Y-%m-%d") for d in common_index],
        "equity_lines": equity_lines,
        "years": [int(y) for y in years],
        "annual_lines": annual_lines,
    }

    # ---- HTML 模板 ----
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
  h2 { margin: 0 0 12px; font-size: 16px; color: #1f1f1f; }
  .chart { width: 100%; height: 420px; }
  .chart-sm { width: 100%; height: 360px; }
  .note { color: #888; font-size: 12px; margin-top: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #f0f0f0; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #fafafa; font-weight: 600; color: #333; }
  .pos { color: #cf1322; } .neg { color: #389e0d; }
</style></head><body>
<h1>__TITLE__</h1><div class="sub">__SUBTITLE__</div>

<div class="group">
  <h2>核心指标对比</h2>
  <div id="bar-sharpe" class="chart-sm"></div>
  <div id="bar-ret" class="chart-sm"></div>
  <div id="bar-dd" class="chart-sm"></div>
  <div class="note">红色=上涨/超额为正/回撤（数值为负），柱长按数值绝对值显示。</div>
</div>

<div class="group">
  <h2>净值曲线叠加</h2>
  <div id="equity" class="chart"></div>
  <div class="note">初始净值归一为 1.0；曲线整体越靠右上越好；重合度高=策略相关性高。</div>
</div>

<div class="group">
  <h2>分年度收益对比</h2>
  <div id="annual" class="chart"></div>
  <div class="note">横轴为年份；同一年的不同策略点连成线，便于看出"策略之间的年度相关性"。</div>
</div>

<div class="group">
  <h2>明细表</h2>
  <table id="tbl"><thead><tr>
    <th>策略</th><th>描述</th><th>总收益</th><th>年化</th><th>夏普</th>
    <th>最大回撤</th><th>年化波动</th><th>年化换手</th><th>超额</th><th>信息比</th>
  </tr></thead><tbody></tbody></table>
</div>

<script>
const DATA = __DATA__;
const COLORS = ['#722ed1','#1677ff','#52c41a','#fa8c16','#eb2f96','#13c2c2','#fadb14','#9254de'];

function barChart(id, title, data, suffix) {
  const ec = echarts.init(document.getElementById(id));
  const option = {
    title: { text: title, left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
               formatter: p => p.map(x => `${x.marker}${x.name}: ${(x.value*100).toFixed(1)}%`).join('<br>') },
    grid: { left: 50, right: 20, top: 40, bottom: 60 },
    xAxis: { type: 'category', data: DATA.strategies, axisLabel: { rotate: 20, fontSize: 11 } },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
    series: [{
      type: 'bar', barWidth: '50%',
      data: data.map((v, i) => ({
        value: +(v*100).toFixed(2),
        itemStyle: { color: COLORS[i % COLORS.length] }
      }))
    }]
  };
  ec.setOption(option);
  return ec;
}

const ec1 = barChart('bar-sharpe', '夏普比率（越大越好）',
  DATA.sharpe.map(v => v / 2), '');  // 缩放到 % 量级，仅用于对比
const ec2 = barChart('bar-ret', '总收益（越大越好）', DATA.total_return, '');
const ec3 = barChart('bar-dd', '最大回撤（柱长=回撤幅度，越小越好）',
  DATA.max_drawdown, '');

const ecEq = echarts.init(document.getElementById('equity'));
ecEq.setOption({
  tooltip: { trigger: 'axis' },
  legend: { top: 0, data: DATA.strategies },
  grid: { left: 60, right: 30, top: 40, bottom: 50 },
  xAxis: { type: 'category', data: DATA.dates, axisLabel: { fontSize: 10 } },
  yAxis: { type: 'value', name: '净值' },
  series: DATA.equity_lines
});

const ecAn = echarts.init(document.getElementById('annual'));
ecAn.setOption({
  tooltip: { trigger: 'axis' },
  legend: { top: 0, data: DATA.strategies },
  grid: { left: 60, right: 30, top: 40, bottom: 40 },
  xAxis: { type: 'category', data: DATA.years },
  yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
  series: DATA.annual_lines
});

// 明细表
const tb = document.querySelector('#tbl tbody');
DATA.strategies.forEach((s, i) => {
  const tr = document.createElement('tr');
  const pct = v => `<td class="${v>=0?'pos':'neg'}">${(v*100).toFixed(2)}%</td>`;
  const num = v => `<td>${(v*100).toFixed(2)}%</td>`;
  const d = DATA;
  tr.innerHTML = `
    <td><b>${s}</b></td><td style="text-align:left;color:#666">${d.descs[i]}</td>
    ${num(d.total_return[i])}${num(d.annual_return[i])}
    <td>${d.sharpe[i].toFixed(3)}</td>${num(d.max_drawdown[i])}
    ${num(d.annual_volatility[i])}${num(d.annual_turnover[i])}
    ${num(d.excess_return[i])}
    <td>${d.information_ratio ? d.information_ratio[i].toFixed(2) : '—'}</td>`;
  tb.appendChild(tr);
});

window.addEventListener('resize', () => {
  ec1.resize(); ec2.resize(); ec3.resize(); ecEq.resize(); ecAn.resize();
});
</script></body></html>
"""