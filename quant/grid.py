"""参数优化：在 (lookback, top_n, buffer) 网格上回测，找「稳健最优」参数。

为什么需要它
------------
阶段5/风险归因已经说明：策略收益高度依赖行情状态，单一「全样本最优」参数
往往是曲线拟合（凑巧在 2020 消费牛里最强）。本模块做网格搜索，并对每个
参数组合同时给出：
  - 全样本指标：总收益 / 年化 / 夏普 / 最大回撤 / 卡玛 / 年化换手
  - 稳健性代理：最差年度收益、正收益年度占比（年度分解）
  - 若提供基准：超额收益、信息比率
让用户既能看「哪个参数最猛」，也能看「哪个参数最不挑行情」。

设计要点
--------
- 价量面板（含预热期）只加载一次，网格内复用；只有「因子打分 / 选股 / 回测」
  随参数变化，避免重复下载。
- 因子默认用动量（主旋钮是 lookback）；top_n / buffer 控制选股范围与换手。
- 多因子 / 其它策略的网格是后续扩展，这里先把动量这一条线做扎实。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from .backtest import run
from .strategy import factor_weights
from . import factors, rolling


def parse_ints(text: str) -> list[int]:
    """把 '60,90,120' 解析成 [60, 90, 120]。"""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def grid_search(prices_all: pd.DataFrame, open_all: pd.DataFrame,
                can_buy, can_sell, args,
                lookbacks: list[int], top_ns: list[int], buffers: list[int],
                start_ts: pd.Timestamp, benchmark_curve: pd.Series | None = None) -> pd.DataFrame:
    """对每组合回测，返回结果 DataFrame（一行一组合）。"""
    rows = []
    combos = list(itertools.product(lookbacks, top_ns, buffers))
    total = len(combos)
    for i, (lb, tn, buf) in enumerate(combos, 1):
        score = factors.momentum(prices_all, lb, args.skip_recent)
        weights_all = factor_weights(score, top_n=tn, freq=args.rebalance,
                                     min_names=args.min_names, buffer=buf,
                                     can_buy=can_buy, can_sell=can_sell)
        keep = weights_all.index >= start_ts
        prices = prices_all.loc[keep]
        weights = weights_all.loc[keep]
        op = open_all.loc[keep] if args.use_open else None
        equity, m, detail = run(prices, weights, open_prices=op,
                                fee=args.fee, stamp_tax=args.stamp_tax)
        annual = rolling.annual_breakdown(equity, detail)
        worst_year = float(annual["return"].min()) if len(annual) else 0.0
        pos_year = float((annual["return"] > 0).mean()) if len(annual) else 0.0

        rec = {
            "lookback": lb, "top_n": tn, "buffer": buf,
            "total_return": m["total_return"], "annual_return": m["annual_return"],
            "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"], "annual_turnover": m["annual_turnover"],
            "worst_year_return": worst_year, "positive_year_ratio": pos_year,
        }
        if benchmark_curve is not None:
            bench = benchmark_curve.reindex(equity.index).ffill().bfill()
            if len(bench) == len(equity) and bench.iloc[0]:
                bench_eq = bench / bench.iloc[0]          # 基准原始收盘价 -> 净值（首日=1）
                excess = equity / bench_eq
                rec["excess_return"] = float(excess.iloc[-1] - 1)
                ex_daily = excess.pct_change().fillna(0.0)
                rec["information_ratio"] = (float(ex_daily.mean() / ex_daily.std() * (252 ** 0.5))
                                           if ex_daily.std() else 0.0)
        rows.append(rec)
        print(f"  [{i}/{total}] lb={lb} n={tn} buf={buf} | 收益 {m['total_return']:.1%}"
              f" 夏普 {m['sharpe']:.2f} 回撤 {m['max_drawdown']:.1%} 最差年 {worst_year:.1%}", flush=True)
    return pd.DataFrame(rows)


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  :root { --bg:#fff; --surface:#fafafa; --border:#e6e6e6; --text:#1f2329; --muted:#6b7280; --accent:#2f6df0; }
  * { box-sizing: border-box; }
  body { font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; margin:0; background:var(--bg); color:var(--text); line-height:1.6; }
  .wrap { max-width:1080px; margin:0 auto; padding:32px 24px 64px; }
  h1 { font-size:22px; font-weight:600; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:24px; }
  .group { margin-bottom:28px; }
  .group h2 { font-size:14px; font-weight:600; color:var(--muted); margin:0 0 10px; border-left:3px solid var(--accent); padding-left:8px; }
  .chart { width:100%; height:420px; background:var(--surface); border:1px solid var(--border); border-radius:12px; margin-bottom:24px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { border:1px solid var(--border); padding:7px 9px; text-align:right; }
  th { background:var(--surface); color:var(--muted); font-weight:600; position:sticky; top:0; }
  td:first-child,th:first-child { text-align:left; }
  tr:nth-child(even) td { background:#fcfcfc; }
  .pos { color:#389e0d; } .neg { color:#d4380d; }
  .note { color:var(--muted); font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="group">
    <h2>夏普热力图（按 lookback × top_n，buffer 取均值）</h2>
    <div class="chart" id="heat"></div>
    <div class="note">颜色越红夏普越高；横轴 top_n、纵轴 lookback。看清「哪片区域普遍好」，比单点最优更抗过拟合。</div>
  </div>
  <div class="group">
    <h2>全部组合明细（按夏普降序）</h2>
    <div style="max-height:520px; overflow:auto;">
      <table id="tbl"></table>
    </div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const axisCommon = { axisLine:{lineStyle:{color:'#e6e6e6'}}, axisLabel:{color:'#6b7280'} };

const topNs = DATA.top_ns, lookbacks = DATA.lookbacks;
const maxV = Math.max.apply(null, DATA.heat.map(function(h){return h[2];}));
const minV = Math.min.apply(null, DATA.heat.map(function(h){return h[2];}));
const heat = echarts.init(document.getElementById('heat'));
heat.setOption({
  tooltip:{ position:'top', formatter:function(p){ return 'lookback='+lookbacks[p.value[1]]+' top_n='+topNs[p.value[0]]+'<br/>夏普 '+p.value[2].toFixed(2); } },
  grid:{ left:80, right:30, top:30, bottom:70 },
  xAxis:{ type:'category', data:topNs.map(function(n){return 'n='+n;}), axisLine:axisCommon.axisLine, axisLabel:axisCommon.axisLabel, name:'top_n', nameLocation:'middle', nameGap:34 },
  yAxis:{ type:'category', data:lookbacks.map(function(l){return 'lb='+l;}), axisLine:axisCommon.axisLine, axisLabel:axisCommon.axisLabel, name:'lookback', nameLocation:'middle', nameGap:56 },
  visualMap:{ min:minV, max:maxV, calculable:true, orient:'horizontal', left:'center', bottom:10,
    inRange:{ color:['#d4380d','#fdf2e0','#389e0d'] } },
  series:[{ type:'heatmap', data:DATA.heat,
    label:{ show:true, formatter:function(p){return p.value[2].toFixed(2);}, color:'#1f2329', fontSize:11 },
    emphasis:{ itemStyle:{ shadowBlur:8, shadowColor:'rgba(0,0,0,0.3)' } } }]
});

const cols = DATA.columns;
const tbl = document.getElementById('tbl');
let html = '<tr>' + cols.map(function(c){ return '<th>'+c.label+'</th>'; }).join('') + '</tr>';
DATA.rows.forEach(function(r){
  let tds = '';
  cols.forEach(function(c){
    const v = r[c.key];
    if (c.fmt === 'pct') { const cls = v>=0?'pos':'neg'; tds += '<td class="'+cls+'">'+(v*100).toFixed(1)+'%</td>'; }
    else if (c.fmt === 'int') { tds += '<td>'+v+'</td>'; }
    else { tds += '<td>'+ (typeof v==='number'? v.toFixed(2): v) +'</td>'; }
  });
  html += '<tr>'+tds+'</tr>';
});
tbl.innerHTML = html;
window.addEventListener('resize', function(){ heat.resize(); });
</script>
</body>
</html>
"""


def build_grid_report(results: pd.DataFrame, out_html: str,
                      title: str = "参数优化网格搜索", subtitle: str = "") -> None:
    """生成自包含 HTML：夏普热力图 + 全组合明细表。"""
    top_ns = sorted(results["top_n"].unique().tolist())
    lookbacks = sorted(results["lookback"].unique().tolist())

    piv = results.groupby(["lookback", "top_n"])["sharpe"].mean().reset_index()
    wide = piv.pivot(index="lookback", columns="top_n", values="sharpe")
    heat = []
    for yi, lb in enumerate(lookbacks):
        for xi, tn in enumerate(top_ns):
            if lb in wide.index and tn in wide.columns:
                v = wide.loc[lb, tn]
                if pd.notna(v):
                    heat.append([xi, yi, round(float(v), 3)])

    ordered = results.sort_values("sharpe", ascending=False)
    columns = [
        {"key": "lookback", "label": "lookback", "fmt": "int"},
        {"key": "top_n", "label": "top_n", "fmt": "int"},
        {"key": "buffer", "label": "buffer", "fmt": "int"},
        {"key": "total_return", "label": "总收益", "fmt": "pct"},
        {"key": "annual_return", "label": "年化", "fmt": "pct"},
        {"key": "sharpe", "label": "夏普", "fmt": "num"},
        {"key": "max_drawdown", "label": "最大回撤", "fmt": "pct"},
        {"key": "calmar", "label": "卡玛", "fmt": "num"},
        {"key": "annual_turnover", "label": "年化换手", "fmt": "pct"},
        {"key": "worst_year_return", "label": "最差年度", "fmt": "pct"},
        {"key": "positive_year_ratio", "label": "正年占比", "fmt": "pct"},
        {"key": "excess_return", "label": "超额收益", "fmt": "pct"},
        {"key": "information_ratio", "label": "信息比率", "fmt": "num"},
    ]
    # 仅保留结果里实际存在的列
    columns = [c for c in columns if c["key"] in results.columns]
    rows = []
    for _, r in ordered.iterrows():
        rows.append({c["key"]: r[c["key"]] for c in columns})

    payload = {
        "top_ns": top_ns, "lookbacks": lookbacks, "heat": heat,
        "columns": columns, "rows": rows,
    }
    html = (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)))
    Path(out_html).write_text(html, encoding="utf-8")
