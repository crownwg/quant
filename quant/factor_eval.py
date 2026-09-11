"""因子有效性检验：IC / IR / 分组收益 / 衰减。

为什么需要它
------------
回测净值为正是**最终结果**，不是**证据**。净值涨了可能是：
  - 因子真的有效（横截面上分数高的票确实跑赢）
  - 运气 / 某段行情的 beta
  - 过拟合出来的参数
三者从一条净值曲线上无法区分。IC 检验直接回答最根本的那个问题：

    这一刻分数排名靠前的股票，未来一段时间是否真的跑赢排名靠后的？

指标口径
--------
  IC   (Information Coefficient) —— 每期的因子值与未来收益的横截面相关系数。
        默认用 **Rank IC**（斯皮尔曼秩相关）：横截面收益有明显的肥尾，
        皮尔逊相关会被少数极端值主导，秩相关稳健得多。
        经验刻度：|IC| 均值 < 0.02 基本无效；0.03~0.05 可用；
        0.05~0.1 已算不错；> 0.1 要警惕数据泄漏或计算错误。
  IR   (Information Ratio)      —— IC 均值 / IC 标准差，衡量因子的**稳定性**。
        比 IC 均值更重要：IC 均值 0.05 但波动 0.3 的因子，本质上不可用。
        经验刻度：IR > 0.5 算合格，> 0.8 优秀。
  t 值                           —— IR × sqrt(期数)，粗判显著性（|t| > 2 约等于 95% 置信）。
  正 IC 占比                     —— 方向上是否稳定（长期在 50% 附近来回摆 = 不可用）。
  分组收益                       —— 按因子分成 N 组，看各组未来收益是否**单调递增**。
        单调性比 IC 更难伪造：IC 只证明「有相关性」，单调分组证明「线性可用」。

⚠️ 关于前视的说明
----------------
本模块用 `prices.shift(-h)` 计算未来收益，天然带前视——这正是因子的定义方式。
因此它**只用于因子诊断，不可用于回测**。回测里的前视已由 backtest.py 的
delay 机制单独处理。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


IC_VERDICT = [
    (0.10, "异常偏强——先排查数据泄漏/复权错误，再考虑是不是真信号"),
    (0.05, "强有效"),
    (0.03, "有效可用"),
    (0.02, "偏弱，需要多因子合成增强"),
    (0.00, "基本无效"),
]


def verdict(mean_abs_ic: float) -> str:
    for th, label in IC_VERDICT:
        if mean_abs_ic >= th:
            return label
    return "基本无效"


def forward_returns(prices: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """未来 horizon 个交易日的收益面板（date×code）。末 horizon 行为 NaN。"""
    return prices.shift(-horizon) / prices - 1.0


def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman",
              min_pairs: int = 10) -> pd.Series:
    """逐期横截面相关系数。

    method="spearman" 时先做横截面秩变换再算皮尔逊相关（等价秩相关）。
    min_pairs 以下的期数置 NaN——有效样本太少的相关系数没有意义。
    """
    a, b = factor.align(fwd, join="inner")
    if method == "spearman":
        a = a.rank(axis=1)
        b = b.rank(axis=1)

    mask = a.notna() & b.notna()
    n = mask.sum(axis=1).astype(float)
    aa = a.where(mask)
    bb = b.where(mask)
    ma = aa.mean(axis=1)
    mb = bb.mean(axis=1)
    da = aa.sub(ma, axis=0)
    db = bb.sub(mb, axis=0)
    cov = (da * db).sum(axis=1) / n
    sa = (da * da).sum(axis=1) / n
    sb = (db * db).sum(axis=1) / n
    denom = np.sqrt(sa * sb)
    ic = cov / denom.replace(0.0, np.nan)
    return ic.where(n >= min_pairs)


def ic_summary(ic: pd.Series) -> dict:
    """把 IC 序列汇总成一组可判断的指标。"""
    s = ic.dropna()
    n = int(len(s))
    if n == 0:
        return {"n": 0, "mean_ic": 0.0, "std_ic": 0.0, "ir": 0.0, "t_stat": 0.0,
                "positive_ratio": 0.0, "abs_mean_ic": 0.0, "verdict": "样本不足"}
    mean = float(s.mean())
    std = float(s.std())
    ir = mean / std if std else 0.0
    return {
        "n": n,
        "mean_ic": mean,
        "std_ic": std,
        "ir": float(ir),
        "t_stat": float(ir * np.sqrt(n)),
        "positive_ratio": float((s > 0).mean()),
        "abs_mean_ic": float(s.abs().mean()),
        "verdict": verdict(abs(mean)),
    }


def ic_decay(factor: pd.DataFrame, prices: pd.DataFrame,
             horizons=(1, 5, 10, 20, 60), method: str = "spearman") -> pd.DataFrame:
    """IC 随持有期衰减：因子预测力能持续多久。

    衰减太快（如只有 1 日 IC 高）说明它只是短期反转/微观结构噪声，
    月度调仓根本吃不到；衰减平缓才适合低频持有。
    """
    rows = []
    for h in horizons:
        ic = ic_series(factor, forward_returns(prices, h), method=method)
        s = ic_summary(ic)
        rows.append({"horizon": int(h), "mean_ic": s["mean_ic"], "std_ic": s["std_ic"],
                     "ir": s["ir"], "positive_ratio": s["positive_ratio"], "n": s["n"]})
    return pd.DataFrame(rows)


def quantile_returns(factor: pd.DataFrame, prices: pd.DataFrame, n_groups: int = 5,
                     horizon: int = 20) -> pd.DataFrame:
    """按因子分 n 组的未来平均收益（组 1 = 因子值最低，组 n = 最高）。

    返回列：group, mean_return, annualized, count。
    单调递增才说明因子可线性使用；若只有一组突出、其余打平，
    多半是少数股票贡献的，实盘复制不了。
    """
    fwd = forward_returns(prices, horizon)
    f, w = factor.align(fwd, join="inner")
    rank = f.rank(axis=1, pct=True)
    grp = np.ceil(rank * n_groups)
    rows = []
    periods_per_year = 252 / max(horizon, 1)
    for g in range(1, n_groups + 1):
        vals = w.where(grp == g)
        stacked = vals.stack()
        stacked = stacked[np.isfinite(stacked)]
        if stacked.empty:
            rows.append({"group": g, "mean_return": 0.0, "annualized": 0.0, "count": 0})
            continue
        mean = float(stacked.mean())
        rows.append({
            "group": g,
            "mean_return": mean,
            "annualized": float((1 + mean) ** periods_per_year - 1) if mean > -1 else -1.0,
            "count": int(len(stacked)),
        })
    return pd.DataFrame(rows)


def long_short_spread(qret: pd.DataFrame) -> float:
    """多空组合的每期平均收益（最高组 − 最低组）。"""
    if qret.empty or len(qret) < 2:
        return 0.0
    return float(qret["mean_return"].iloc[-1] - qret["mean_return"].iloc[0])


def evaluate(name: str, factor: pd.DataFrame, prices: pd.DataFrame,
             horizons=(1, 5, 10, 20, 60), n_groups: int = 5) -> dict:
    """对单个因子做完整诊断，返回可序列化的结果字典。"""
    ic = ic_series(factor, forward_returns(prices, 20))
    summary = ic_summary(ic)
    decay = ic_decay(factor, prices, horizons=horizons)
    qret = quantile_returns(factor, prices, n_groups=n_groups, horizon=20)
    return {
        "name": name,
        "summary": summary,
        "ic_dates": [d.strftime("%Y-%m-%d") for d in ic.index],
        "ic_values": [None if pd.isna(v) else round(float(v), 4) for v in ic.values],
        "cum_ic": [None if pd.isna(v) else round(float(v), 4) for v in ic.fillna(0).cumsum().values],
        "decay": decay.to_dict("records"),
        "quantiles": qret.to_dict("records"),
        "long_short": long_short_spread(qret),
    }


def evaluate_many(factors: dict[str, pd.DataFrame], prices: pd.DataFrame,
                  horizons=(1, 5, 10, 20, 60), n_groups: int = 5) -> list[dict]:
    return [evaluate(n, f, prices, horizons=horizons, n_groups=n_groups)
            for n, f in factors.items()]


# --------------------------------------------------------------- 报告

def print_summary(results: list[dict]) -> None:
    print("\n【因子有效性 · IC 检验】横截面 Rank IC（未来 20 日收益）")
    print(f"  {'因子':<16s}{'IC均值':>9s}{'IC标准差':>10s}{'IR':>8s}"
          f"{'t值':>8s}{'正IC占比':>10s}  评价")
    for r in results:
        s = r["summary"]
        if s["n"] == 0:
            print(f"  {r['name']:<16s}{'—':>9s}{'—':>10s}{'—':>8s}{'—':>8s}{'—':>10s}  样本不足")
            continue
        print(f"  {r['name']:<16s}{s['mean_ic']:>9.4f}{s['std_ic']:>10.4f}"
              f"{s['ir']:>8.3f}{s['t_stat']:>8.2f}{s['positive_ratio']:>9.1%}  {s['verdict']}")
    print("  多空组合（最高组−最低组，每 20 日）：")
    for r in results:
        print(f"    {r['name']:<16s}{r['long_short']:>8.2%}")


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  body { margin:0; padding:24px; background:#f5f5f5; color:#1f1f1f;
         font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  h1 { margin:0 0 6px; font-size:22px; }
  .sub { color:#666; margin-bottom:20px; font-size:13px; }
  .group { background:#fff; border:1px solid #e8e8e8; border-radius:12px;
           padding:16px 20px; margin-bottom:16px; }
  h2 { margin:0 0 12px; font-size:16px; }
  .chart { width:100%; height:360px; }
  .note { color:#888; font-size:12px; margin-top:8px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { padding:6px 10px; border-bottom:1px solid #f0f0f0; text-align:right; }
  th:first-child,td:first-child { text-align:left; }
  th { background:#fafafa; font-weight:600; }
  .pos { color:#cf1322; } .neg { color:#389e0d; }
  .tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px;
         background:#f0f5ff; color:#2f54eb; }
</style></head><body>
<h1>__TITLE__</h1><div class="sub">__SUBTITLE__</div>

<div class="group">
  <h2>汇总指标</h2>
  <table id="summary"><thead><tr>
    <th>因子</th><th>IC均值</th><th>IC标准差</th><th>IR</th><th>t值</th>
    <th>正IC占比</th><th>期数</th><th>多空(每期)</th><th>评价</th>
  </tr></thead><tbody></tbody></table>
  <div class="note">IC 均值是方向性，IR 是稳定性。<b>IR 比 IC 均值更该被重视</b>——
    一个 IC 均值 0.05、标准差 0.3 的因子在实盘里等于没有。</div>
</div>

<div class="group">
  <h2>累计 IC 曲线</h2>
  <div id="cumic" class="chart"></div>
  <div class="note">累计 IC 单调上行 = 因子持续有效；走平或来回震荡 = 该段行情里因子失效。
    斜率突然变陡往往意味着数据问题，而不是因子突然变强。</div>
</div>

<div class="group">
  <h2>IC 衰减（不同持有期）</h2>
  <div id="decay" class="chart"></div>
  <div class="note">月度调仓需要 IC 在 20 日仍有留存；若只在 1 日高、之后迅速归零，
    说明这只是短期反转/微观结构噪声，低频策略吃不到。</div>
</div>

<div class="group">
  <h2>分组收益（按因子分 5 组，组 5 = 分数最高）</h2>
  <div id="quant" class="chart"></div>
  <div class="note"><b>单调性比 IC 更难伪造</b>。只有最高组一枝独秀、其余打平时，
    收益多半由少数个股贡献，实盘无法复制。</div>
</div>

<script>
const DATA = __DATA__;
const COLORS = ['#722ed1','#1677ff','#52c41a','#fa8c16','#eb2f96','#13c2c2','#fadb14','#9254de'];
const axisCommon = { axisLine:{lineStyle:{color:'#e6e6e6'}}, axisLabel:{color:'#6b7280'} };

// 汇总表
const tb = document.querySelector('#summary tbody');
DATA.results.forEach(r => {
  const s = r.summary;
  const tr = document.createElement('tr');
  if (!s.n) {
    tr.innerHTML = `<td><b>${r.name}</b></td><td colspan="8">样本不足</td>`;
  } else {
    const cls = v => v >= 0 ? 'pos' : 'neg';
    tr.innerHTML = `<td><b>${r.name}</b></td>
      <td class="${cls(s.mean_ic)}">${s.mean_ic.toFixed(4)}</td>
      <td>${s.std_ic.toFixed(4)}</td>
      <td>${s.ir.toFixed(3)}</td>
      <td>${s.t_stat.toFixed(2)}</td>
      <td>${(s.positive_ratio*100).toFixed(1)}%</td>
      <td>${s.n}</td>
      <td class="${cls(r.long_short)}">${(r.long_short*100).toFixed(2)}%</td>
      <td><span class="tag">${s.verdict}</span></td>`;
  }
  tb.appendChild(tr);
});

// 累计 IC
const cum = echarts.init(document.getElementById('cumic'));
let commonDates = null;
DATA.results.forEach(r => {
  commonDates = commonDates === null ? r.ic_dates
    : commonDates.filter(d => r.ic_dates.indexOf(d) >= 0);
});
cum.setOption({
  tooltip:{ trigger:'axis' },
  legend:{ top:0, data:DATA.results.map(r=>r.name) },
  grid:{ left:60, right:30, top:40, bottom:50 },
  xAxis:{ type:'category', data:commonDates, axisLabel:{ fontSize:10, color:'#6b7280' } },
  yAxis:{ type:'value', name:'累计IC', ...axisCommon },
  series: DATA.results.map((r,i) => {
    const map = {}; r.ic_dates.forEach((d,j)=>{ map[d]=r.cum_ic[j]; });
    return { name:r.name, type:'line', showSymbol:false, smooth:true,
             lineStyle:{ width:2, color:COLORS[i%COLORS.length] },
             itemStyle:{ color:COLORS[i%COLORS.length] },
             data: commonDates.map(d => map[d] === undefined ? null : map[d]) };
  })
});

// IC 衰减
const dec = echarts.init(document.getElementById('decay'));
const horizons = DATA.results[0] ? DATA.results[0].decay.map(x => x.horizon+'日') : [];
dec.setOption({
  tooltip:{ trigger:'axis' },
  legend:{ top:0, data:DATA.results.map(r=>r.name) },
  grid:{ left:60, right:30, top:40, bottom:50 },
  xAxis:{ type:'category', data:horizons, ...axisCommon },
  yAxis:{ type:'value', name:'IC均值', ...axisCommon },
  series: DATA.results.map((r,i) => ({
    name:r.name, type:'line', showSymbol:true, smooth:false,
    lineStyle:{ width:2, color:COLORS[i%COLORS.length] },
    itemStyle:{ color:COLORS[i%COLORS.length] },
    data: r.decay.map(x => +(x.mean_ic).toFixed(4))
  }))
});

// 分组收益
const qua = echarts.init(document.getElementById('quant'));
const groupLabels = DATA.results[0] ? DATA.results[0].quantiles.map(x => '组'+x.group) : [];
qua.setOption({
  tooltip:{ trigger:'axis', axisPointer:{type:'shadow'},
            formatter: p => p.map(x=>`${x.marker}${x.seriesName} ${x.name}: ${x.value.toFixed(2)}%`).join('<br>') },
  legend:{ top:0, data:DATA.results.map(r=>r.name) },
  grid:{ left:60, right:30, top:40, bottom:50 },
  xAxis:{ type:'category', data:groupLabels, ...axisCommon },
  yAxis:{ type:'value', axisLabel:{ formatter:'{value}%' }, ...axisCommon },
  series: DATA.results.map((r,i) => ({
    name:r.name, type:'bar', barMaxWidth:36,
    itemStyle:{ color:COLORS[i%COLORS.length] },
    data: r.quantiles.map(x => +(x.mean_return*100).toFixed(3))
  }))
});

window.addEventListener('resize', () => { cum.resize(); dec.resize(); qua.resize(); });
</script></body></html>
"""


def build_ic_report(results: list[dict], out_html: str,
                    title: str = "因子有效性检验", subtitle: str = "") -> None:
    html = (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__DATA__", json.dumps({"results": results}, ensure_ascii=False)))
    Path(out_html).write_text(html, encoding="utf-8")
