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

import numpy as np
import pandas as pd

from .backtest import run, cost_kwargs
from .strategy import factor_weights, weights_from_args
from . import factors, rolling


def parse_ints(text: str) -> list[int]:
    """把 '60,90,120' 解析成 [60, 90, 120]。"""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def grid_search(prices_all: pd.DataFrame, open_all: pd.DataFrame,
                can_buy, can_sell, args,
                lookbacks: list[int], top_ns: list[int], buffers: list[int],
                start_ts: pd.Timestamp, benchmark_curve: pd.Series | None = None,
                weight_cap: pd.DataFrame | None = None) -> pd.DataFrame:
    """对每组合回测，返回结果 DataFrame（一行一组合）。

    ⚠️ 这是**全样本网格搜索**：在整段样本上挑最优参数，等于「事后选参」，
    结果天然偏乐观。要判断参数是否真的稳健，请用 walk_forward_search。
    """
    rows = []
    combos = list(itertools.product(lookbacks, top_ns, buffers))
    total = len(combos)
    cost = cost_kwargs(args)
    for i, (lb, tn, buf) in enumerate(combos, 1):
        score = factors.momentum(prices_all, lb, args.skip_recent)
        weights_all = weights_from_args(score, args, can_buy=can_buy, can_sell=can_sell,
                                        prices=prices_all, weight_cap=weight_cap,
                                        top_n=tn, buffer=buf)
        keep = weights_all.index >= start_ts
        prices = prices_all.loc[keep]
        weights = weights_all.loc[keep]
        op = open_all.loc[keep] if args.use_open else None
        equity, m, detail = run(prices, weights, open_prices=op, **cost)
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


# ============================================================ 样本外验证
#
# 为什么必须做 walk-forward
# ------------------------
# 全样本网格搜索有两个致命问题：
#   1. 事后选参：你在整段历史上挑出最猛的那组参数，等于让参数「看见」了未来；
#   2. 多重比较：试的组合越多，「碰巧最优」的那个就越漂亮——这是纯粹的噪声。
# 表现为：训练期夏普 1.2，实盘立刻掉到 0.3。
#
# 做法：把样本切成若干「训练窗 + 紧邻测试窗」，在训练窗内挑参、在测试窗里
# **完全不改参**地跑一遍，把各测试窗的净值接起来。这条拼接曲线才是这套方法
# 真实能拿到的成绩。同时给出两个对照组：
#   - 全样本最优固定参数（事后选参）：它在测试窗上的表现 ≈ 上界幻觉
#   - 默认参数（不调参）           ：不调参反而常常赢过「精心调参」
# 如果 walk-forward 明显跑输全样本最优，说明那组「最优参数」是拟合出来的。


def _equity_metrics(equity: pd.Series, periods_per_year: int = 252) -> dict:
    """从净值曲线算核心指标（与 backtest.run 口径一致，但不需要成本明细）。"""
    if equity is None or len(equity) < 2:
        return {"total_return": 0.0, "annual_return": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "annual_volatility": 0.0}
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ann = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    vol = float(rets.std() * np.sqrt(periods_per_year)) if len(rets) else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(periods_per_year)) if len(rets) and rets.std() else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    return {"total_return": total, "annual_return": ann, "sharpe": sharpe,
            "max_drawdown": dd, "annual_volatility": vol,
            "trading_days": int(len(equity))}


def walk_forward_folds(n: int, start_pos: int, train_n: int, test_n: int) -> list[tuple[int, int, int]]:
    """生成 (训练起点, 训练终点=测试起点, 测试终点) 三元组，测试窗之间不重叠。

    滚动方式：每次向后移动一个测试窗宽度，保证每个测试窗只用它之前的数据选参。
    """
    folds = []
    s = start_pos
    while s + train_n + test_n <= n:
        folds.append((s, s + train_n, s + train_n + test_n))
        s += test_n
    return folds


def walk_forward_search(prices_all: pd.DataFrame, open_all: pd.DataFrame,
                        can_buy, can_sell, args,
                        lookbacks: list[int], top_ns: list[int], buffers: list[int],
                        start_ts: pd.Timestamp, train_years: float = 2.0,
                        test_years: float = 0.5, weight_cap: pd.DataFrame | None = None,
                        select_metric: str = "sharpe",
                        periods_per_year: int = 252) -> dict:
    """滚动训练/测试的样本外验证。

    返回 dict：
      folds      每折明细 DataFrame（含训练期最优参数与其样本内/样本外表现）
      curves     {'walk_forward'|'full_sample_best'|'default': 测试期拼接净值}
      summary    三者的汇总指标 DataFrame
      chosen     每折选中的参数
    """
    idx_all = prices_all.index
    n_all = len(idx_all)
    train_n = int(round(train_years * periods_per_year))
    test_n = int(round(test_years * periods_per_year))
    if train_n < 120 or test_n < 20:
        raise ValueError(f"训练窗 {train_n} 日 / 测试窗 {test_n} 日过短，无法做 walk-forward")

    start_pos = int(np.searchsorted(idx_all.values, np.datetime64(pd.Timestamp(start_ts))))
    folds = walk_forward_folds(n_all, start_pos, train_n, test_n)
    if len(folds) < 2:
        raise ValueError(
            f"样本不足以做 walk-forward：区间内仅 {len(folds)} 折。"
            f"请拉长 --start/--end，或缩小 --fw-train / --fw-test")

    combos = list(itertools.product(lookbacks, top_ns, buffers))
    score_cache = {lb: factors.momentum(prices_all, lb, getattr(args, "skip_recent", 0))
                   for lb in lookbacks}
    cost = cost_kwargs(args)

    def run_combo(combo, end_pos, lo, hi):
        """在 [lo, hi) 上跑 combo，权重只用 < hi 的数据计算（无前视）。"""
        lb, tn, buf = combo
        sc = score_cache[lb].iloc[:end_pos]
        w = weights_from_args(
            sc, args,
            can_buy=(can_buy.iloc[:end_pos] if can_buy is not None else None),
            can_sell=(can_sell.iloc[:end_pos] if can_sell is not None else None),
            prices=prices_all.iloc[:end_pos],
            weight_cap=(weight_cap.iloc[:end_pos] if weight_cap is not None else None),
            top_n=tn, buffer=buf)
        w = w.iloc[lo:hi]
        p = prices_all.iloc[lo:hi]
        o = open_all.iloc[lo:hi] if getattr(args, "use_open", False) else None
        eq, m, _ = run(p, w, open_prices=o, **cost)
        return eq, m

    default_combo = (int(getattr(args, "lookback", 120)),
                     int(getattr(args, "top_n", 10)),
                     int(getattr(args, "buffer", 0)))

    # ---- 对照组 A：全样本事后选参 ----
    print(f"  [对照组] 全样本网格 {len(combos)} 组，用于对比「事后选参」的幻觉")
    full_rows = []
    for combo in combos:
        _, m = run_combo(combo, n_all, start_pos, n_all)
        full_rows.append({"combo": combo, select_metric: m[select_metric],
                          "total_return": m["total_return"], "sharpe": m["sharpe"],
                          "max_drawdown": m["max_drawdown"]})
    full_df = pd.DataFrame(full_rows)
    full_best_combo = tuple(full_df.sort_values(select_metric, ascending=False)["combo"].iloc[0])

    # ---- 逐折：训练窗选参 → 测试窗纯验证 ----
    print(f"\n【Walk-Forward】{len(folds)} 折，训练 {train_n} 日 / 测试 {test_n} 日"
          f"（选参依据：训练期 {select_metric}）")
    fold_rows = []
    curves = {"walk_forward": [], "full_sample_best": [], "default": []}
    for fi, (s, se, te) in enumerate(folds, 1):
        best = None
        for combo in combos:
            _, m_tr = run_combo(combo, se, s, se)
            if best is None or m_tr[select_metric] > best[1][select_metric]:
                best = (combo, m_tr)
        wf_combo, wf_train_m = best

        _, m_wf = run_combo(wf_combo, te, se, te)
        eq_fb, m_fb = run_combo(full_best_combo, te, se, te)
        eq_df, m_df = run_combo(default_combo, te, se, te)

        # 记录测试期的净值（后续按年化收益拼接成连续曲线）
        curves["walk_forward"].append((se, te, m_wf["sharpe"], m_wf["total_return"], m_wf["max_drawdown"]))
        curves["full_sample_best"].append((se, te, m_fb["sharpe"], m_fb["total_return"], m_fb["max_drawdown"]))
        curves["default"].append((se, te, m_df["sharpe"], m_df["total_return"], m_df["max_drawdown"]))

        fold_rows.append({
            "fold": fi,
            "train_start": idx_all[s].date(), "train_end": idx_all[se - 1].date(),
            "test_start": idx_all[se].date(), "test_end": idx_all[te - 1].date(),
            "lookback": wf_combo[0], "top_n": wf_combo[1], "buffer": wf_combo[2],
            "train_sharpe": round(float(wf_train_m["sharpe"]), 3),
            "train_return": round(float(wf_train_m["total_return"]), 4),
            "test_sharpe": round(float(m_wf["sharpe"]), 3),
            "test_return": round(float(m_wf["total_return"]), 4),
            "test_max_drawdown": round(float(m_wf["max_drawdown"]), 4),
            "decay": round(float(m_wf["sharpe"] - wf_train_m["sharpe"]), 3),
        })
        print(f"  第{fi}折 {idx_all[s].date()}~{idx_all[se-1].date()} 选参"
              f" lb={wf_combo[0]} n={wf_combo[1]} buf={wf_combo[2]}"
              f" | 样本内夏普 {wf_train_m['sharpe']:.2f} → 样本外 {m_wf['sharpe']:.2f}"
              f"（衰减 {fold_rows[-1]['decay']:+.2f}）"
              f" | 样本外收益 {m_wf['total_return']:.1%}", flush=True)

    folds_df = pd.DataFrame(fold_rows)

    # ---- 拼接测试期净值（按各段收益连乘）----
    def chain(records):
        eq = [1.0]
        for _, _, _, ret, _ in records:
            eq.append(eq[-1] * (1 + ret))
        return eq

    summary_rows = []
    for key, label in (("walk_forward", "Walk-Forward 自适应参数"),
                       ("full_sample_best", "全样本最优固定参数（事后选参）"),
                       ("default", "默认参数（不调参）")):
        recs = curves[key]
        chain_vals = chain(recs)
        total = float(chain_vals[-1] - 1)
        n_years = sum((idx_all[b - 1] - idx_all[a]).days for a, b, *_ in recs) / 365.25
        n_years = max(n_years, 1 / 365.25)
        ann = float((1 + total) ** (1 / n_years) - 1) if total > -1 else -1.0
        # 样本外夏普：用各折夏普的加权平均（近似，避免拼接空隙造成的失真）
        days = [len(idx_all[a:b]) for a, b, *_ in recs]
        w_sharpe = float(np.average([r[2] for r in recs], weights=days)) if days else 0.0
        worst_dd = float(min(r[4] for r in recs)) if recs else 0.0
        summary_rows.append({
            "strategy": label,
            "total_return": total,
            "annual_return": ann,
            "avg_sharpe": w_sharpe,
            "worst_fold_drawdown": worst_dd,
            "positive_folds": float(np.mean([r[3] > 0 for r in recs])) if recs else 0.0,
            "folds": len(recs),
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "folds": folds_df,
        "summary": summary_df,
        "curves": curves,
        "full_best_combo": full_best_combo,
        "default_combo": default_combo,
        "train_n": train_n,
        "test_n": test_n,
        "train_years": train_years,
        "test_years": test_years,
        "select_metric": select_metric,
    }


_WF_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  :root { --bg:#fff; --surface:#fafafa; --border:#e6e6e6; --text:#1f2329; --muted:#6b7280; --accent:#2f6df0; }
  * { box-sizing:border-box; }
  body { font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; margin:0;
         background:var(--bg); color:var(--text); line-height:1.6; }
  .wrap { max-width:1080px; margin:0 auto; padding:32px 24px 64px; }
  h1 { font-size:22px; font-weight:600; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:24px; }
  .group { margin-bottom:28px; }
  .group h2 { font-size:14px; font-weight:600; color:var(--muted); margin:0 0 10px;
              border-left:3px solid var(--accent); padding-left:8px; }
  .chart { width:100%; height:400px; background:var(--surface); border:1px solid var(--border);
           border-radius:12px; margin-bottom:14px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { border:1px solid var(--border); padding:7px 9px; text-align:right; }
  th { background:var(--surface); color:var(--muted); font-weight:600; }
  td:first-child,th:first-child { text-align:left; }
  tr:nth-child(even) td { background:#fcfcfc; }
  .pos { color:#cf1322; } .neg { color:#389e0d; }
  .note { color:var(--muted); font-size:12px; margin-top:8px; }
  .card { display:inline-block; border:1px solid var(--border); border-radius:10px;
          padding:10px 16px; margin:0 10px 10px 0; background:var(--surface); }
  .card b { display:block; font-size:18px; }
  .card span { color:var(--muted); font-size:12px; }
</style></head><body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>

  <div class="group">
    <h2>核心结论</h2>
    <div id="cards"></div>
    <div class="note" id="verdict"></div>
  </div>

  <div class="group">
    <h2>样本外测试期拼接净值（各折连乘）</h2>
    <div class="chart" id="curves"></div>
    <div class="note">只看<b>测试期</b>——训练期不参与画图。
      若「全样本最优固定参数」明显高于「Walk-Forward 自适应」，
      说明那组最优参数吃的是样本内的运气，实盘拿不到。</div>
  </div>

  <div class="group">
    <h2>过拟合诊断：每折「训练期夏普 vs 测试期夏普」</h2>
    <div class="chart" id="decay"></div>
    <div class="note">训练期夏普普遍高、测试期大幅回落（落差越大过拟合越严重）。
      落差长期为正说明选参过程本身在拟合噪声。</div>
  </div>

  <div class="group">
    <h2>汇总对比</h2>
    <table id="summary"></table>
    <div class="note">「正收益折占比」= 测试窗里赚钱的比例，衡量方法在时间上的稳定性。</div>
  </div>

  <div class="group">
    <h2>逐折明细</h2>
    <table id="folds"></table>
    <div class="note">decay = 测试期夏普 − 训练期夏普。</div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const axisCommon = { axisLine:{lineStyle:{color:'#e6e6e6'}}, axisLabel:{color:'#6b7280'} };

// 卡片
const cards = document.getElementById('cards');
const wf = DATA.summary.find(s => s.key === 'walk_forward');
const fb = DATA.summary.find(s => s.key === 'full_sample_best');
const df = DATA.summary.find(s => s.key === 'default');
const card = (label, v, hint) => `<div class="card"><span>${label}</span><b>${v}</b>` +
  (hint ? `<span>${hint}</span>` : '') + '</div>';
cards.innerHTML =
  card('Walk-Forward 样本外总收益', (wf.total_return*100).toFixed(1)+'%',
       '年化 '+(wf.annual_return*100).toFixed(1)+'% · 平均夏普 '+wf.avg_sharpe.toFixed(2)) +
  card('全样本最优（事后选参）', (fb.total_return*100).toFixed(1)+'%',
       '同期对照，高于 WF 的部分是幻觉') +
  card('默认参数（不调参）', (df.total_return*100).toFixed(1)+'%',
       '不调参的基准线') +
  card('正收益折占比', (wf.positive_folds*100).toFixed(0)+'%', wf.folds+' 折测试窗');

const gap = fb.total_return - wf.total_return;
document.getElementById('verdict').innerHTML = gap > 0.05
  ? `<b>⚠️ 事后选参高估了 ${(gap*100).toFixed(1)} 个百分点</b>——全样本最优参数在样本外明显跑输自适应选参，说明调参过程在拟合噪声。`
  : `全样本最优与自适应选参差距 ${(gap*100).toFixed(1)} 个百分点，参数对样本外的影响有限。`;

// 净值曲线
const cv = echarts.init(document.getElementById('curves'));
const names = {walk_forward:'Walk-Forward 自适应', full_sample_best:'全样本最优固定', default:'默认参数'};
const colors = {walk_forward:'#cf1322', full_sample_best:'#fadb14', default:'#8c8c8c'};
cv.setOption({
  tooltip:{ trigger:'axis' },
  legend:{ top:0, data:Object.keys(names).map(k=>names[k]) },
  grid:{ left:60, right:30, top:40, bottom:50 },
  xAxis:{ type:'category', data:DATA.eq_x, axisLabel:{ fontSize:10, color:'#6b7280' }, name:'交易日（仅测试窗）' },
  yAxis:{ type:'value', name:'净值', ...axisCommon },
  series: Object.keys(names).map(k => ({
    name:names[k], type:'line', showSymbol:false, smooth:true,
    lineStyle:{ width:2, color:colors[k] }, itemStyle:{ color:colors[k] },
    data: DATA.eq[k]
  }))
});

// 过拟合诊断
const dc = echarts.init(document.getElementById('decay'));
dc.setOption({
  tooltip:{ trigger:'axis', axisPointer:{type:'shadow'} },
  legend:{ top:0, data:['训练期夏普','测试期夏普'] },
  grid:{ left:60, right:30, top:40, bottom:50 },
  xAxis:{ type:'category', data:DATA.folds.map(f=>'第'+f.fold+'折'), ...axisCommon },
  yAxis:{ type:'value', name:'夏普', ...axisCommon },
  series:[
    { name:'训练期夏普', type:'bar', barMaxWidth:28, itemStyle:{ color:'#91caff' },
      data: DATA.folds.map(f=>f.train_sharpe) },
    { name:'测试期夏普', type:'bar', barMaxWidth:28, itemStyle:{ color:'#cf1322' },
      data: DATA.folds.map(f=>f.test_sharpe) }
  ]
});

// 汇总表
const pct = v => `<td class="${v>=0?'pos':'neg'}">${(v*100).toFixed(2)}%</td>`;
let sh = '<tr><th>参数策略</th><th>样本外总收益</th><th>年化</th><th>平均夏普</th><th>最差折回撤</th><th>正收益折占比</th><th>折数</th></tr>';
DATA.summary.forEach(s => {
  sh += `<tr><td>${s.label}</td>${pct(s.total_return)}${pct(s.annual_return)}
    <td>${s.avg_sharpe.toFixed(3)}</td>${pct(s.worst_fold_drawdown)}
    <td>${(s.positive_folds*100).toFixed(0)}%</td><td>${s.folds}</td></tr>`;
});
document.getElementById('summary').innerHTML = sh;

// 逐折表
let fh = '<tr><th>折</th><th>训练期</th><th>测试期</th><th>选中参数</th>' +
         '<th>训练夏普</th><th>测试夏普</th><th>衰减</th><th>测试收益</th><th>测试回撤</th></tr>';
DATA.folds.forEach(f => {
  fh += `<tr><td>${f.fold}</td><td>${f.train_start}~${f.train_end}</td>
    <td>${f.test_start}~${f.test_end}</td>
    <td>lb=${f.lookback} n=${f.top_n} buf=${f.buffer}</td>
    <td>${f.train_sharpe.toFixed(2)}</td><td>${f.test_sharpe.toFixed(2)}</td>
    <td class="${f.decay>=0?'pos':'neg'}">${f.decay>=0?'+':''}${f.decay.toFixed(2)}</td>
    ${pct(f.test_return)}${pct(f.test_max_drawdown)}</tr>`;
});
document.getElementById('folds').innerHTML = fh;

window.addEventListener('resize', () => { cv.resize(); dc.resize(); });
</script></body></html>
"""


def build_wf_report(res: dict, out_html: str, title: str = "样本外验证（Walk-Forward）",
                    subtitle: str = "") -> None:
    """生成 walk-forward 报告：拼接净值 + 过拟合诊断 + 汇总对比 + 逐折明细。"""
    folds = res["folds"]
    curves = res["curves"]

    # 逐折明细：日期字段转成字符串，否则 json 序列化会报 TypeError
    fold_records = []
    for _, r in folds.iterrows():
        rec = {}
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                rec[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
            else:
                rec[k] = v
        fold_records.append(rec)

    # 拼接净值的横轴：按折数展开（每折天数不同，这里用折内序号表示）
    eq = {}
    max_len = 0
    for key in ("walk_forward", "full_sample_best", "default"):
        vals, cur = [], 1.0
        for _, _, _, ret, _ in curves[key]:
            cur *= (1 + ret)
            vals.append(round(cur, 4))
        eq[key] = vals
        max_len = max(max_len, len(vals))
    for key in eq:
        while len(eq[key]) < max_len:
            eq[key].append(eq[key][-1])

    summary_payload = []
    for key, row in zip(("walk_forward", "full_sample_best", "default"),
                        res["summary"].to_dict("records")):
        summary_payload.append({
            "key": key,
            "label": str(row["strategy"]),
            "total_return": float(row["total_return"]),
            "annual_return": float(row["annual_return"]),
            "avg_sharpe": float(row["avg_sharpe"]),
            "worst_fold_drawdown": float(row["worst_fold_drawdown"]),
            "positive_folds": float(row["positive_folds"]),
            "folds": int(row["folds"]),
        })

    payload = {
        "folds": fold_records,
        "summary": summary_payload,
        "eq_x": [f"第{i+1}折" for i in range(max_len)],
        "eq": eq,
    }

    html = (_WF_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)))
    Path(out_html).write_text(html, encoding="utf-8")


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
