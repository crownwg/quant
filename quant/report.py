"""生成 ECharts HTML 回测报告。

build_report 接收一个已经整理好的数据字典，输出一个自包含的 .html 文件
（内嵌 ECharts CDN、净值对比图、回撤图、指标卡片、调仓摘要）。
所有格式化（标签、百分比、分组）都在调用方完成，本模块只负责渲染。
"""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  :root {
    --bg: #ffffff; --surface: #fafafa; --border: #e6e6e6;
    --text: #1f2329; --muted: #6b7280; --accent: #2f6df0;
    --up: #d4380d; --down: #389e0d;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    margin: 0; background: var(--bg); color: var(--text); line-height: 1.6; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin-bottom: 28px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
  .card .k { color: var(--muted); font-size: 12px; }
  .card .v { font-size: 18px; font-weight: 600; margin-top: 4px; }
  .card.good .v { color: var(--down); }
  .card.bad .v { color: var(--up); }
  .group { margin-bottom: 24px; }
  .group h2 { font-size: 14px; font-weight: 600; color: var(--muted); margin: 0 0 10px;
    border-left: 3px solid var(--accent); padding-left: 8px; }
  .chart { width: 100%; height: 380px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px; margin-bottom: 24px; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
  table.attr-table { width: 100%; border-collapse: collapse; margin-bottom: 8px; font-size: 13px; }
  table.attr-table th, table.attr-table td { border: 1px solid var(--border); padding: 8px 10px; text-align: right; }
  table.attr-table th { background: var(--surface); color: var(--muted); font-weight: 600; }
  table.attr-table td:first-child, table.attr-table th:first-child { text-align: left; }
  .pos { color: var(--down); font-weight: 600; }
  .neg { color: var(--up); font-weight: 600; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>

  <div class="group">
    <h2>回测配置</h2>
    <div class="cards" id="config-cards"></div>
  </div>

  <div class="group">
    <h2>评价指标</h2>
    <div id="metric-groups"></div>
  </div>

  <div class="group">
    <h2>净值曲线（策略 vs 基准）</h2>
    <div class="chart" id="equity-chart"></div>
  </div>

  <div class="group">
    <h2>回撤</h2>
    <div class="chart" id="dd-chart"></div>
    <div class="note">回撤 = 净值 / 历史最高点 - 1，负值代表从高点回落的幅度。</div>
  </div>

  <div class="group" id="rolling-group">
    <h2>滚动 / walk-forward 样本外净值</h2>
    <div class="chart" id="rolling-chart"></div>
    <div class="note">把样本切成若干不重叠时段，每段独立回测后拼接。曲线持续创新高，说明策略不只依赖某一两段行情。</div>
  </div>

  <div class="group" id="annual-group">
    <h2>分年度表现</h2>
    <div class="chart" id="annual-chart"></div>
    <div class="note">按自然年切分，逐年独立计算收益与夏普。</div>
  </div>

  <div class="group" id="attribution-group">
    <h2>风险归因（按市场状态拆分）</h2>
    <div class="chart" id="attr-chart"></div>
    <div class="note">市场状态由基准指数中期趋势划分：过去 N 日涨幅 ≥ +band 为牛市、≤ -band 为熊市，其余为震荡。看策略靠哪类行情吃饭。</div>
    <div class="chart" id="attr-contrib-chart"></div>
    <div class="note">收益贡献度按「对数分解」，各状态之和恰好为 100%（复利不可简单相加，故不用算术占比）。</div>
    <table class="attr-table" id="attr-table"></table>
  </div>

  <div class="group" id="plan-group">
    <h2>调仓清单摘要</h2>
    <div class="cards" id="plan-cards"></div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;

const cfg = document.getElementById('config-cards');
(DATA.config || []).forEach(function (row) {
  const d = document.createElement('div'); d.className = 'card';
  d.innerHTML = '<div class="k">' + row[0] + '</div><div class="v" style="font-size:14px">' + row[1] + '</div>';
  cfg.appendChild(d);
});

const mg = document.getElementById('metric-groups');
(DATA.metrics_groups || []).forEach(function (g) {
  const h = document.createElement('h2'); h.textContent = g.title; mg.appendChild(h);
  const wrap = document.createElement('div'); wrap.className = 'cards';
  g.items.forEach(function (row) {
    const v = row[1];
    const cls = (v.indexOf('+') === 0) ? 'good' : ((v.indexOf('-') === 0 && v.indexOf('%') >= 0) ? 'bad' : '');
    const d = document.createElement('div'); d.className = 'card ' + cls;
    d.innerHTML = '<div class="k">' + row[0] + '</div><div class="v">' + v + '</div>';
    wrap.appendChild(d);
  });
  mg.appendChild(wrap);
});

const pg = document.getElementById('plan-group');
const pc = document.getElementById('plan-cards');
if (DATA.plan && Object.keys(DATA.plan).length) {
  Object.entries(DATA.plan).forEach(function (e) {
    const d = document.createElement('div'); d.className = 'card';
    d.innerHTML = '<div class="k">' + e[0] + '</div><div class="v" style="font-size:14px">' + e[1] + '</div>';
    pc.appendChild(d);
  });
} else { pg.style.display = 'none'; }

const axisCommon = { axisLine: { lineStyle: { color: '#e6e6e6' } }, axisLabel: { color: '#6b7280' } };
const grid = { left: 56, right: 24, top: 30, bottom: 40 };

const ec = echarts.init(document.getElementById('equity-chart'));
ec.setOption({
  tooltip: { trigger: 'axis' },
  legend: { data: DATA.benchmark ? ['策略', '基准'] : ['策略'], top: 0 },
  grid: grid,
  xAxis: { type: 'category', data: DATA.dates, axisLine: axisCommon.axisLine, axisLabel: axisCommon.axisLabel },
  yAxis: { type: 'value', axisLine: axisCommon.axisLine, axisLabel: axisCommon.axisLabel },
  series: [
    { name: '策略', type: 'line', showSymbol: false, data: DATA.equity,
      lineStyle: { width: 2, color: '#2f6df0' }, itemStyle: { color: '#2f6df0' } }
  ].concat(DATA.benchmark ? [{ name: '基准', type: 'line', showSymbol: false, data: DATA.benchmark,
      lineStyle: { width: 2, color: '#9aa0a6', type: 'dashed' }, itemStyle: { color: '#9aa0a6' } }] : [])
});

const dd = echarts.init(document.getElementById('dd-chart'));
dd.setOption({
  tooltip: { trigger: 'axis', valueFormatter: function (v) { return (v * 100).toFixed(2) + '%'; } },
  legend: { data: DATA.bench_dd ? ['策略回撤', '基准回撤'] : ['策略回撤'], top: 0 },
  grid: grid,
  xAxis: { type: 'category', data: DATA.dates, axisLine: axisCommon.axisLine, axisLabel: axisCommon.axisLabel },
  yAxis: { type: 'value', axisLine: axisCommon.axisLine,
    axisLabel: { color: '#6b7280', formatter: function (v) { return (v * 100).toFixed(0) + '%'; } } },
  series: [
    { name: '策略回撤', type: 'line', showSymbol: false, data: DATA.drawdown,
      areaStyle: { color: 'rgba(212,56,13,0.12)' }, lineStyle: { width: 1, color: '#d4380d' }, itemStyle: { color: '#d4380d' } }
  ].concat(DATA.bench_dd ? [{ name: '基准回撤', type: 'line', showSymbol: false, data: DATA.bench_dd,
      lineStyle: { width: 1, color: '#9aa0a6', type: 'dashed' }, itemStyle: { color: '#9aa0a6' } }] : [])
});

// 滚动回测净值
var rc = null;
const rg = document.getElementById('rolling-group');
if (DATA.rolling_equity && DATA.rolling_equity.length) {
  rc = echarts.init(document.getElementById('rolling-chart'));
  rc.setOption({
    tooltip: { trigger: 'axis' },
    grid: grid,
    xAxis: { type: 'category', data: DATA.rolling_dates, axisLine: axisCommon.axisLine, axisLabel: axisCommon.axisLabel },
    yAxis: { type: 'value', axisLine: axisCommon.axisLine, axisLabel: axisCommon.axisLabel },
    series: [{ name: '样本外净值', type: 'line', showSymbol: false, data: DATA.rolling_equity,
      lineStyle: { width: 2, color: '#722ed1' }, itemStyle: { color: '#722ed1' } }]
  });
} else { rg.style.display = 'none'; }

// 分年度收益柱状图
var ac = null;
const ag = document.getElementById('annual-group');
if (DATA.annual && DATA.annual.length) {
  ac = echarts.init(document.getElementById('annual-chart'));
  ac.setOption({
    tooltip: { trigger: 'axis', valueFormatter: function (v) { return (v * 100).toFixed(2) + '%'; } },
    grid: grid,
    xAxis: { type: 'category', data: DATA.annual.map(function (r) { return r.year; }),
      axisLine: axisCommon.axisLine, axisLabel: axisCommon.axisLabel },
    yAxis: { type: 'value', axisLine: axisCommon.axisLine,
      axisLabel: { color: '#6b7280', formatter: function (v) { return (v * 100).toFixed(0) + '%'; } } },
    series: [{ name: '年度收益', type: 'bar',
      data: DATA.annual.map(function (r) {
        return { value: r.return, itemStyle: { color: r.return >= 0 ? '#389e0d' : '#d4380d' } };
      }) }]
  });
} else { ag.style.display = 'none'; }

// 风险归因
const atg = document.getElementById('attribution-group');
if (DATA.attribution && DATA.attribution.length) {
  const labels = DATA.attribution.map(function (r) { return r.label; });
  const atc = echarts.init(document.getElementById('attr-chart'));
  atc.setOption({
    tooltip: { trigger: 'axis', valueFormatter: function (v) { return (v * 100).toFixed(2) + '%'; } },
    legend: { data: ['策略收益', '基准收益', '超额收益'], top: 0 },
    grid: grid,
    xAxis: { type: 'category', data: labels, axisLine: axisCommon.axisLine, axisLabel: axisCommon.axisLabel },
    yAxis: { type: 'value', axisLine: axisCommon.axisLine,
      axisLabel: { color: '#6b7280', formatter: function (v) { return (v * 100).toFixed(0) + '%'; } } },
    series: [
      { name: '策略收益', type: 'bar', data: DATA.attribution.map(function (r) {
        return { value: r.strategy_return, itemStyle: { color: r.strategy_return >= 0 ? '#389e0d' : '#d4380d' } }; }) },
      { name: '基准收益', type: 'bar', data: DATA.attribution.map(function (r) {
        return { value: r.benchmark_return, itemStyle: { color: r.benchmark_return >= 0 ? '#cfd8dc' : '#b0bec5' } }; }) },
      { name: '超额收益', type: 'bar', data: DATA.attribution.map(function (r) {
        return { value: r.excess_return, itemStyle: { color: r.excess_return >= 0 ? '#2f6df0' : '#9aa0a6' } }; }) }
    ]
  });

  const acc = echarts.init(document.getElementById('attr-contrib-chart'));
  acc.setOption({
    tooltip: { trigger: 'axis', valueFormatter: function (v) { return v.toFixed(1) + '%'; } },
    grid: grid,
    xAxis: { type: 'category', data: labels, axisLine: axisCommon.axisLine, axisLabel: axisCommon.axisLabel },
    yAxis: { type: 'value', axisLine: axisCommon.axisLine, axisLabel: { color: '#6b7280', formatter: '{value}%' } },
    series: [{ name: '收益贡献度', type: 'bar', data: DATA.attribution.map(function (r) {
      return { value: r.contribution_pct, itemStyle: { color: r.strategy_return >= 0 ? '#389e0d' : '#d4380d' } }; }) }]
  });

  const tbl = document.getElementById('attr-table');
  const cols = [['状态', 'label'], ['天数', 'days'], ['占样本', 'day_ratio'],
    ['策略收益', 'strategy_return'], ['基准收益', 'benchmark_return'], ['超额收益', 'excess_return'],
    ['夏普', 'sharpe'], ['日胜率', 'win_rate'], ['收益贡献', 'contribution_pct']];
  let html = '<tr>' + cols.map(function (c) { return '<th>' + c[0] + '</th>'; }).join('') + '</tr>';
  DATA.attribution.forEach(function (r) {
    let tds = '';
    cols.forEach(function (c) {
      const k = c[1]; const v = r[k];
      if (k === 'days') { tds += '<td>' + v + '</td>'; }
      else if (k === 'day_ratio') { tds += '<td>' + (v * 100).toFixed(1) + '%</td>'; }
      else if (k === 'strategy_return' || k === 'benchmark_return' || k === 'excess_return') {
        const cls = v >= 0 ? 'pos' : 'neg';
        tds += '<td class="' + cls + '">' + (v * 100).toFixed(1) + '%</td>';
      }
      else if (k === 'sharpe') { tds += '<td>' + v.toFixed(2) + '</td>'; }
      else if (k === 'win_rate') { tds += '<td>' + (v * 100).toFixed(1) + '%</td>'; }
      else if (k === 'contribution_pct') { tds += '<td>' + v.toFixed(0) + '%</td>'; }
      else { tds += '<td>' + v + '</td>'; }
    });
    html += '<tr>' + tds + '</tr>';
  });
  tbl.innerHTML = html;
} else { atg.style.display = 'none'; }

window.addEventListener('resize', function () { ec.resize(); dd.resize(); if (rc) rc.resize(); if (ac) ac.resize(); if (atc) atc.resize(); if (acc) acc.resize(); });
</script>
</body>
</html>
"""


def build_report(data: dict, out_html: str) -> None:
    """生成自包含的 HTML 回测报告。

    参数
    ----
    data : 需包含以下键
        title          报告标题
        subtitle       副标题（配置摘要）
        config         [[label, value], ...]    回测配置卡片
        metrics_groups [{title, items:[[label,value],...]}, ...]  指标分组卡片
        dates          ["2020-01-02", ...]       横轴日期
        equity         [float, ...]              策略净值
        benchmark      [float, ...] | None       基准净值
        drawdown       [float, ...]              策略回撤（负值）
        bench_dd       [float, ...] | None       基准回撤
        plan           {label: value, ...} | None  调仓摘要
        annual         [{year, return, sharpe, max_drawdown}, ...] | None  分年度表现
        rolling_dates  [str, ...] | None           滚动净值横轴
        rolling_equity [float, ...] | None         滚动样本外净值
        attribution    [{label, strategy_return, benchmark_return, excess_return,
                         sharpe, win_rate, contribution_pct, days, day_ratio}, ...] | None
                        风险归因：按市场状态(牛/熊/震荡)拆分的收益与贡献
    out_html : 输出文件路径
    """
    payload = json.dumps(data, ensure_ascii=False)
    html = (_TEMPLATE
            .replace("__TITLE__", data.get("title", "A股低频选股策略回测报告"))
            .replace("__SUBTITLE__", data.get("subtitle", ""))
            .replace("__PAYLOAD__", payload))
    Path(out_html).write_text(html, encoding="utf-8")
