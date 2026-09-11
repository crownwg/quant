# 量化工具 · A 股低频选股回测

> 一个本地跑的 A 股量化回测工具：从数据拉取 → 多池组合 → 多策略对比 → 稳健性检验 → 每日实盘清单，全链路打通。

---

## 这是什么

一套基于 **akshare + pandas + numpy** 的低频选股回测框架。从最初的"60 日动量策略能不能赚钱"到现在，工具链已经覆盖：

| 阶段 | 解决的问题 | 入口 |
|---|---|---|
| **数据层** | 增量拉取 / 本地缓存 / 涨跌停与停牌过滤 | `quant/data.py` 自动 |
| **因子库** | 7 个单因子 + 多因子合成 | `quant/factors.py` |
| **回测引擎** | 次日开盘成交 / 涨跌停 / 复权 / 印花税 | `quant/backtest.py` |
| **组合构建** | 月度调仓 / 换手缓冲 / 可执行性约束 | `quant/strategy.py` |
| **调仓清单** | 100 股整手 / 佣金与印花税 | `quant/rebalance.py` |
| **稳健性检验** | 滚动 / 样本外 / 稳健性 | `quant/rolling.py` |
| **风险归因** | 牛/熊/震荡状态拆分 | `quant/attribution.py` |
| **参数优化** | (lookback × top_n × buffer) 网格搜索 | `quant/grid.py` |
| **多策略对比** | 同池子横向 7 策略 + 报告 | `quant/compare.py` |
| **多池组合** | 多池子加权 + 相关性矩阵 + 报告 | `quant/combine.py` |
| **每日清单** | 一键生成今日调仓 + 可选 webhook 推送 | `quant/daily.py` |

---

## 快速开始

```bash
# 0. 一次性：创建虚拟环境 + 安装依赖
python -m venv .venv
.venv/Scripts/python.exe -m pip install akshare pandas numpy

# 1. 经典 60 日动量，月度调仓，次日开盘成交
.venv/Scripts/python.exe -m quant.main --codes "600031,000858,002557" \
    --start 20200101 --end 20240909 --use-open

# 2. 用沪深300 池子跑 reversal_20，看是否真比动量强
.venv/Scripts/python.exe -m quant.main --pool hs300 \
    --strategy reversal --reversal-lookback 20 --use-open --top-n 20

# 3. 多策略横向对比（消费池）
.venv/Scripts/python.exe -m quant.main --pool 消费 --compare \
    --use-open --benchmark sh000932

# 4. 多池组合（沪深300 + 消费 + 医药，每池用对策略）
.venv/Scripts/python.exe -m quant.main \
    --combine "hs300=reversal_20,消费=low_vol_60,医药=reversal_20" \
    --combine-weights "0.4,0.3,0.3" --use-open --benchmark sh000300

# 5. 每日清单（一键生成今日调仓）
.venv/Scripts/python.exe -m quant.daily --config daily.json
```

---

## 项目结构

```
quant/
├── data.py        数据层：akshare 拉取 + 本地缓存 + 增量更新
├── factors.py     因子库：7 个单因子 + zscore/rank 标准化 + combine
├── filters.py     可执行性约束：涨跌停 / 停牌 / 流动性
├── backtest.py    回测引擎：完整指标 + 成本分解
├── strategy.py    组合构建：factor_weights（含 buffer 与约束内联）
├── rebalance.py   调仓清单：build_plan / summarize
├── report.py      ECharts HTML 报告（净值/回撤/年化/滚动/年度/归因/明细）
├── universe.py    股票池：指数成分股 / 概念板块 / 自定义
├── rolling.py     稳健性：annual_breakdown / 滚动回测
├── attribution.py 风险归因：市场状态划分 + 状态拆分收益
├── grid.py        参数优化：网格搜索 + 报告
├── compare.py     多策略对比：预定义 8 策略 + 横向对比报告
├── combine.py     多池组合：跨池子加权 + 相关性矩阵
├── daily.py       日常监控：一键今日清单 + 可选 webhook 推送
└── main.py        CLI 入口：串联所有能力
```

---

## 数据层（增量缓存）

**核心设计**（`quant/data.py`）：
- 本地缓存每个股票 `data/{code}.csv`，沪深300 指数 `data/index_{symbol}.csv`
- `_ensure_cached` 决策：①已覆盖直读 ②往后追加（最常见） ③区间无数据则全量重拉
- **不补头**：akshare 的 qfq 前复权以"调用当天"为基点，补头会产生价格跳变——宁可重拉

**验证**：干净起点跑 002475（立讯精密），4 步链路：首次 1 full 2.7s → 延长 1 in_range → 再延长 1 in_range → 同窗口 0 fetch 0.0s。

---

## 因子库

| 因子 | 函数 | 含义 |
|---|---|---|
| momentum | `factors.momentum(prices, lookback, skip_recent)` | 追涨（涨得越多分越高） |
| reversal | `factors.reversal(prices, lookback)` | 反转（跌得越惨分越高） |
| ma_trend | `factors.ma_trend(prices, short, long)` | 均线趋势（短均线相对长均线） |
| ma_breakout | `factors.ma_breakout(prices, window)` | 均线突破（价格相对均线的偏离） |
| low_volatility | `factors.low_volatility(prices, lookback)` | 低波动异象（波动越小分越高） |
| volume_trend | `factors.volume_trend(volume, short, long)` | 量能（近 short 日均量/近 long 日均量） |
| combine | `factors.combine(factors, weights, method)` | 多因子合成（zscore / rank） |

---

## 7 大功能详解

### 1. 经典动量回测（最简用法）

```bash
.venv/Scripts/python.exe -m quant.main --codes "600031,000858,002557" \
    --start 20200101 --end 20240909 --use-open
```

输出：`equity_curve.csv`、`metrics.csv`、`rebalance_plan.csv`、可选 `report.html`。

### 2. 股票池自动化（不用手敲代码）

支持的池（`quant/universe.py`）：
- 指数：`hs300`、`zz500`、`zz1000`、`sz50`、`cyb`、`kcb`
- 行业：`消费`、`可选消费`、`医药`、`白酒`、`蓝筹`
- 任意 6 位指数代码（如 `000932`）
- 概念板块：`concept:白酒`

```bash
.venv/Scripts/python.exe -m quant.main --pool hs300 --top-n 20 --use-open
# 多池合并
.venv/Scripts/python.exe -m quant.main --pool "消费,hs300" --top-n 15
# 手动 + 池子混合
.venv/Scripts/python.exe -m quant.main --codes 600031,000858 --pool zz500
```

### 3. 滚动回测 / 样本外检验

```bash
.venv/Scripts/python.exe -m quant.main --pool 消费 \
    --start 20200101 --end 20240909 --use-open \
    --rolling --roll-window 504
```

每段 504 个交易日（≈2 年），不重叠；输出 `rolling_curve.csv` + `rolling_metrics.csv`。

**关键发现**：动量策略在 2020-2021 牛市 +70~+100% 但 2022-2024 持续亏损——**总收益好看≠策略稳健**。

### 4. 风险归因（按市场状态拆收益）

```bash
.venv/Scripts/python.exe -m quant.main --pool 消费 \
    --start 20200101 --end 20240909 --use-open \
    --attribution --benchmark sh000932
```

基于基准中期趋势（默认 60 日涨跌幅 ±5%）把每日标为「牛/熊/震荡」，输出每个状态下的策略收益/超额/夏普。

**关键发现**：消费动量在熊市 **跌得比基准少**（-46.6% vs -52.4%），有防御性；沪深300 动量在熊市**跌得比基准还多**（-43.0% vs -38.0%），反而放大风险。

### 5. 参数优化（网格搜索）

```bash
.venv/Scripts/python.exe -m quant.main --pool 消费 \
    --start 20200101 --end 20240909 --use-open --grid \
    --grid-lookbacks 60,90,120,160 \
    --grid-topn 10,15,20 \
    --grid-buffers 0,2
```

24 组网格；输出 `grid_results.csv` + `grid_report.html`（夏普热力图 + 明细表）。

**关键发现**：消费最优 **lb=120, n=10**；沪深300 最优 **lb=160, n=20**——**不同池子的最优参数不一样**，必须 per-pool 调优。

### 6. 多策略对比（横向挑稳健策略）

```bash
.venv/Scripts/python.exe -m quant.main --pool 消费 \
    --start 20200101 --end 20240909 --use-open \
    --compare --benchmark sh000932
```

7 个预定义策略 + 1 个可选对比（`--compare-strategies`），输出 `compare_results.csv` + `compare_report.html`。

| 策略 | 含义 |
|---|---|
| `momentum_120` / `momentum_60` | 120 日 / 60 日动量 |
| `reversal_20` | 20 日反转 |
| `low_vol_60` | 60 日低波动 |
| `ma_trend` / `ma_breakout` | 均线趋势 / 均线突破 |
| `combine_mom_lv` | 动量+低波动合成 |
| `combine_mom_rev` | 动量+反转合成 |

**关键发现（消费池 + 沪深300 池）**：
- `reversal_20` 在两个池都是**绝对收益和夏普都最强**（消费 +56.7% / 沪深300 +140.5%）
- `low_vol_60` 在两个池都是**回撤最小**（消费 -29.4% / 沪深300 -16.0%）
- **跨池"防御 + 进攻"组合**思路成立

### 7. 多池组合配置

```bash
.venv/Scripts/python.exe -m quant.main \
    --combine "hs300=reversal_20,消费=low_vol_60,医药=reversal_20" \
    --combine-weights "0.4,0.3,0.3" \
    --use-open --benchmark sh000300
```

每个池子用各自的稳健策略独立回测 → 按权重加权合成 → 输出 `combine_results.csv` + `combine_correlation.csv` + `combine_equity.csv` + `combine_report.html`。

**最优组合**（HS300 reversal_20 + 消费 low_vol_60 + 医药 reversal_20, 0.4/0.3/0.3）：
- 总收益 **+68.3%** / 夏普 **0.61** / 回撤 **-29.8%**
- 超额 **+118.9%** / IR **0.78**

**核心洞察**：**池子数量不重要，关键是每个池子用对的策略**。加错了策略（如白酒 ma_trend）反而拖累。

### 8. 每日实盘清单

```bash
# 单日清单（无持仓 CSV）
.venv/Scripts/python.exe -m quant.main --pool 消费 \
    --strategy reversal --reversal-lookback 20 \
    --start 20200101 --end 20240909 --use-open \
    --today 20240903

# 带实际持仓的 delta 清单（推荐）
# 1) 先准备 my_holdings.csv：当前券商持仓（code, shares 两列）
# 2) 运行
.venv/Scripts/python.exe -m quant.main --pool 消费 \
    --strategy reversal --reversal-lookback 20 \
    --start 20200101 --end 20240909 --use-open \
    --today 20240903 --holdings my_holdings.csv
```

**一键日常入口**（推荐）：

```json
// daily.json
{
  "pool": "消费",
  "strategy": "reversal_20",
  "top_n": 15,
  "buffer": 2,
  "use_open": true,
  "benchmark": "sh000932",
  "capital": 1000000.0,
  "holdings": "my_holdings.csv",
  "today": "20240903"
}
```

```bash
.venv/Scripts/python.exe -m quant.daily --config daily.json
# 可选：推送到 webhook（飞书/钉钉/企业微信机器人）
.venv/Scripts/python.exe -m quant.daily --config daily.json --webhook https://...
```

---

## 完整 CLI 参数参考

### `python -m quant.main`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--codes` | "" | 逗号分隔股票代码（如 `600031,000858`） |
| `--pool` | "" | 股票池（指数名/行业名/6位代码/concept:xxx） |
| `--start` / `--end` | 20200101 / 20251231 | 回测区间 |
| `--strategy` | momentum | 单因子策略（momentum/reversal/ma_trend/...） |
| `--factors` | "" | 多因子，如 `momentum=1,low_volatility=0.5` |
| `--standardize` | zscore | 多因子标准化方式 |
| `--top-n` | 10 | 选股数 |
| `--rebalance` | M | 调仓频率（M/W/Q/2M） |
| `--buffer` | 0 | 换手缓冲（0=不启用） |
| `--lookback` | 120 | 动量回看 |
| `--use-open` | False | 次日开盘成交（推荐常开） |
| `--min-volume` | 0 | 流动性门槛 |
| `--no-limit-filter` | False | 关闭涨跌停过滤 |
| `--no-suspend-filter` | False | 关闭停牌过滤 |
| `--benchmark` | sh000300 | 基准指数（空串=关闭） |
| `--attribution` | False | 启用风险归因 |
| `--regime-window` | 60 | 市场状态判定窗口 |
| `--regime-band` | 0.05 | 牛熊阈值 |
| `--rolling` | False | 启用滚动样本外回测 |
| `--roll-window` | 504 | 滚动窗口长度 |
| `--grid` | False | 启用网格搜索 |
| `--grid-lookbacks` | 60,90,120,160 | 网格 lookback 候选 |
| `--grid-topn` | 10,15,20 | 网格 top_n 候选 |
| `--grid-buffers` | 0,2 | 网格 buffer 候选 |
| `--compare` | False | 多策略对比 |
| `--compare-strategies` | "" | 自定义对比列表 |
| `--combine` | "" | 多池组合，如 `消费=reversal_20,蓝筹=low_vol_60` |
| `--combine-weights` | "" | 池子间权重（缺省等权） |
| `--today` | "" | 今日清单日期 YYYYMMDD |
| `--holdings` | "" | 当前持仓 CSV（code, shares） |
| `--out-prefix` | "" | 输出文件名前缀 |
| `--capital` | 1_000_000 | 组合本金 |
| `--fee` | 0.0003 | 佣金费率 |
| `--stamp-tax` | 0.0005 | 印花税率（仅卖出） |

### `python -m quant.daily`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | "" | JSON 配置文件路径 |
| `--pool` / `--strategy` / `--benchmark` / `--top-n` / `--webhook` | "" / "" / "" / 10 / "" | 覆盖 config 中的值 |

---

## 已发现的关键洞察

### 1. pandas 3.0 的 `rebalance_flags` breaking change
`PeriodIndex != PeriodIndex.shift(1)` 在 pandas 3.0.5 返回**全 True**，会让所有"月度调仓"变成"每日调仓"。

**修复**（`quant/strategy.py`）：用 `~period.duplicated(keep="first")`。

**影响**：在 pandas 3.0 下报告过的所有"月度/周度"回测结果都需要重跑。

### 2. reversal_20 在 A 股是个常被忽视的稳健策略
- 沪深300 reversal_20：+140.5% / 夏普 0.82 / 回撤 -35.9%
- 消费 reversal_20：+56.7% / 夏普 0.52 / 回撤 -38.5%
- 比动量更稳——尤其在 2021 年牛市后的熊市里

### 3. low_vol_60 是抗跌神器
- 沪深300 low_vol_60：+19.8% / 回撤 -16.0% ← **回撤只有其他策略的 1/2**
- 适合"防守型"配置

### 4. 沪深300 vs 消费池的熊市表现相反
- 消费动量在熊市**跌得少**（超额 +5.8%）
- 沪深300 动量在熊市**跌得更多**（超额 -5.0%）
- **沪深300 动量在熊市应该减仓**

### 5. 多池组合 ≠ 越多越好
- 消费+医药（2池）：+53.4%
- 消费+白酒+医药（3池）：+36.1%（白酒 ma_trend 拖后腿 -11%）
- HS300+消费+医药（3池，但每池用对策略）：**+68.3%**
- **关键：每个池子用对的策略 + 板块间分散**

---

## 已知坑

1. **pandas 3.0 + akshare 在 miniconda 下行为不同**——本项目所有回测必须在 `.venv` 下验证（pandas 3.0.5 / akshare 1.18.94 / numpy 2.5）。
2. **akshare qfq 复权基点会随调用日期变化**——补头会产生跳变，所以数据层故意不支持补头（要补就全量重拉）。
3. **akshare 网络不稳**：东财 push2his.eastmoney.com 偶尔断连（`RemoteDisconnected`），但 Sina 一般稳定。如果遇到 "跳过 XX 只" 的提示，可以重跑让增量补齐。
4. **价格过高的股票无法 100 股卖出**：例如清仓 6.67% × 100 万 = 6.67 万 = 茅台 51 股 < 100 股——工具自动跳过该笔并在清单中标记为 0 股。
5. **baskshare 实盘接口限制**：本工具只产出清单，不直接连券商下单——实盘前用户必须自己核对并手动下单。

---

## 输出文件清单

| 文件 | 来源 | 说明 |
|---|---|---|
| `*equity_curve.csv` | main | 净值曲线 |
| `*metrics.csv` | main | 评价指标 |
| `*rebalance_plan.csv` | main | 完整调仓清单 |
| `*report.html` | main | ECharts 可视化报告 |
| `*annual_metrics.csv` | main | 年度分解 |
| `*benchmark_curve.csv` | main | 基准净值曲线 |
| `*attribution.csv` | main + --attribution | 风险归因（牛/熊/震荡） |
| `*rolling_curve.csv` | main + --rolling | 滚动样本外净值 |
| `*rolling_metrics.csv` | main + --rolling | 滚动段指标 |
| `*grid_results.csv` | main + --grid | 网格搜索结果 |
| `*grid_report.html` | main + --grid | 网格报告（夏普热力图） |
| `*compare_results.csv` | main + --compare | 多策略对比结果 |
| `*compare_report.html` | main + --compare | 多策略对比报告 |
| `*combine_results.csv` | main + --combine | 多池组合各池指标 |
| `*combine_correlation.csv` | main + --combine | 池子间相关性矩阵 |
| `*combine_equity.csv` | main + --combine | 组合净值曲线 |
| `*combine_report.html` | main + --combine | 多池组合报告（净值 + 热力图） |
| `*today_plan.csv` | main + --today, daily | 今日调仓清单 |

---

## 进阶用法

### 自定义因子
在 `quant/factors.py` 加新函数 → 在 `quant/compare.py` / `quant/daily.py` / `quant/combine.py` 的策略路由里注册 → 即可通过 CLI 使用。

### 自定义股票池
在 `quant/universe.py` 的 `INDEX_POOLS` 里加新映射：
```python
INDEX_POOLS["工程机械"] = "000xxx"  # 工程机械指数代码
```
或者用 `concept:xxx` 直接拉概念板块。

### 自定义 webhook
`quant/daily.py` 的 `send_webhook` 是个简单 POST——飞书/钉钉/企业微信机器人都支持自定义 JSON 格式，按需修改 payload 即可。

---

## 路线图

| 阶段 | 状态 |
|---|---|
| 数据层 / 回测引擎 | ✅ |
| 股票池自动化 | ✅ |
| 滚动回测 / 样本外 | ✅ |
| 风险归因 | ✅ |
| 数据增量缓存 | ✅ |
| 多策略对比 | ✅ |
| 参数优化（网格搜索） | ✅ |
| 每日实盘清单 | ✅ |
| 多池组合配置 | ✅ |
| 日常监控脚本 | ✅ |