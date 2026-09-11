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
| **样本外验证** | 滚动训练/测试，识别参数过拟合 | `quant/grid.py --walk-forward` |
| **多策略对比** | 同池子横向 7 策略 + 报告 | `quant/compare.py` |
| **多池组合** | 多池子加权 + 相关性矩阵 + 报告 | `quant/combine.py` |
| **因子中性化** | 剥离行业 / 小市值 beta | `quant/neutralize.py` |
| **因子有效性** | IC / IR / 衰减 / 分组收益 | `quant/factor_eval.py` |
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
├── grid.py        参数优化：网格搜索 + walk-forward 样本外验证 + 报告
├── compare.py     多策略对比：预定义 8 策略 + 横向对比报告
├── combine.py     多池组合：跨池子加权 + 相关性矩阵
├── neutralize.py  因子中性化：行业映射缓存 + 行业/市值分组去均值
├── factor_eval.py 因子有效性：IC / IR / 衰减 / 分组收益 + 报告
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
- 全市场：`all`（含已退市个股）

```bash
.venv/Scripts/python.exe -m quant.main --pool hs300 --top-n 20 --use-open
# 多池合并
.venv/Scripts/python.exe -m quant.main --pool "消费,hs300" --top-n 15
# 手动 + 池子混合
.venv/Scripts/python.exe -m quant.main --codes 600031,000858 --pool zz500
```

#### ⚠️ 幸存者偏差与「时点快照」

指数成分股接口返回的是**当前**成分股。用它回溯历史，等于「只在今天还活着的公司里选股」，
当年在指数里、后来被剔除或退市暴跌的股票压根不在池子里，收益会被系统性高估。

本项目用三层机制处理，按推荐度排序：

**① `--pool all`：全市场建池（最彻底）**
```bash
.venv/Scripts/python.exe -m quant.main --pool all --min-listed-days 250 --use-open
```
池子不依赖指数名单，「某股当时能否交易」完全由行情数据（当日有无 K 线）决定，
退市股也在池内，因此**从构造上就不存在幸存者偏差**。
代价是标的数量大（数千只），首次下载慢。

**② `--as-of` + 成分股时点快照**
每次联网拉取成分股都会自动归档到 `data/cons_snapshots/cons_<指数>_<YYYYMMDD>.csv`，
所以从启用之日起会逐步积累可回溯的时点历史：
```bash
.venv/Scripts/python.exe -m quant.main --pool 消费 --start 20220101 --as-of 20220101
```
请求日期早于所有已有快照时会**显式告警**并提示偏差，绝不静默使用有偏数据。

**③ `--min-listed-days`：剔除次新股（便宜的补充）**
```bash
.venv/Scripts/python.exe -m quant.main --pool 消费 --min-listed-days 250
```
上市未满 N 个自然日的股票不参与选股，避开新股连续一字板与「上市前 5 日不设涨跌停」的失真区间。
注意：启用后预热期会自动前移，代价是首次运行要多拉一段历史。

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
| `--pool` | "" | 股票池（指数名/行业名/6位代码/concept:xxx/`all` 全市场含退市） |
| `--as-of` | "" | 股票池时点日期，用于取历史成分股（缺省取 `--start`） |
| `--min-listed-days` | 0 | 剔除上市未满 N 自然日的次新股（建议 250） |
| `--start` / `--end` | 20200101 / 20251231 | 回测区间 |
| `--strategy` | momentum | 单因子策略（momentum/reversal/ma_trend/...） |
| `--factors` | "" | 多因子，如 `momentum=1,low_volatility=0.5` |
| `--standardize` | zscore | 多因子标准化方式 |
| `--top-n` | 10 | 选股数 |
| `--rebalance` | M | 调仓频率（M/W/Q/2M） |
| `--buffer` | 0 | 换手缓冲（0=不启用） |
| `--lookback` | 120 | 动量回看 |
| `--use-open` | False | 次日开盘成交（推荐常开） |
| `--weighting` | equal | 权重方案：equal/inv_vol/score/rank |
| `--max-weight` | 0 | 单票权重上限（0=不限） |
| `--neutralize` | "" | 因子中性化：`industry,size` |
| `--min-volume` | 0 | 流动性门槛（股数口径） |
| `--min-amount` | 0 | 流动性门槛（金额口径，元） |
| `--max-participation` | 0 | 容量约束参与率（如 0.05，0=不限） |
| `--no-limit-filter` | False | 关闭涨跌停过滤 |
| `--no-suspend-filter` | False | 关闭停牌过滤 |
| `--slippage` | 0.0005 | 单边滑点率 |
| `--min-commission` | 5 | 单笔最低佣金（元） |
| `--impact-coef` | 0 | 平方根冲击成本系数（0=关闭） |
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
| `--walk-forward` | False | 样本外验证（训练窗选参 → 测试窗纯验证） |
| `--fw-train` | 2.0 | walk-forward 训练窗（年） |
| `--fw-test` | 0.5 | walk-forward 测试窗（年） |
| `--fw-metric` | sharpe | 训练窗选参依据（sharpe/calmar/total_return） |
| `--factor-eval` | False | 因子有效性检验（IC/IR/衰减/分组） |
| `--compare` | False | 多策略对比 |
| `--compare-strategies` | "" | 自定义对比列表 |
| `--combine` | "" | 多池组合，如 `消费=reversal_20,蓝筹=low_vol_60` |
| `--combine-weights` | "" | 池子间权重（缺省等权） |
| `--today` | "" | 今日清单日期 YYYYMMDD |
| `--holdings` | "" | 当前持仓 CSV（code, shares） |
| `--out-prefix` | "" | 输出文件名前缀 |
| `--capital` | 1_000_000 | 组合本金（同时驱动最低佣金与容量约束） |
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
2. **akshare qfq 复权基点会随区间变化**——所以「往前补头」不能增量拼接，必须**全量重拉后覆盖**。数据层现在这样做，并用 `data/_coverage.json` 记录「已验证的起始日」，避免每次运行都重复回补。
3. **akshare 网络不稳**：东财 push2his.eastmoney.com 偶尔断连（`RemoteDisconnected`），但 Sina 一般稳定；某些网络环境下 `www.bse.cn`（北交所）会被代理直接拒绝。全市场池构建已做多源降级，单个源失败不影响整体。遇到 "跳过 XX 只" 可重跑让增量补齐。
4. **价格过高的股票无法 100 股卖出**：例如清仓 6.67% × 100 万 = 6.67 万 = 茅台 51 股 < 100 股——工具自动跳过该笔并在清单中标记为 0 股。
5. **本工具只产出清单，不直接连券商下单**——实盘前必须自己核对并手动下单。
6. **涨跌停判定必须与成交价同口径**：按收盘价成交就用收盘价判（封板/一字板），按次日开盘价成交就只用**开盘价**判。
   混用等于用买入时还没发生的信息决定能不能下单（前视偏差）。`filters.tradability` 的
   `exec_at_open` 与 `factor_weights` 的 `exec_shift` 就是为此存在的，改调用链时不要漏传。
7. **「一字板」的判据方向不同**：涨停看 `low`（全天没跌破），跌停看 `high`（全天没涨破）。
   两个方向用同一字段会把「开盘跌停后拉起」误判成一字跌停——本项目真实踩过这个坑
   （002157 于 2020-02-04 开盘 10.57 跌停、最高 11.68、收 11.48）。
8. **权重矩阵里绝不能出现 NaN**。末尾的 `ffill()` 把 NaN 解释成「沿用上期持仓」，
   所以任何一行只要有 NaN，那一列就会一直持有下去、永不卖出。
   真实踩过：某次改动让 `_allocate` 只返回入选股票的权重（未补全 0），
   结果 13 次调仓 382 笔订单里 **0 笔卖出**，回撤凭空变大。
   新增权重方案时必须 `.reindex(score.columns).fillna(0.0)`。
9. **`inv_vol` 必须取倒数**。写成 `vol / vol.sum()` 会变成「波动率越大权重越大」，
   与风险平价方向完全相反。也已踩过（高波动票拿到 57% 仓位），现有单测锁死。
10. **最低佣金要按「笔数」算**：佣金下限是对每一笔委托成立的，
    不是对当日成交额汇总成立。写成 `max(当日总成交额×费率, 5元)` 就永远不生效。
11. **市值必须用真实成交均价**：缓存里的收盘价是前复权价，
    乘流通股本会系统性低估早期市值（复权价随时间被整体缩放）。
    正解是 `成交额 ÷ 成交量 × 流通股本`——这两个字段本地缓存里都有，无需联网。
12. **成本敏感度**：把 `--slippage` 从 0 调到 5bp、加上 5 元最低佣金后，
    消费池同期总收益从 42.2% 掉到 40.4%。换手越高的策略、本金越小的账户，
    这项影响越大。对比历史结果时务必确认成本口径一致。

---

## 方法论修正记录（P0）

以下三项是「会让回测结论本身失真」的问题，已修复。它们不影响功能，但影响可信度。
（P1 的六项严谨性修正见后文。）

### P0-1 幸存者偏差 → 成分股时点化

**问题**：指数成分股接口返回的是**当前**成分股。用它回溯历史等于只在「今天还活着的公司」里选股，
后来被剔除/退市暴跌的股票从未进池，收益被系统性高估。

**修复**（`quant/universe.py`）：
- 每次拉取成分股自动归档快照 `data/cons_snapshots/cons_<指数>_<日期>.csv`，逐日积累时点历史；
- `--as-of` 按历史时点取成分股；无可用快照时**显式告警**并给出缓解建议，绝不静默放行；
- `--pool all` 全市场建池（含已退市，多源降级获取），从构造上消除该偏差；
- `--min-listed-days` 按上市自然日剔除次新股（避开新股连续一字板与无涨跌停限制期）。

### P0-2 判定口径与成交价错位 → exec_at_open / exec_shift

**问题**：统一用收盘价判定「能否成交」，但 `--use-open` 时实际按开盘价成交。
等于用买入时还没发生的信息决定能不能下单，属前视偏差。两处错位：
1. 涨跌停判定用收盘价 → 改为按成交价判定（`filters.tradability(exec_at_open=...)`）；
2. 调仓决策在 t 日、成交在 t+1 开盘，可交易性却取 t 日 → 新增
   `factor_weights(exec_shift=...)`，把可交易性面板后移到**成交当日**。

### P0-3 涨跌停判定粗糙 → 改用涨跌停价 + 一字板识别

**问题**：旧实现比 `close.pct_change() >= 涨跌幅度 - 0.5%`，既无法识别一字板，
又因容差过宽而误禁（实测消费池 36 只 × 1320 日中，误禁买 15 次、误禁卖 11 次）。

**修复**（`quant/filters.py`）：由前收盘价推出涨跌停**价**（分位四舍五入），
再用 open/high/low/close 与之比较，区分「收盘封板」与「一字板」；
判据方向按方向区分（涨停看 low、跌停看 high）。

**影响**：消费池 2020-01-01~2024-09-09 同参数对比，修复后总收益 43.60% → 42.23%、
夏普 0.441 → 0.433、超额 68.6% → 67.0%。方向正确（去掉的是虚增部分），
但幅度不大——因为消费池都是流动性好的大盘股，很少触及板。
**换成小盘池或 `--pool all`，差异会显著放大。**

### 回归测试

```bash
.venv/Scripts/python.exe -m pytest tests -q     # 105 项
```

覆盖：涨跌停价计算与一字板识别（含「开盘跌停后拉起」这一真实坑）、判定口径对齐、
`exec_shift` 语义、成分股快照 as-of 选取与回退告警、次新股掩码、缓存头部回补幂等性；
以及 P1 的成本模型、权重方案与上限分配、容量约束、中性化、IC 计算、walk-forward 窗口切分。

---

## 严谨性修正记录（P1）

P0 修的是「结论会不会错」，P1 修的是「结论会不会被打折」。六项全部落地。

### P1-1 成本模型 → 滑点 + 最低佣金 + 冲击成本

**问题**：只算佣金和印花税。月频换手 10 只票，滑点通常吃掉 1~3%/年；
且佣金没有「单笔最低 5 元」约束——低本金下这一项比费率本身更重要。

**修复**（`quant/backtest.py`）：成本拆成四块
```
佣金     = max(成交额 × 费率, 笔数 × 最低佣金)   ← 必须按笔数，不能按当日汇总额
印花税   = 卖出额 × 税率
滑点     = 双边成交额 × 滑点率                   ← 新增，默认 5bp
冲击成本 = coef × sqrt(该票成交额 / 该票 ADV)     ← 新增，默认关闭
```
新增 `cost_kwargs()` 统一六个入口的取参，杜绝「主回测算滑点、网格搜索不算」的口径分裂。

**影响**（消费池 2020-01-01~2024-09-09 同参数）：
总收益 42.23% → **40.42%**，夏普 0.433 → **0.422**，超额 67.0% → **64.9%**，
成本年化拖累 **0.57%**。改用 `--slippage 0 --min-commission 0` 可回到旧口径。

```bash
--slippage 0.0005        # 单边滑点，默认 5bp
--min-commission 5       # 单笔最低佣金（元），默认 5
--impact-coef 0.1        # 平方根冲击模型，0=关闭
```

### P1-2 成交量约束 → 按 ADV 限制单票权重

**问题**：回测假设「想买多少都买得到」。小市值票实际吃不下大仓位，
一次性吃掉某只票当日 30% 成交额时，回测里假设的成交价根本拿不到。

**修复**：`filters.capacity_cap()` —— 单票权重上限 = 参与率 × 20日均成交额 / 本金。
超出部分按比例再分配给未触顶标的（注水法），全部触顶则**留现金**，不强行满仓。

```bash
--max-participation 0.05    # 参与率 5%，0=不限
--min-amount 50000000       # 金额口径流动性门槛（比 --min-volume 更合理）
```

> 本金 100 万时对消费类大盘股不构成约束；把 `--capital` 调到 1 亿以上才会显现。

### P1-3 因子中性化 → 行业 + 市值

**问题**：动量因子在 A 股同时是一个小市值因子。小盘股暴涨的行情里，
你以为赚的是「动量」，其实赚的是「小盘 beta」——行情切换时原样还回去，净值曲线上完全看不出来。

**修复**（`quant/neutralize.py`）：每个交易日内先减行业均值、再减市值分组均值。
市值用 **真实成交均价（成交额÷成交量）× 流通股本**，而不是前复权价×股本
（后者会系统性低估早期市值）。行业映射走新浪行业（49 个），缓存 30 天。

```bash
--neutralize industry,size
```

> 单行业池（如「白酒」）做行业中性化只是全体平移，**不改变选股结果**——池子里本来就没有行业差异。

### P1-4 因子有效性 → IC / IR / 衰减 / 分组收益

**问题**：净值上涨 ≠ 因子有效。可能是运气、可能是 beta、可能是过拟合，
三者从一条净值曲线上无法区分。

**修复**（`quant/factor_eval.py`）：直接回答「分数靠前的股票是否真的跑赢」。

```bash
--factor-eval
```

指标刻度：`|IC| < 0.02` 基本无效 / `0.03~0.05` 可用 / `> 0.1` 先排查数据泄漏。
**IR（IC均值÷IC标准差）比 IC 均值更该被重视**——IC 均值 0.05 但波动 0.3 的因子实则不可用。
分组收益的**单调性**比 IC 更难伪造：只有最高组一枝独秀时，收益多半由少数个股贡献，实盘复制不到。

消费池实测（2020-2024，未来 20 日收益）：

| 因子 | IC均值 | IR | 正IC占比 | 评价 |
|---|---|---|---|---|
| momentum(120) | 0.0425 | 0.159 | 58.2% | 有效可用 |
| low_volatility(60) | 0.0504 | 0.192 | 56.2% | 强有效 |
| reversal(20) | 0.0127 | 0.054 | 53.5% | 基本无效 |

### P1-5 权重方案 → 波动率倒数 / 分数加权 / 单票上限

**问题**：`1/len(picks)` 硬编码等权，集中度无处约束。

**修复**（`quant/strategy.py`）：

```bash
--weighting inv_vol      # equal(默认) / inv_vol 波动率倒数 / score 分数加权 / rank 排名加权
--max-weight 0.2         # 单票上限，超出部分注水式再分配，全触顶则留现金
```

`inv_vol` 是风险平价在忽略相关性时的实用近似：低波动票拿更多权重，组合的事前波动更均衡。

### P1-6 样本外验证 → walk-forward

**问题**：`--grid` 是全样本网格搜索，即「事后选参」+「多重比较」。
试的组合越多，碰巧最优的那个越漂亮——这是纯粹的噪声。
表现是训练期夏普 1.2、实盘立刻掉到 0.3。

**修复**（`grid.walk_forward_search`）：把样本切成「训练窗 + 紧邻测试窗」，
训练窗内选参、测试窗内**完全不改参**，把各测试窗收益连乘成一条真实可得的曲线。
同时给出两个对照组：**全样本最优固定参数**（事后选参的上界幻觉）与**默认参数**（不调参）。

```bash
--walk-forward --fw-train 2 --fw-test 0.5 --fw-metric sharpe
```

消费池实测（2020-01~2024-09，5 折）：

| 参数策略 | 样本外总收益 | 年化 | 平均夏普 | 正收益折占比 |
|---|---|---|---|---|
| Walk-Forward 自适应 | −12.6% | −5.1% | −0.26 | 40% |
| 全样本最优固定（事后选参） | −7.8% | −3.1% | −0.07 | 60% |
| 默认参数（不调参） | −10.7% | −4.3% | −0.22 | 40% |

**事后选参在样本外比自适应选参「好」4.8 个百分点——这 4.8pp 就是幻觉的量化值。**
更值得注意的是：5 折里每一折选出来的都是 `lb=120 n=10`（即默认参数），
说明这个网格上根本不存在比默认更好的参数，调参空间是空的。

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
| `*wf_folds.csv` | main + --walk-forward | 每折选中的参数与样本内外表现 |
| `*wf_summary.csv` | main + --walk-forward | 三种参数策略的样本外汇总对比 |
| `*wf_report.html` | main + --walk-forward | 样本外验证报告（拼接净值 + 过拟合诊断） |
| `*factor_eval.html` | main + --factor-eval | 因子有效性报告（IC 曲线 / 衰减 / 分组收益） |
| `*combine_results.csv` | main + --combine | 多池组合各池指标 |
| `*combine_correlation.csv` | main + --combine | 池子间相关性矩阵 |
| `*combine_equity.csv` | main + --combine | 组合净值曲线 |
| `*combine_report.html` | main + --combine | 多池组合报告（净值 + 热力图） |
| `*today_plan.csv` | main + --today, daily | 今日调仓清单 |
| `data/industry_map.csv` | neutralize | 股票→行业映射缓存（30 天有效） |

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
| Web 控制台 | ✅ |
| P0 方法论纠错（幸存者偏差 / 判定口径 / 涨跌停价法） | ✅ |
| P1 严谨性（成本模型 / 容量约束 / 中性化 / IC检验 / 权重方案 / 样本外验证） | ✅ |

### 仍未做（按价值排序）

1. **择时 / 仓位管理**——现在永远满仓或空仓。归因已证明纯多头躲不开 beta
   （消费池牛市 +118.7%、熊市 −46.6%，超额全程为正）。
   加一层基于指数的趋势择时或波动率目标（vol targeting），回撤能明显压下来。
2. **回撤熔断 / 止损**——−30% 到 −45% 的回撤目前是硬扛过去的。
3. **协方差感知的权重**——`inv_vol` 只用了对角阵（忽略相关性）。真正的风险平价需要协方差矩阵估计。
4. **多因子权重自动寻优**——现在 `--factors momentum=1,low_volatility=0.5` 的权重靠手调，
   可以用 IC 加权或最大化 IC_IR 来定。
5. **行业映射覆盖率**——新浪行业只覆盖约 50% 的 A 股（2985/5923）。
   未覆盖的标的不会被行业中性化，做全市场中性化时需注意。
6. **工程化**——`main.py` 仍是 40KB 单文件，该拆成 cli / pipeline / output 三层；
   Web 层用 `subprocess` 同步跑长任务会阻塞。