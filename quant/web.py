"""量化回测 Web UI: FastAPI 后端。

启动:  python -m quant.web
访问:  http://127.0.0.1:8765/
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from quant.data import cache_freshness

# 项目根目录: .../量化/
ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "index.html"

app = FastAPI(title="量化回测 Web", version="1.0")


# ----- 入参模型 -----
class RunParams(BaseModel):
    pool: str = Field("", description="股票池（消费/医药/hs300 等），留空则用 codes")
    codes: str = Field("", description="手动股票代码，逗号分隔")
    strategy: str = Field("momentum", description="单因子名")
    lookback: int = Field(120, description="动量 / 低波等窗口")
    top_n: int = Field(15, description="每次调仓选几只")
    buffer: int = Field(0, description="缓冲带")
    rebalance: str = Field("M", description="调仓频率 M/W/Q")
    start: str = Field("20200101")
    end: str = Field("20240909")
    use_open: bool = Field(True)
    benchmark: str = Field("sh000300")

    # ----- 仓位管理（择时 / 波动率目标）-----
    timing: str = Field("off", description="趋势择时模式：off/ma/momentum/dual")
    timing_lookback: int = Field(120, description="择时均线或动量窗口（交易日）")
    timing_band: float = Field(0.0, description="滞回带，0.02 = ±2% 缓冲")
    timing_min_exposure: float = Field(0.0, description="看空时保留的最低仓位")
    timing_max_exposure: float = Field(1.0, description="最高仓位，1.0 = 不加杠杆")
    vol_target: float = Field(0.0, description="目标年化波动率（如 0.15），0 = 关闭")
    timing_proxy: str = Field("auto", description="当「市场」用的指数，auto = 按股票池自动选")

    # ----- 样本外验证 -----
    walk_forward: bool = Field(False, description="跑 walk-forward 样本外验证")
    fw_train: float = Field(2.0, description="训练窗（年）")
    fw_test: float = Field(0.5, description="测试窗（年）")
    grid_timing: str = Field("", description="扫择时模式，如 off,ma,dual；留空则不扫")

    # ----- 调仓清单（实盘用）-----
    holdings_text: str = Field(
        "", description="持仓，每行『代码,股数』；留空则用项目里的 my_holdings.csv")
    plan_date: str = Field("", description="清单基准日 YYYYMMDD，留空 = 回测区间最后一天")


# ----- 因子评价（IC / IR）入参 -----
# 与 RunParams 分开：因为 IC 评估需要「因子名 + 因子专属窗口」，
# 而回测只需要「单因子名 + 通用窗口」。让用户在页面上直观地选多个因子诊断。
class FactorEvalParams(BaseModel):
    pool: str = Field("", description="股票池（消费/医药/hs300 等）")
    codes: str = Field("", description="手动代码，逗号分隔；与股票池取并集")
    factors: str = Field(
        "momentum,reversal,low_volatility",
        description="因子列表，逗号分隔；按 list 顺序出报告")
    horizon: int = Field(20, description="前向持有期（交易日）；月度调仓用 20")
    n_groups: int = Field(5, description="分组数，默认 5")

    # 因子窗口参数（不同因子用不同窗口）
    lookback: int = Field(120, description="动量 / 低波回看期")
    skip_recent: int = Field(0, description="动量跳过最近 N 日（12-1 动量的 -1）")
    reversal_lookback: int = Field(20, description="反转回看期")
    ma_short: int = Field(20, description="均线趋势短期")
    ma_long: int = Field(60, description="均线趋势长期")
    ma_window: int = Field(60, description="均线突破窗口")
    vol_lookback: int = Field(60, description="低波动回看期")
    vol_short: int = Field(5, description="量能短期")
    vol_long: int = Field(60, description="量能长期")

    start: str = Field("20210101")
    end: str = Field("20240909")


# ----- 路由 -----
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    if not TEMPLATE.exists():
        raise HTTPException(500, f"模板未找到: {TEMPLATE}")
    return TEMPLATE.read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "python": str(PYTHON),
        "python_exists": PYTHON.exists(),
        "cwd": str(ROOT),
        "template_exists": TEMPLATE.exists(),
    }


def _safe_file(filename: str, suffixes: tuple[str, ...]) -> Path:
    """只允许访问项目根目录下的指定类型文件，挡掉路径穿越。"""
    safe = Path(filename).name
    fp = ROOT / safe
    if safe != filename or not fp.exists() or fp.suffix.lower() not in suffixes:
        raise HTTPException(404, f"文件不存在: {filename}")
    return fp


@app.get("/report/{filename}", response_class=HTMLResponse)
async def report(filename: str) -> str:
    """内嵌展示回测报告 HTML（单次 / walk-forward / 网格）。"""
    return _safe_file(filename, (".html", ".htm")).read_text(encoding="utf-8")


@app.get("/download/{filename}")
async def download(filename: str) -> FileResponse:
    """下载结果 CSV。"""
    fp = _safe_file(filename, (".csv",))
    return FileResponse(fp, media_type="text/csv; charset=utf-8", filename=fp.name)


def _materialize_holdings(text: str, prefix: str) -> tuple[str | None, str]:
    """把持仓文本落成 CLI 需要的 CSV 文件，返回 (路径, 来源说明)。

    文本为空时回落到项目根目录的 `my_holdings.csv`——用户什么都不填也能直接出清单。
    解析故意宽容：逗号/空格/分号/制表符都能当分隔符、允许 `#` 注释行、
    代码不足 6 位自动补零（用户很可能直接粘「600519 100」过来）。
    """
    raw = (text or "").strip()
    if not raw:
        default = ROOT / "my_holdings.csv"
        if default.exists():
            return str(default), "项目自带的 my_holdings.csv"
        return None, ""

    rows: list[dict] = []
    for line in raw.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = [x for x in re.split(r"[,\s;，、\t]+", line) if x]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            shares = int(float(parts[1]))
        except ValueError:
            continue
        if shares <= 0:
            continue
        rows.append({"code": parts[0].zfill(6), "shares": shares})
    if not rows:
        return None, ""
    path = ROOT / f"{prefix}holdings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path), f"页面填写的 {len(rows)} 只持仓"


def _build_cmd(p: RunParams, prefix: str, plan_only: bool = False,
               holdings_path: str | None = None,
               refresh_only: bool = False) -> list[str]:
    """把表单参数翻译成 quant.main 的命令行。

    独立成函数是为了可被单测覆盖——参数名拼错在页面上只表现为「结果不对」，
    不看命令行很难发现。

    plan_only=True 时只生成调仓清单：main.py 的 holdings 分支是独立 return 的
    （不会产出净值曲线），所以清单与完整回测不能在同一次调用里兼得。
    refresh_only=True 时只更新数据缓存，同样不产出任何回测结果。
    """
    cmd = [
        str(PYTHON), "-m", "quant.main",
        "--start", p.start,
        "--end", p.end,
        "--top-n", str(p.top_n),
        "--strategy", p.strategy,
        "--lookback", str(p.lookback),
        "--rebalance", p.rebalance,
        "--buffer", str(p.buffer),
        "--benchmark", p.benchmark,
        "--out-prefix", prefix,
    ]
    if p.use_open:
        cmd.append("--use-open")
    # --pool 与 --codes 可以同时传：main.py 会把两者合并去重（manual + pool_codes）。
    # 页面早期写成「二选一」，导致想「在消费池基础上补一只 002557 洽洽食品」
    # 时只能二选一——而洽洽食品并不在任何主流指数成分股里，就没法加进来了。
    if p.pool.strip():
        cmd += ["--pool", p.pool.strip()]
    if p.codes.strip():
        cmd += ["--codes", p.codes.strip()]

    # --- 数据维护：只更新缓存，早退，不跑回测 ---
    if refresh_only:
        return cmd + ["--refresh-data"]

    # --- 仓位管理：趋势择时 ---
    timing_mode = (p.timing or "off").strip().lower()
    if timing_mode != "off":
        cmd += [
            "--timing", timing_mode,
            "--timing-lookback", str(int(p.timing_lookback)),
            "--timing-band", str(float(p.timing_band)),
            "--timing-min-exposure", str(float(p.timing_min_exposure)),
            "--timing-max-exposure", str(float(p.timing_max_exposure)),
        ]
    # --- 仓位管理：波动率目标 ---
    if float(p.vol_target or 0.0) > 0:
        cmd += ["--vol-target", str(float(p.vol_target))]
    # --- 择时用的「市场」代理指数 ---
    if (p.timing_proxy or "auto").strip() not in ("", "auto"):
        cmd += ["--timing-proxy", p.timing_proxy.strip()]

    # --- 样本外验证 ---
    if p.walk_forward:
        # 把选股网格钉死在用户填的这一组上，只让「择时」这一维去扫。
        # 为什么不放开：不传这三个参数时 main.py 会套用 CLI 默认的
        # 4(窗口)×3(topN)×2(缓冲)=24 组选股网格，再乘择时组合会膨胀到 700+ 组，
        # 一次要跑几小时。网页上只需要回答「我这组参数可不可信、要不要择时」。
        cmd += [
            "--walk-forward",
            "--fw-train", str(float(p.fw_train)),
            "--fw-test", str(float(p.fw_test)),
            "--grid-lookbacks", str(int(p.lookback)),
            "--grid-topn", str(int(p.top_n)),
            "--grid-buffers", str(int(p.buffer)),
        ]
        gt = (p.grid_timing or "").strip() or "off,ma,momentum,dual"
        cmd += ["--grid-timing", gt]

    # --- 调仓清单 ---
    if plan_only and holdings_path:
        cmd += ["--holdings", holdings_path]
        if (p.plan_date or "").strip():
            cmd += ["--today", p.plan_date.strip()]

    return cmd


@app.post("/api/run")
async def run_backtest(p: RunParams) -> dict:
    if not PYTHON.exists():
        raise HTTPException(500, f"找不到 venv 解释器: {PYTHON}")
    if not p.pool.strip() and not p.codes.strip():
        raise HTTPException(400, "必须填写「股票池」或「手动代码」其中之一")

    prefix = f"web_{uuid.uuid4().hex[:8]}_"
    cmd = _build_cmd(p, prefix)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT),
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"error": "回测超时（>30 分钟），请缩小池子、缩短区间，或关掉样本外验证。"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"启动回测失败: {exc}"}

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        return {"error": f"回测退出码 {proc.returncode}\n{tail}"}

    try:
        if p.walk_forward:
            return _parse_wf_results(prefix, proc.stdout, cmd)
        return _parse_results(prefix, cmd)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"结果解析失败: {exc}"}


def _resolve_codes(p) -> tuple[list[str], str | None]:
    """把页面上的「股票池 / 手动代码」解析成 6 位代码列表。

    直接复用 quant.universe，不另写一套解析——两边口径一旦不同，
    页面会告诉你「池里有 37 只」而回测其实用了别的池子，很难查。

    两者的关系是**并集**而不是二选一，与 main.py 一致：这样才能
    「在消费池上补一只 002557」——它在所有指数成分股里都没有，
    只能靠手动代码加进来。
    返回 (codes, error)；error 非空表示解析失败。

    p 只要有 .pool / .codes 两个字段即可，RunParams 和 FactorEvalParams
    都用这个解析路径，避免两套口径漂移。
    """
    from quant import universe

    pool = (getattr(p, "pool", "") or "").strip()
    raw_codes = (getattr(p, "codes", "") or "").strip()
    codes: list[str] = []
    if pool:
        try:
            codes, _ = universe.build_universe_report(pool)
        except Exception as exc:  # noqa: BLE001  网络/池名错都归到这里
            return [], f"股票池解析失败: {exc}"

    raw = [c.strip() for c in re.split(r"[,\s;，、]+", raw_codes) if c.strip()]
    seen = set(codes)
    for c in raw:
        if c.isdigit():
            c6 = c.zfill(6)
            if c6 not in seen:
                seen.add(c6)
                codes.append(c6)

    if not codes:
        return [], "请填写「股票池」或「手动代码」"
    return codes, None


@app.post("/api/data_freshness")
async def data_freshness(p: RunParams) -> dict:
    """数据体检：池里每只标的的本地缓存更新到哪天。

    只读本地 CSV，不联网，所以可以随页面加载自动调。
    存在的意义：默认结束日是 2024-09-09，用户很可能拿到的是一份
    **历史回放清单**却不自知——这个接口让页面能把这件事明确指出来。
    """
    codes, err = _resolve_codes(p)
    if err:
        return {"ok": False, "error": err}

    rows = cache_freshness(codes)
    dated = [r for r in rows if r["last"]]
    if not dated:
        # 字段与下面正常分支保持一致：前端只写一套渲染逻辑，
        # 少一个键就会渲染成 undefined，而不是报错。
        return {"ok": True, "n_codes": len(codes), "with_data": 0, "latest": "",
                "end": p.end, "usable": False, "behind_end": False,
                "ahead_of_data": False, "stale": [], "n_stale": 0,
                "empty": codes[:50], "n_empty": len(codes),
                "message": "池内没有任何本地缓存数据，请先点「更新数据」（约 1 分钟）。"}

    latest = max(r["last"] for r in dated)
    latest_ts = pd.Timestamp(latest)
    end_ts = pd.Timestamp(p.end)
    stale = [r for r in dated
             if (latest_ts - pd.Timestamp(r["last"])).days > 5]
    empty = [r["code"] for r in rows if not r["last"]]

    # 两个方向都要看，方向不同、后果不同：
    #   end 远晚于数据  → 数据没更新够，最后一截是空的；
    #   end 远早于数据  → （默认 20240909 就落在这里）拿到的是一份
    #                     「如果在 2024-09-09 调仓该买什么」的历史回放清单，
    #                     看着像今天的操作建议，其实不是。这个更危险。
    gap_end_too_new = (end_ts - latest_ts).days
    gap_end_too_old = (latest_ts - end_ts).days
    # 留 5 天容差：周末 / 长假 / 停牌都会让「最新交易日」天然早于日历日，
    # 差 1~5 天属正常，不该天天报警。
    too_new = gap_end_too_new > 5
    too_old = gap_end_too_old > 5

    msgs: list[str] = []
    if too_old:
        msgs.append(f"⚠️ 结束日 {end_ts.date()} 比数据最新日 {latest} 早了 "
                    f"{gap_end_too_old} 天：这份清单是**历史回放**"
                    f"（回答的是「那天该买什么」），不是「今天该买什么」。"
                    f"要拿能用的清单，把结束日改成 {latest}。")
    elif too_new:
        msgs.append(f"⚠️ 数据只到 {latest}，比结束日 {end_ts.date()} 早 "
                    f"{gap_end_too_new} 天：末尾这段没有数据，"
                    f"请先点「更新数据」。")
    else:
        msgs.append(f"✓ 数据已更新到 {latest}，与结束日 {end_ts.date()} 一致。")

    if stale:
        msgs.append(f"（池内 {len(stale)} 只明显落后，多为长期停牌或新上市，属正常。）")
    if empty:
        msgs.append(f"（有 {len(empty)} 只完全没有缓存数据，选股会跳过它们。）")

    return {
        "ok": True,
        "n_codes": len(codes),
        "with_data": len(dated),
        "latest": latest,
        "end": str(end_ts.date()),
        "behind_end": bool(too_new),
        "ahead_of_data": bool(too_old),
        "usable": bool(not too_old and not too_new),
        "stale": stale[:30],
        "n_stale": len(stale),
        "empty": empty[:50],
        "n_empty": len(empty),
        "message": " ".join(msgs),
    }


@app.post("/api/refresh_data")
async def refresh_data(p: RunParams) -> dict:
    """把池内数据增量补齐到 --end（联网拉取，耗时随池子大小增长）。

    单独一个按钮而不是回测的副作用：不这样的话，用户只会更新到
    「本次选中的那几只」，池里其余标的永远是旧的。
    """
    if not PYTHON.exists():
        raise HTTPException(500, f"找不到 venv 解释器: {PYTHON}")
    if not p.pool.strip() and not p.codes.strip():
        raise HTTPException(400, "必须填写「股票池」或「手动代码」其中之一")

    prefix = f"web_{uuid.uuid4().hex[:8]}_"
    cmd = _build_cmd(p, prefix, refresh_only=True)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, cwd=str(ROOT),
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"error": "更新数据超时（>60 分钟）。池子太大时请改用命令行分批更新。"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"启动失败: {exc}"}

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        return {"error": f"退出码 {proc.returncode}\n{tail}"}

    res = await data_freshness(p)
    res["stdout_tail"] = (proc.stdout or "")[-3000:]
    res["cmd"] = cmd
    return res


@app.post("/api/plan")
async def run_plan(p: RunParams) -> dict:
    """只生成「今日调仓清单」：按实际持仓算出该买 / 该卖多少股。

    单独做一个入口而不是回测里的勾选项 —— main.py 的 holdings 分支会直接
    return，不产出净值曲线，两者没法在同一次调用里兼得。
    """
    if not PYTHON.exists():
        raise HTTPException(500, f"找不到 venv 解释器: {PYTHON}")
    if not p.pool.strip() and not p.codes.strip():
        raise HTTPException(400, "必须填写「股票池」或「手动代码」其中之一")

    prefix = f"web_{uuid.uuid4().hex[:8]}_"
    hold_path, hold_src = _materialize_holdings(p.holdings_text, prefix)
    if hold_path is None:
        raise HTTPException(400, "持仓格式无法识别：每行写「代码,股数」，例如 600519,100")

    cmd = _build_cmd(p, prefix, plan_only=True, holdings_path=hold_path)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT),
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"error": "生成清单超时（>15 分钟），请缩短回测区间。"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"启动失败: {exc}"}

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        return {"error": f"退出码 {proc.returncode}\n{tail}"}

    try:
        res = _parse_results(prefix, cmd)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"结果解析失败: {exc}"}

    # 取不到价格的持仓（通常是池外股票）：main.py 会打印一行告警，这里捞出来单独展示。
    # 不显示的话，用户会把「不在清单里」误读成「这只不用动」。
    skipped: list[str] = []
    m = re.search(r"无法计算[^\n]*?[:：]\s*([0-9、,，\s]+)", proc.stdout or "")
    if m:
        skipped = re.findall(r"\d{6}", m.group(1))

    res["mode"] = "plan"
    res["holding_source"] = hold_src
    res["skipped"] = skipped
    res["plan_date"] = (p.plan_date or "").strip() or "回测区间最后一天"
    return res


# ----- 因子评价（IC / IR） -----
# 与上面走 subprocess 模式不同，本接口走 in-process：
#   - 数据量小（一只池子 < 200 只 × 几百期），不需要像回测那样隔离
#   - 复用 main.py 的 FACTOR_BUILDERS 保证因子口径与回测完全一致
#   - 复用 factor_eval.evaluate_many 拿 IC / IR / 衰减 / 分组收益
#   - 复用 factor_eval.build_ic_report 出 HTML 报告（含 ECharts 三图 + 汇总表）
# 不进 subprocess 的另一理由：网页上「按月 IC」跑 6 个因子也就 10~20 秒，
# 起一个新 Python 进程的 1~2 秒比值不大，但能少维护一个 CLI 子命令。
_KNOWN_FACTORS = {"momentum", "reversal", "ma_trend", "ma_breakout",
                  "low_volatility", "volume_trend"}


class _FactorArgs:
    """容器：装给 FACTOR_BUILDERS 用的窗口参数。

    FACTOR_BUILDERS 的签名是 (args, prices, volume)，其中 args 像 argparse
    Namespace 一样属性访问。这里只装它真正读的几个字段。
    """

    def __init__(self, p: FactorEvalParams) -> None:
        self.lookback = p.lookback
        self.skip_recent = p.skip_recent
        self.reversal_lookback = p.reversal_lookback
        self.ma_short = p.ma_short
        self.ma_long = p.ma_long
        self.ma_window = p.ma_window
        self.vol_lookback = p.vol_lookback
        self.vol_short = p.vol_short
        self.vol_long = p.vol_long


@app.post("/api/factor_eval")
async def factor_eval(p: FactorEvalParams) -> dict:
    """对一组因子做 IC / IR 诊断，生成 HTML 报告与摘要表。

    为什么单独有这条而不是回测的隐藏模式：
    IC 检验直接回答「这个因子到底有没有用」，是回测赚钱之前的必要一步。
    把它从回测里拉出来独立成入口，是为了防止用户「没诊断就上策略」。
    """
    from quant import factor_eval as fe
    from quant.data import load_panel
    from quant.main import FACTOR_BUILDERS

    codes, err = _resolve_codes(p)
    if err:
        return {"error": err}

    # 因子名过滤：未知直接报错，HTML 报告里 render 会 panic
    factors = [s.strip() for s in p.factors.split(",") if s.strip()]
    unknown = [f for f in factors if f not in _KNOWN_FACTORS]
    if unknown:
        return {"error": f"不支持的因子: {','.join(unknown)}；可选: {','.join(sorted(_KNOWN_FACTORS))}"}
    if not factors:
        return {"error": "请至少选一个因子"}

    # 区间反推 fetch_start（多留一倍最大窗口，避免 IC 第一期没数据）
    # 这些窗口 1~120 不等，120 倍数 1.5 倍够用；保守起见按 240 天回看。
    fetch_start = (pd.Timestamp(p.start) - pd.Timedelta(days=240)).strftime("%Y%m%d")

    try:
        panel = load_panel(codes, fetch_start, p.end, sleep=0.3, on_error="skip")
    except Exception as exc:  # noqa: BLE001  数据源打嗝不该让接口崩
        return {"error": f"加载数据失败: {exc}"}

    if panel["close"].empty:
        return {"error": "没有任何可用数据，请先点「更新数据」把池子补齐"}

    if len(panel["close"].columns) < len(codes):
        skipped_codes = set(codes) - set(panel["close"].columns)
        # 不返回 error：部分缺失仍可出报告，前端在 UI 上提示即可
        load_warning = f"已过滤 {len(skipped_codes)} 只无数据标的: {','.join(sorted(skipped_codes)[:20])}"
    else:
        load_warning = ""

    args = _FactorArgs(p)
    score_dict: dict[str, pd.DataFrame] = {}
    for name in factors:
        try:
            score_dict[name] = FACTOR_BUILDERS[name](args, panel["close"], panel["volume"])
        except Exception as exc:  # noqa: BLE001  单个因子挂了不应该让整组都失败
            return {"error": f"计算 {name} 因子失败: {exc}"}

    try:
        results = fe.evaluate_many(score_dict, panel["close"],
                                   n_groups=p.n_groups)
    except Exception as exc:
        return {"error": f"IC 评估失败: {exc}"}

    # 报告文件写到项目根目录，UUID 前缀防撞
    out_html = ROOT / f"factor_{uuid.uuid4().hex[:8]}.html"
    title = f"因子有效性 · {p.pool.strip() or '自定义代码'}"
    subtitle = (f"区间 {p.start}~{p.end} · {len(score_dict)} 个因子 · "
                f"{len(panel['close'].columns)} 只标的 · 前向 {p.horizon} 日")
    fe.build_ic_report(results, str(out_html), title=title, subtitle=subtitle)

    # 摘要表给前端展示（不必再让用户打开完整报告就有结论）
    summary = []
    for r in results:
        s = r["summary"]
        summary.append({
            "name": r["name"],
            "n": s["n"],
            "mean_ic": _json_safe(s["mean_ic"]),
            "std_ic": _json_safe(s["std_ic"]),
            "ir": _json_safe(s["ir"]),
            "t_stat": _json_safe(s["t_stat"]),
            "positive_ratio": _json_safe(s["positive_ratio"]),
            "verdict": s["verdict"],
            "long_short": _json_safe(r["long_short"]),
        })

    return {
        "ok": True,
        "report": out_html.name,
        "factors": factors,
        "n_codes": len(panel["close"].columns),
        "n_periods": next((s["n"] for s in [r["summary"] for r in results] if s["n"]), 0),
        "summary": summary,
        "warning": load_warning,
    }


# ----- 结果解析 -----
def _json_safe(v):
    """把单个单元格转成 JSON 能安全序列化的值。

    NaN / Inf → None（否则 FastAPI 抛 Out of range float values）；
    其余按原类型返回，**不要碰字符串**。
    """
    if v is None:
        return None
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, np.floating):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (int, str, bool)):
        return v
    return str(v)


def _read_csv_codes(path: Path) -> pd.DataFrame:
    """读结果 CSV，并把 code 列强制还原成 6 位字符串。

    pandas 默认会把「全是数字」的列推断成整数：`000858` 进来就是 `858`，
    页面上股票代码直接少三位、`000001` 变成 `1`。
    输出端（`_json_safe`）怎么修都没用——数据在这一步就已经错了。
    """
    df = pd.read_csv(path)
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    return df


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → JSON 安全的记录列表。

    两个坑都要绕开，这里各踩过一次：
    1. `to_dict('records')` 保留 NaN，FastAPI 序列化时抛
       「Out of range float values are not JSON compliant」。
    2. `to_json` 能把 NaN 变成 null，但它**会把「看起来像数字的字符串」列转成数字** ——
       股票代码 "000858" 变成 858、"002714" 变成 2714，页面上代码就全错了。
    所以逐格判断类型：只把 NaN/Inf 变 None，字符串原样保留。
    """
    if df is None or len(df) == 0:
        return []
    return [{k: _json_safe(v) for k, v in rec.items()}
            for rec in df.to_dict("records")]


def _parse_results(prefix: str, cmd: list | None = None) -> dict:
    eq_path = ROOT / f"{prefix}equity_curve.csv"
    if not eq_path.exists():
        raise FileNotFoundError(f"找不到 {eq_path}")

    edf = pd.read_csv(eq_path)
    edf["date"] = pd.to_datetime(edf["date"])
    edf = edf.sort_values("date").reset_index(drop=True)
    dates = [d.strftime("%Y-%m-%d") for d in edf["date"]]
    equity = edf["equity"].astype(float).tolist()
    drawdown = _drawdown_series(pd.Series(equity))

    # metrics
    metrics: dict = {}
    metrics_path = ROOT / f"{prefix}metrics.csv"
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        for _, row in mdf.iterrows():
            try:
                metrics[str(row["metric"])] = {
                    "label": str(row["label"]),
                    "value": float(row["value"]),
                }
            except (ValueError, TypeError):
                pass

    # benchmark
    benchmark = None
    bench_path = ROOT / f"{prefix}benchmark_curve.csv"
    if bench_path.exists():
        bdf = pd.read_csv(bench_path)
        bdf["date"] = pd.to_datetime(bdf["date"])
        benchmark = {
            "dates": [d.strftime("%Y-%m-%d") for d in bdf["date"]],
            "values": bdf["benchmark"].astype(float).tolist(),
        }

    # annual
    annual: list[dict] = []
    annual_path = ROOT / f"{prefix}annual_metrics.csv"
    if annual_path.exists():
        annual = _records(pd.read_csv(annual_path))

    # monthly heatmap
    monthly = _compute_monthly(edf)

    # 仓位管理指标（只有启用择时 / 波动率目标时才存在）
    exposure = {k: metrics[k] for k in ("avg_exposure", "in_market_ratio",
                                        "exposure_switches") if k in metrics}

    # rebalance plan（完整周期的调仓计划）
    plan: list[dict] = []
    plan_path = ROOT / f"{prefix}rebalance_plan.csv"
    if plan_path.exists():
        plan = _records(_read_csv_codes(plan_path))

    # today_plan（给了 --holdings / --today 才产出：该买/该卖多少股）
    today_plan: list[dict] = []
    tp_path = ROOT / f"{prefix}today_plan.csv"
    if tp_path.exists():
        today_plan = _records(_read_csv_codes(tp_path))

    return {
        "ok": True,
        "mode": "single",
        "metrics": metrics,
        "dates": dates,
        "equity": equity,
        "drawdown": drawdown,
        "benchmark": benchmark,
        "annual": annual,
        "monthly": monthly,
        "exposure": exposure,
        "plan": plan,
        "today_plan": today_plan,
        "prefix": prefix,
        "report_file": f"{prefix}report.html",
        "cmd": cmd or [],
        "stdout_tail": "",
    }


def _parse_wf_results(prefix: str, stdout: str = "", cmd: list | None = None) -> dict:
    """解析 walk-forward 的结果：各臂汇总 + 逐折明细 + 完整报告链接。"""
    sm_path = ROOT / f"{prefix}wf_summary.csv"
    if not sm_path.exists():
        raise FileNotFoundError(f"找不到 {sm_path}（walk-forward 未产出结果）")
    summary = _records(pd.read_csv(sm_path))
    folds: list[dict] = []
    folds_path = ROOT / f"{prefix}wf_folds.csv"
    if folds_path.exists():
        folds = _records(pd.read_csv(folds_path))
    return {
        "ok": True,
        "mode": "walk_forward",
        "summary": summary,
        "folds": folds,
        "prefix": prefix,
        "report_file": f"{prefix}wf_report.html",
        "summary_file": f"{prefix}wf_summary.csv",
        "folds_file": f"{prefix}wf_folds.csv",
        "cmd": cmd or [],
        "stdout_tail": (stdout or "")[-4000:],
    }


def _drawdown_series(equity: pd.Series) -> list[float]:
    cummax = equity.cummax()
    dd = (equity / cummax - 1.0).fillna(0.0)
    return dd.astype(float).tolist()


def _compute_monthly(equity_df: pd.DataFrame) -> dict:
    """生成月度收益热力图数据: years x months 矩阵。"""
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["ret"] = df["equity"].pct_change().fillna(0.0)
    monthly_ret = df.groupby(["year", "month"], as_index=False)["ret"].sum()
    matrix = monthly_ret.pivot(index="year", columns="month", values="ret").fillna(0.0)
    return {
        "years": matrix.index.astype(int).tolist(),
        "months": matrix.columns.astype(int).tolist(),
        "values": [[round(float(v), 4) for v in row] for row in matrix.values.tolist()],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("WEB_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")