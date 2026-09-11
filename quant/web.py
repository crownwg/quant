"""量化回测 Web UI: FastAPI 后端。

启动:  python -m quant.web
访问:  http://127.0.0.1:8765/
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

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


def _build_cmd(p: RunParams, prefix: str) -> list[str]:
    """把表单参数翻译成 quant.main 的命令行。

    独立成函数是为了可被单测覆盖——参数名拼错在页面上只表现为「结果不对」，
    不看命令行很难发现。
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
    if p.pool.strip():
        cmd += ["--pool", p.pool.strip()]
    else:
        cmd += ["--codes", p.codes.strip()]

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


# ----- 结果解析 -----
def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → JSON 安全的记录列表。

    不能直接用 to_dict('records')：它会保留 NaN，FastAPI 序列化时抛
    「Out of range float values are not JSON compliant」。走 to_json 会把 NaN 变成 null。
    """
    if df is None or len(df) == 0:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


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

    # rebalance plan（调仓清单，有就带上）
    plan: list[dict] = []
    plan_path = ROOT / f"{prefix}rebalance_plan.csv"
    if plan_path.exists():
        plan = _records(pd.read_csv(plan_path))

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