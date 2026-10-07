"""ASC QiboTN 大题：CPU 环境下的量子线路张量网络仿真基准与优化。

workloads : QFT / QAOA / Supremacy(随机等价线路)
engines   : sv  = qibo Statevector 基线
            tn  = 官方 qibotn quimb 后端 (platform="qutensornet", CPU)

计时口径：建线 + 演化 全部计入；正确性验证在计时区之外执行。
"""

import argparse
import csv
import gc
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import qibo  # noqa: E402
from qibo import gates  # noqa: E402
from qibo.models import Circuit, QFT  # noqa: E402

QIBO_PLATFORMS = ("cutensornet", "qutensornet", "qmatchatea")


def _mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def build_qaoa(n, layers=4, seed=42):
    """标准 QAOA 线路结构（TFIM 类问题），参数固定不做寻优。"""
    rng = np.random.RandomState(seed)
    c = Circuit(n)
    for q in range(n):
        c.add(gates.H(q))
    betas = rng.uniform(0.1, np.pi, layers)
    gammas = rng.uniform(0.1, np.pi, layers)
    for L in range(layers):
        for q in range(n - 1):
            # CNOT + RZ 的 ZZ 等价分解，OpenQASM / quimb 均识别
            c.add(gates.CNOT(q, q + 1))
            c.add(gates.RZ(q + 1, 2 * gammas[L]))
            c.add(gates.CNOT(q, q + 1))
        for q in range(n):
            c.add(gates.RX(q, 2 * betas[L]))
    return c


def build_random(n, depth=10, seed=42):
    """随机等价线路（Supremacy 风格）：随机置换 CNOT 层 + 全比特 RY 层。"""
    rng = np.random.RandomState(seed)
    c = Circuit(n)
    for _ in range(depth):
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(0, n - 1, 2):
            c.add(gates.CNOT(perm[i], perm[i + 1]))
        for q in range(n):
            c.add(gates.RY(q, rng.uniform(0, 2 * np.pi)))
    return c


BUILD = {
    "QFT": lambda n, depth: QFT(n),
    "QAOA": lambda n, depth: build_qaoa(n, layers=max(1, depth // 2)),
    "Supremacy": lambda n, depth: build_random(n, depth=depth),
}


def make_tn(mps=True, max_bond=None, cutoff=1e-6):
    """加载官方 qibotn 的 CPU 张量网络后端。

    优化项：按 runcard 缓存后端实例，避免每个规模重复 MetaBackend.load /
    重建 QuimbBackend（省去重复 import quimb 与后端初始化开销）。
    """
    from qibotn.backends import MetaBackend

    if not mps:
        mps_val = False
    elif max_bond:
        mps_val = {
            "method": "svd",
            "cutoff": cutoff,
            "cutoff_mode": "abs",
            "max_bond": max_bond,
        }
    else:
        mps_val = True
    runcard = {
        "MPI_enabled": False,
        "MPS_enabled": mps_val,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }
    key = str(runcard)
    cached = make_tn._cache.get(key)
    if cached is not None:
        return cached
    be = MetaBackend.load(platform="qutensornet", runcard=runcard)
    make_tn._cache[key] = be
    return be


make_tn._cache = {}


def fidelity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return None
    return float(abs(np.vdot(a / na, b / nb)) ** 2)


def timed(fn, reps):
    """第 0 次为预热并留下结果，其余 reps 次计入统计。"""
    times, out = [], None
    for r in range(reps + 1):
        gc.collect()
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        if r == 0:
            out = res
        else:
            times.append(dt)
    return times, out


def one(workload, n, engine, depth, reps, tn_kwargs, sv_ref, max_sv_ref=18):
    row = {
        "workload": workload,
        "qubits": n,
        "engine": engine,
        "depth": depth,
        "reps": reps,
        "threads": os.environ.get("OMP_NUM_THREADS", "default"),
        "ram_peak_mb": 0.0,
        "runtime": None,
        "t_min": None,
        "t_max": None,
        "fidelity": None,
        "status": "failed",
        "note": "",
    }
    try:
        if engine == "sv":
            fn = lambda: np.asarray(BUILD[workload](n, depth)().state()).copy()  # noqa: E731
            ts, state = timed(fn, reps)
            if n <= max_sv_ref:
                sv_ref[(workload, n)] = state  # 复用预热结果作参考，不重算
        else:
            be = make_tn(**tn_kwargs)
            fn = lambda: np.asarray(be.execute_circuit(BUILD[workload](n, depth)).state()).copy()  # noqa: E731
            ts, state = timed(fn, reps)
            row["note"] = "MPS_enabled=%s" % (False if not tn_kwargs.get("mps", True) else tn_kwargs.get("max_bond", "default"))
        row["runtime"] = round(statistics.median(ts), 4)
        row["t_min"] = round(min(ts), 4)
        row["t_max"] = round(max(ts), 4)
        row["ram_peak_mb"] = round(_mem_mb(), 1)
        ref = sv_ref.get((workload, n))
        if ref is not None and ref.shape == state.shape:
            f = fidelity(ref, state)
            row["fidelity"] = round(f, 8) if f is not None else None
        elif engine == "sv":
            row["fidelity"] = 1.0
        row["status"] = "success"
    except Exception as e:
        row["note"] = "%s: %s" % (type(e).__name__, str(e)[:120])
        if "MemoryError" in type(e).__name__ or "allocate" in row["note"]:
            row["note"] = "MemoryError (32GB RAM 上限) | " + row["note"]
    return row


COLS = ["workload", "qubits", "command", "runtime", "status"]
DETAIL = [
    "workload", "qubits", "engine", "depth", "reps", "threads",
    "runtime", "t_min", "t_max", "fidelity", "ram_peak_mb", "status", "note",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="QFT")
    ap.add_argument("--qubits", default="8,10,12,14,16,18,20")
    ap.add_argument("--engines", default="sv")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mps", default="on", help="on|off|<max_bond>")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--detail", default="results/results_detail.csv")
    ap.add_argument("--max-sv-ref", type=int, default=18, help="超过此规模不再算 SV 参考保真度")
    a = ap.parse_args()

    workloads = a.workloads.split(",")
    qubits = [int(x) for x in a.qubits.split(",")]
    engines = a.engines.split(",")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    tn_kwargs = {}
    if a.mps == "off":
        tn_kwargs["mps"] = False
    elif a.mps != "on":
        tn_kwargs["max_bond"] = int(a.mps)

    new_detail = not os.path.exists(a.detail)
    fd = open(a.detail, "a", newline="", encoding="utf-8")
    fw = open(a.out, "a", newline="", encoding="utf-8")
    dw, ww = csv.DictWriter(fd, DETAIL), csv.DictWriter(fw, COLS)
    if new_detail:
        fd.write("# qibo %s | numpy %s | python %s | OMP=%s\n" % (
            qibo.__version__, np.__version__, sys.version.split()[0],
            os.environ.get("OMP_NUM_THREADS", "default")))
        dw.writeheader()
        ww.writerow({"workload": "# qibo %s | numpy %s | python %s | OMP=%s | mps=%s" % (
            qibo.__version__, np.__version__, sys.version.split()[0],
            os.environ.get("OMP_NUM_THREADS", "default"), a.mps)})
        ww.writeheader()

    sv_ref = {}
    for w in workloads:
        for n in qubits:
            for eng in engines:
                cmd = "python run_asc.py --workloads %s --qubits %d --engines %s --depth %d --reps %d --mps %s" % (
                    w, n, eng, a.depth, a.reps, a.mps)
                if eng == "sv" or n not in sv_ref or w not in sv_ref:
                    pass
                r = one(w, n, eng, a.depth, a.reps, tn_kwargs, sv_ref, a.max_sv_ref)
                r["command"] = cmd
                dw.writerow({k: r.get(k) for k in DETAIL})
                ww.writerow({k: r.get(k) for k in COLS})
                fd.flush()
                fw.flush()
                print("%-9s N=%-3d %-3s runtime=%-8s fid=%-10s mem=%-9s %s %s" % (
                    w, n, eng, r["runtime"], r["fidelity"], r["ram_peak_mb"],
                    r["status"], r["note"][:60]), flush=True)
    fd.close()
    fw.close()


if __name__ == "__main__":
    main()
