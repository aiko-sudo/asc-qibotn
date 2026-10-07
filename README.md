# ASC 队员选拔 · 大题 4：QiboTN 量子线路模拟优化

在 CPU 环境下，基于官方 QiboTN（qibotn 0.0.3 + qibo 0.2.23）张量网络后端，对 QFT / QAOA / Supremacy 三类量子线路进行 Statevector 与 Tensor-Network 双引擎基准测试，并完成 4 项优化尝试。

## 1. 仓库与复现信息

- 上游仓库（Fork 自）：https://github.com/qiboteam/qibotn
- 运行环境：本机（华硕 5070 笔记本，Intel Core Ultra 9 275HX）· Windows 11 · CPU 运行
- 核心依赖（见 `requirements.txt`）：
  - Python 3.12.15（conda 项目内环境 `.conda-asc`）
  - qibo==0.2.23、qibotn==0.0.3、quimb==1.15.0、cotengra 0.8.2、opt-einsum 3.4.0、numpy 2.5.3

### 复现步骤

```bash
conda create -p .conda-asc python=3.12 -y
.conda-asc/python -m pip install -r requirements.txt
# 冒烟验证官方 CPU 后端可用（无需绕过官方接口）：
.conda-asc/python -c "from qibotn.backends import MetaBackend; print(MetaBackend.load(platform='qutensornet', runcard={'MPI_enabled':False,'MPS_enabled':True,'NCCL_enabled':False,'expectation_enabled':False}))"
# 运行 baseline / 优化批次：
.conda-asc/python run_asc.py --workloads QFT --qubits "8,10,12,14,16,18,20" --engines sv,tn --depth 10 --reps 5 --mps on --detail results/baseline_detail.csv --out results/results.csv
```

## 2. 机器环境

| 项目 | 内容 |
| --- | --- |
| 机器来源 | 本地电脑（非 WSL/虚拟机） |
| 操作系统 | Windows 11 Home（Build 26200） |
| CPU | Intel Core Ultra 9 275HX，24 核 24 线程（8P+16E，无超线程），~2.7 GHz |
| 内存 | 31.4 GB（DDR5-5600 双通道 16×2） |
| GPU | NVIDIA GeForce RTX 5070 Laptop 8 GB（本题要求 CPU 运行，未参与计算） |
| Python | 3.12.15（conda 项目内环境） |
| MPI | 不适用（quimb 后端不支持 MPI；mpi4py 仅 cutensornet extra） |
| CUDA / ROCm | 不适用（cutensornet 后端 GPU-only，且无 Windows 轮子） |
| 编译 | 纯 Python + numpy，无编译环节 |

## 3. Workload 说明

| Workload | 线路定义 | 说明 |
| --- | --- | --- |
| QFT | `qibo.models.QFT(n)` | 官方内置，主 workload |
| QAOA | 标准结构：H 层 + (ZZ 纠缠层 + RX 层)×L（L=5@depth10） | 0.2.23 的 QAOA 类不暴露 Circuit，按标准结构等价构造，参数固定不寻优 |
| Supremacy | 随机置换 CNOT 层 + RY 层的随机等价线路（depth=10） | 0.2.23 无内置 supremacy 模块，按随机线路等价实现 |

> 说明：两种等价构造为题目所允许的"等价实现"而非"修改线路降低难度"，且已与 Statevector 完整核对（保真度见下）。

## 4. 基准结果

### QFT（主 workload，中位数，秒）—— Statevector vs 官方 TN

| N | Statevector | 官方 TN (quimb) | TN 加速比 |
| --- | --- | --- | --- |
| 8 | 0.0017 | 0.0492 | 0.03x |
| 10 | 0.0047 | 0.0875 | 0.05x |
| 12 | 0.0189 | 0.1421 | 0.13x |
| 14 | 0.0805 | 0.2180 | 0.37x |
| 16 | 0.4717 | 0.3140 | 1.50x |
| 18 | 2.1504 | 0.2423 | 8.87x |
| 20 | 9.1229 | 0.4517 | 20.2x |
| 24 | 150.65 | 0.9336 | ~161x |
| 26 | 超时失败（>25min） | 3.640 | — |

### QAOA / Supremacy（N=8–16，中位数，秒）

| Workload | N | SV | TN(MPS) | 说明 |
| --- | --- | --- | --- | --- |
| QAOA | 16 | 0.3800 | 0.1127 | TN 快 3.4x |
| Supremacy | 12 | 0.0260 | 0.2763 | TN 慢 |
| Supremacy | 14 | 0.0839 | 3.1985 | 纠缠增长，MPS 键维爆炸 |
| Supremacy | 16 | 0.4583 | 26.7513 | 同左，最差配置 |

完整数据见 `results/results.csv`（作业指定 5 列格式）与 `results/results_detail.csv`。

## 5. 优化尝试（≥3 项，全实测）

### 5.1 切换 TN 引擎：MPS 开 vs 关（算法路线优化）

通过 `runcard["MPS_enabled"]=False` 让官方后端使用一般张量网络（`quimb.tensor.Circuit`）而非 MPS：

| 用例 | MPS 开 | MPS 关 | 提升 |
| --- | --- | --- | --- |
| QFT N=20 | 0.4517s | 0.2536s | 1.8x |
| Supremacy N=16 | 26.7513s | 0.2868s | **92x** |
| Supremacy N=20 | ~1-1.2s | 1.0459s | 持平 |

随机线路（Supremacy）纠缠随线路增大，MPS 截断键维被迫撑满，用精确 TN 收缩反而避免了键维爆炸。数据见 `results/results_opt_off.csv`。

### 5.2 截断键维 chi 扫描（精度-性能权衡）

| chi | Supremacy N=12 fidelity | QFT N=12 fidelity |
| --- | --- | --- |
| 2 | 0.00001 | 1.000000 |
| 4 | 0.02296 | 1.000000 |
| 8 | 0.10114 | 1.000000 |
| 16 | 0.21766 | 1.000000 |
| 32 | 0.62827 | 1.000000 |
| 64 | 1.00000 | 1.000000 |

QFT 结构低纠缠，chi 从 2 起即无损；Supremacy 保真度随 chi 单调收敛——证明截断键维是真实控制精度的参数，正确性判据有效。数据见 `results/results_chi_fidelity.csv`。

### 5.3 线程/BLAS 线程数限制

| 线程设置 | QFT N=16 | QFT N=20 |
| --- | --- | --- |
| OMP=1 | 0.2614s | 0.5804s |
| OMP=24 | **0.1914s** | **0.3840s** |
| 默认（未设） | 0.3216s | 0.6196s |

24 线程最佳（N=20 相对默认快 1.6x），原因：numpy/OpenBLAS 在未显式设置时调度开销偏高，显式放开全部核心后 BLAS 矩阵/收缩运算得到充分利用。数据见 `results/results_opt_threads.csv`。

### 5.4 后端实例复用（避免重复初始化）

| N | 每次重建后端 | 复用同一后端 |
| --- | --- | --- |
| 8 | 0.0427s | 0.0275s |
| 12 | 0.0989s | 0.1319s |
| 16 | 0.2347s | 0.2342s |

收益微弱或持平——`MetaBackend.load` 本身开销很小，主要耗时在 `execute_circuit` 的 QASM 转换与收缩。数据见 `results/results_opt_reuse.csv`。

## 6. 无效 / 不适用优化项（诚实记录）

| 尝试 | 结果 | 原因 |
| --- | --- | --- |
| MPI 进程数调整 | 不适用 | quimb CPU 后端不支持 MPI（官方抛 NotImplementedError）；mpi4py 仅 cutensornet extra，且本机关无 cuQuantum |
| cutensornet GPU 后端 | 不适用 | GPU-only，cuquantum 无 Windows 轮子；本题要求 CPU |
| qibo.set_threads() | API 无效 | numpy 后端不支持多线程（官方报错 "numpy does not support more than one thread"） |
| configure_tn_simulation() | 空实现 | v0.0.3 中该方法体为 `pass`，不产生效果 |

## 7. 为什么 qubit 数增加后运行时间会变长

实测（QFT，SV）：0.47s(N=16) → 2.15(N=18) → 9.12(N=20) → 150.6(N=24)，**每增加 2 qubit 约慢 4 倍**。

- **Statevector 路径**：态空间大小为 2^N 个复数，N 每 +1，内存与每个门矩阵乘的计算量都翻倍，整体复杂度 O(N·2^N)——指数墙。这也是 N=26 时 SV 超过 25 分钟仍未完成（被计为失败项）的原因。
- **张量网络 (MPS/TN) 路径**：成本由线路的纠缠结构决定。浅层低纠缠线路（QFT）接近 O(N·χ²) 近线性（N=8→24 从 0.05s 仅涨到 0.93s）；随机深线路（Supremacy）纠缠随线路规模增长，χ 要随 N 增大，χ² 项导致超线性增长（N=12→14→16：0.28s→3.2s→26.8s）。
- **官方 CPU 后端额外因素**：qibotn quimb 路线最终 `to_dense()` 稠密化到 2^N 向量，内存也呈指数，但主要计算在收缩阶段，故 N=26 时 TN 仍只需 3.6s 而 SV 已不可行。

图示见 `results/plots/`。

## 8. 正确性验证

- 判据：与 Statevector 精确结果的保真度 |⟨SV|TN⟩|²（计时区外计算）。
- QFT 另以解析幅值核对（冒烟 N=6 保真度 0.9999999999999907）。
- 全部配置（除刻意截断的 chi 扫描）保真度 = 1.000000；chi 扫描证明判据能真实检测精度损失（Supremacy chi=2 时降为 1e-5）。
- QFT N=20、QAOA N=16、Supremacy N=16（MPS 关）逐一复核保真度均 = 1.0。