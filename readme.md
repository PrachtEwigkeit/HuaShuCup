# 华数杯 B 题：VLSI 布图规划设计

本仓库用于求解 2026 年第七届“华数杯”大学生数学建模竞赛 B 题
“VLSI 布图规划设计”。项目包含赛题原始资料、三组 GSRC 芯片数据、
B\*-Tree 布图表示、模拟退火优化、端口锚定图拉普拉斯谱初始化，以及问题
1、2、3 的正式计算结果。

## 目录结构

```text
HuaShuCup/
├── readme.md                     # 仓库总览（当前文件）
├── .vscode/                      # VS Code 工作区配置
├── source/                       # 赛题、参考论文和原始附件
│   └── B题 VLSI布图规划设计/
│       ├── B题 VLSI布图规划设计.pdf
│       ├── B题 VLSI布图规划设计_00.png
│       ├── Modern Floorplanning Based on B ∗ -Tree and Fast Simulated.pdf
│       └── 附件/                 # n100、n200、n300 的 blocks/nets/pl 文件
├── B_VLSI_Q1_v1/                # 三问的 Python 建模与求解工程
│   ├── configs/                 # 正式与快速验证配置
│   ├── data/raw/                # 求解器使用的数据副本
│   ├── scripts/                 # 问题 1、2、3 的运行入口
│   ├── src/                     # 算法、解析、验证和输出模块
│   ├── tests/                   # 单元测试
│   ├── results/                 # 三问的布局和统计结果
│   ├── requirements.txt
│   └── README.md                # 算法工程的详细说明
└── tmp/                         # 临时文件目录，可为空
```

## 三问求解方法

### 问题 1：面积优先的无固定轮廓布图

- 使用 B\*-Tree 表示不可二划分矩形布局；
- 采用整数 skyline 解码，保证模块不重叠；
- 使用 Rotate、Swap 和 Move 邻域进行模拟退火；
- 按 `(芯片轮廓面积, 长宽比)` 字典序保存最优方案。

运行入口：`B_VLSI_Q1_v1/scripts/run_q1.py`。

### 问题 2：固定死区比下最小化 HPWL

- 从 `.nets` 构造低度超图，将 `.pl` 固定端口作为谱锚点；
- 使用图拉普拉斯调和坐标和 Fiedler 方向生成谱初始位置；
- 通过高度递减 shelf、跨行谱交换和 B\*-Tree 合法化生成初始布局；
- 固定题设死区比 `0.15`，以轮廓可行性优先、总 HPWL 最小为目标退火；
- 对每个候选布局解析求解轮廓内的最优整体平移；
- HPWL 使用低度超图填充矩阵进行向量化计算。

运行入口：`B_VLSI_Q1_v1/scripts/run_q2.py`。

### 问题 3：最小可行死区比

- 将模块包围盒边长作为整数变量搜索；
- 由模块总面积、最大模块边长和固定端口范围建立解析下界；
- 优先探测解析下界；若下界不可行，则从问题 2 解逐单位缩小轮廓；
- 使用上一可行 B\*-Tree 热启动下一轮搜索；
- 固定最小可行边长后，再单独优化该轮廓下的 HPWL。

当前模型采用“`.pl` 中的固定端口必须位于芯片轮廓内”的工程假设。三组正式
结果均在端口解析下界处构造出可行布局，因此最小死区比由下界和可行解共同
确认。

运行入口：`B_VLSI_Q1_v1/scripts/run_q3.py`。

## 环境安装

建议使用 Python 3.10 或更高版本。在项目根目录执行：

```powershell
cd B_VLSI_Q1_v1
python -m pip install -r requirements.txt
```

主要依赖为 NumPy、Matplotlib 和 PyYAML，不要求 SciPy。

## 运行方法

以下命令均在 `B_VLSI_Q1_v1/` 下执行。

### 运行测试

```powershell
python -m unittest discover -s tests -v
```

### 问题 1

```powershell
python scripts/run_q1.py --dataset n100 --seed 42
python scripts/run_q1.py --dataset n200 --seed 42
python scripts/run_q1.py --dataset n300 --seed 42
```

运行问题 1 的多随机种子汇总：

```powershell
python scripts/run_q1_all.py
```

### 问题 2

```powershell
python scripts/run_q2.py --dataset n100 --seed 42 --config configs/q2.yaml
python scripts/run_q2.py --dataset n200 --seed 42 --config configs/q2.yaml
python scripts/run_q2.py --dataset n300 --seed 42 --config configs/q2.yaml
```

### 问题 3

```powershell
python scripts/run_q3.py --dataset n100 --seed 42 --config configs/q2.yaml
python scripts/run_q3.py --dataset n200 --seed 42 --config configs/q2.yaml
python scripts/run_q3.py --dataset n300 --seed 42 --config configs/q2.yaml
```

快速检查程序流程时，可将正式配置替换为：

```powershell
--config configs/q2_smoke.yaml
```

快速配置只用于合法性和流程检查，不能替代正式计算结果。

## 配置文件

```text
configs/
├── q1.yaml          # 问题 1 正式配置
├── q1_smoke.yaml    # 问题 1 快速检查
├── q2.yaml          # 问题 2、3 正式配置
└── q2_smoke.yaml    # 问题 2、3 快速检查
```

`q2.yaml` 可调整：

- 谱初始化变体数量和谱扩散强度；
- 初始温度、终止温度、降温系数和每温度扰动次数；
- Rotate、Swap、Move、连线引导邻域概率；
- 问题 3 的端口轮廓假设、下界探测和重复求解次数。

## 核心源码

```text
src/
├── data.py              # .blocks 解析
├── netlist.py           # .nets/.pl 解析及超图结构
├── structures.py        # B*-Tree、Layout 等数据结构
├── bstar_init.py        # 问题 1 初始树
├── spectral_init.py     # 端口锚定谱初始化与合法化
├── bstar_pack.py        # B*-Tree + skyline 解码
├── operators.py         # Rotate、Swap、Move 邻域
├── annealing.py         # 问题 1 模拟退火
├── fixed_outline.py     # 固定轮廓 HPWL、最优平移和问题 2 退火
├── q3_search.py         # 问题 3 最小轮廓搜索
├── objective.py         # 问题 1 目标函数
├── validate.py          # 树、边界和模块重叠验证
├── fixed_io.py          # 问题 2、3 结果输出
└── visualize.py         # 布局图与收敛图
```

## 结果目录

```text
results/
├── q1/{dataset}/seed_42/
├── q2/{dataset}/seed_42/
├── q3/{dataset}/seed_42/
└── 第二问和第三问结果.zip
```

每个问题的单次结果通常包含：

- `layout.csv`：模块坐标、尺寸和旋转状态；
- `layout.png`：模块布局可视化；
- `summary.json`：轮廓、死区比、HPWL 等关键指标；
- `history.csv`：模拟退火迭代历史；
- `convergence.png`：收敛曲线。

问题 3 另外包含 `outline_search.csv`，记录候选轮廓边长、死区比、可行性和
对应 HPWL。

## 当前正式结果

| 问题 | 数据集 | 轮廓边长 | 模块包围盒 | 总 HPWL | 死区比 |
|---|---|---:|---:|---:|---:|
| 2 | n100 | 454.341446 | 454 × 454 | 245306.305 | 0.15 |
| 2 | n200 | 449.500167 | 449 × 448 | 444068.500 | 0.15 |
| 2 | n300 | 560.486842 | 560 × 560 | 627811.566 | 0.15 |
| 3 | n100 | 444 | 444 × 442 | 252796.000 | 0.09824458 |
| 3 | n200 | 438 | 438 × 438 | 457592.000 | 0.09190875 |
| 3 | n300 | 548 | 548 × 548 | 636447.500 | 0.09933009 |

正式结果已经过模块数量、轮廓边界和两两不重叠验证。详细指标以各结果目录
中的 `summary.json` 为准。
