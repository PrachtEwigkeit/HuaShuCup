# 华数杯 B题：B*-Tree + Simulated Annealing

工程保留第一问的面积优化实现，并新增第二、三问的固定轮廓 HPWL 优化。

## 问题2、3新增模型

- 将 `.nets` 看作超图，`.pl` 端口作为固定锚点，构造度数归一化图拉普拉斯；
- 调和坐标与 Fiedler 方向融合，得到带端口约束的谱初始位置；
- 通过面积均衡 shelf 将连续谱位置转换为合法 B*-Tree；
- 固定轮廓退火采用“可行性优先、可行后最小 HPWL”的字典序准则；
- 每个候选布局在轮廓剩余空间内求解析最优整体平移；
- 问题3按整数轮廓边长逐单位缩小，并以上一可行树热启动下一轮搜索。

问题2：

```bash
python scripts/run_q2.py --dataset n100 --seed 42
python scripts/run_q2.py --dataset n200 --seed 42
python scripts/run_q2.py --dataset n300 --seed 42
```

问题3：

```bash
python scripts/run_q3.py --dataset n100 --seed 42
```

快速检查流程可使用 `--config configs/q2_smoke.yaml`。正式结果使用
`configs/q2.yaml`。问题3输出中的最小死区比是给定搜索预算下的启发式最小值，
`outline_search.csv` 同时记录每个候选边长的可行性结果。

## 1. 目录

```text
B_VLSI_Q1_v1/
├── data/raw/              # n100/n200/n300.blocks
├── configs/q1.yaml        # SA 和邻域概率
├── src/
│   ├── data.py            # .blocks 解析
│   ├── structures.py      # 数据结构
│   ├── bstar_init.py      # B*-Tree 初始化
│   ├── bstar_pack.py      # B*-Tree + skyline 解码
│   ├── operators.py       # Rotate / Swap / Move
│   ├── objective.py       # 面积、长宽比、字典序
│   ├── annealing.py       # 普通 SA
│   ├── validate.py        # 树和布局合法性验证
│   ├── visualize.py       # 布局图、收敛图
│   └── io_utils.py        # CSV/JSON 输出
├── scripts/
│   ├── run_q1.py
│   └── run_q1_all.py
└── tests/
```

## 2. 安装

```bash
pip install -r requirements.txt
```

## 3. 先跑测试

在项目根目录执行：

```bash
python -m unittest discover -s tests -v
```

应保证 parser、B*-Tree 解码、扰动操作全部通过。

## 4. 跑 n100

```bash
python scripts/run_q1.py --dataset n100 --seed 42
```

结果在：

```text
results/q1/n100/seed_42/
├── layout.csv
├── summary.json
├── history.csv
├── layout.png
└── convergence.png
```

## 5. 跑 n200 / n300

```bash
python scripts/run_q1.py --dataset n200 --seed 42
python scripts/run_q1.py --dataset n300 --seed 42
```

## 6. 多随机种子

第一版默认每组 5 个种子：

```bash
python scripts/run_q1_all.py
```

会汇总到：

```text
results/q1/all_runs.csv
```

## 7. 第一版算法逻辑

1. 读取 `.blocks`，第一问不读 `.nets/.pl`；
2. 生成 B*-Tree；
3. 左孩子放在父模块右边，右孩子与父模块同 x；
4. 用整数 skyline 决定最低合法 y；
5. 用 Rotate / Swap / Move 产生邻域；
6. 普通模拟退火接受或拒绝；
7. 历史最优严格按 `(area, aspect)` 字典序保存；
8. 最终独立执行 O(n^2) 两两重叠验证。

## 8. 已核对的输入数据

- n100: 100 blocks，总面积 179501
- n200: 200 blocks，总面积 175696
- n300: 300 blocks，总面积 273170

## 9. 后续升级方向（先不要混入 v1）

- Fast-SA 三阶段温度策略；
- 更强的 B*-Tree Move/Delete-Reinsert；
- 多起点并行；
- 自适应邻域概率；
- 分段 contour 替代整数 skyline；
- 增量面积计算；
- 第二问加入 `.nets/.pl` 和 HPWL。
