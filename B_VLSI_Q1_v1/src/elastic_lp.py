from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .netlist import NetlistData
from .structures import Layout


@dataclass(frozen=True)
class ElasticLPResult:
    success: bool
    layout: Layout
    objective_hpwl: float
    message: str


def optimize_fixed_topology_lp(
    layout: Layout,
    netlist: NetlistData,
    outline_side: float,
    time_limit_seconds: float = 20.0,
) -> ElasticLPResult:
    """固定当前两两分离方向，用线性规划重分配轮廓内部空白。

    每对模块保留当前布局中间隙最小的一条合法水平/垂直分离关系。
    在这些关系不变时，非重叠约束和 HPWL 都可线性化。
    """

    n = len(layout.x)
    m = netlist.n_nets
    if n != netlist.n_blocks:
        raise ValueError("layout 与 netlist 的模块数不一致")

    x0 = 0
    y0 = n
    xmin0 = 2 * n
    xmax0 = xmin0 + m
    ymin0 = xmax0 + m
    ymax0 = ymin0 + m
    n_vars = 2 * n + 4 * m

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add_constraint(coefficients: tuple[tuple[int, float], ...], bound: float) -> None:
        row = len(rhs)
        for col, value in coefficients:
            rows.append(row)
            cols.append(int(col))
            values.append(float(value))
        rhs.append(float(bound))

    x = layout.x.astype(float, copy=False)
    y = layout.y.astype(float, copy=False)
    width = layout.width.astype(float, copy=False)
    height = layout.height.astype(float, copy=False)
    tol = 1e-8

    # 为每对模块固定一条当前已满足、且间隙最小的分离关系。
    for i in range(n):
        for j in range(i + 1, n):
            relations: list[tuple[float, str, int, int]] = []
            if x[i] + width[i] <= x[j] + tol:
                relations.append((max(0.0, x[j] - x[i] - width[i]), "x", i, j))
            if x[j] + width[j] <= x[i] + tol:
                relations.append((max(0.0, x[i] - x[j] - width[j]), "x", j, i))
            if y[i] + height[i] <= y[j] + tol:
                relations.append((max(0.0, y[j] - y[i] - height[i]), "y", i, j))
            if y[j] + height[j] <= y[i] + tol:
                relations.append((max(0.0, y[i] - y[j] - height[j]), "y", j, i))
            if not relations:
                return ElasticLPResult(False, layout, float("inf"), "输入布局存在重叠")
            _, axis, before, after = min(relations, key=lambda item: item[0])
            if axis == "x":
                add_constraint(
                    ((x0 + before, 1.0), (x0 + after, -1.0)),
                    -width[before],
                )
            else:
                add_constraint(
                    ((y0 + before, 1.0), (y0 + after, -1.0)),
                    -height[before],
                )

    # 网络包围盒线性化：min <= pin <= max。
    for net_id, net in enumerate(netlist.nets):
        for raw_vertex in net:
            vertex = int(raw_vertex)
            if vertex < n:
                add_constraint(
                    ((xmin0 + net_id, 1.0), (x0 + vertex, -1.0)),
                    width[vertex] / 2.0,
                )
                add_constraint(
                    ((x0 + vertex, 1.0), (xmax0 + net_id, -1.0)),
                    -width[vertex] / 2.0,
                )
                add_constraint(
                    ((ymin0 + net_id, 1.0), (y0 + vertex, -1.0)),
                    height[vertex] / 2.0,
                )
                add_constraint(
                    ((y0 + vertex, 1.0), (ymax0 + net_id, -1.0)),
                    -height[vertex] / 2.0,
                )
            else:
                terminal = vertex - n
                tx = float(netlist.terminal_x[terminal])
                ty = float(netlist.terminal_y[terminal])
                add_constraint(((xmin0 + net_id, 1.0),), tx)
                add_constraint(((xmax0 + net_id, -1.0),), -tx)
                add_constraint(((ymin0 + net_id, 1.0),), ty)
                add_constraint(((ymax0 + net_id, -1.0),), -ty)

    objective = np.zeros(n_vars, dtype=float)
    objective[xmin0:xmax0] = -1.0
    objective[xmax0:ymin0] = 1.0
    objective[ymin0:ymax0] = -1.0
    objective[ymax0:n_vars] = 1.0

    bounds: list[tuple[float | None, float | None]] = []
    for i in range(n):
        upper = float(outline_side) - width[i]
        if upper < -tol:
            return ElasticLPResult(False, layout, float("inf"), "模块宽度超过轮廓")
        bounds.append((0.0, max(0.0, upper)))
    for i in range(n):
        upper = float(outline_side) - height[i]
        if upper < -tol:
            return ElasticLPResult(False, layout, float("inf"), "模块高度超过轮廓")
        bounds.append((0.0, max(0.0, upper)))
    bounds.extend([(None, None)] * (4 * m))

    matrix = coo_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(cols))),
        shape=(len(rhs), n_vars),
    ).tocsr()
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=np.asarray(rhs, dtype=float),
        bounds=bounds,
        method="highs",
        options={"time_limit": float(time_limit_seconds)},
    )
    if not result.success or result.x is None:
        return ElasticLPResult(False, layout, float("inf"), str(result.message))

    new_x = np.where(np.abs(result.x[x0:y0]) < 1e-9, 0.0, result.x[x0:y0])
    new_y = np.where(np.abs(result.x[y0:xmin0]) < 1e-9, 0.0, result.x[y0:xmin0])
    packed_width = float(np.max(new_x + width))
    packed_height = float(np.max(new_y + height))
    optimized = Layout(
        x=new_x,
        y=new_y,
        width=layout.width.copy(),
        height=layout.height.copy(),
        rotated=layout.rotated.copy(),
        W=packed_width,
        H=packed_height,
        area=packed_width * packed_height,
    )
    return ElasticLPResult(True, optimized, float(result.fun), str(result.message))
