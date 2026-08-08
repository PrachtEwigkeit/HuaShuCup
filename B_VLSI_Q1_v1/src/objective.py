from __future__ import annotations

from .structures import Layout, Score


def evaluate(layout: Layout) -> Score:
    if layout.W <= 0 or layout.H <= 0:
        raise ValueError("非法布局轮廓")
    aspect = max(layout.W, layout.H) / min(layout.W, layout.H)
    return Score(area=int(layout.area), aspect=float(aspect))


def better(a: Score, b: Score) -> bool:
    """严格按照题意进行字典序比较：面积优先，面积相同再比长宽比。"""
    if a.area != b.area:
        return a.area < b.area
    return a.aspect < b.aspect


def annealing_delta(
    current: Score,
    candidate: Score,
    total_area: int,
    shape_scale: float = 0.01,
) -> float:
    """模拟退火用标量增量；最终 best 仍严格按 better() 的字典序保存。"""
    if candidate.area != current.area:
        return (candidate.area - current.area) / float(total_area)
    return shape_scale * (candidate.aspect - current.aspect)


def utilization(total_block_area: int, layout: Layout) -> float:
    return total_block_area / float(layout.area)


def dead_space_ratio(total_block_area: int, layout: Layout) -> float:
    return (layout.area - total_block_area) / float(total_block_area)
