from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .data import BlockData


_HEADER_RE = re.compile(r"^(?P<key>NumNets|NumPins)\s*:\s*(?P<value>\d+)\s*$")
_DEGREE_RE = re.compile(r"^NetDegree\s*:\s*(?P<degree>\d+)\s*$")


@dataclass(frozen=True)
class NetlistData:
    """模块和固定端口组成的超图。

    nets 中的顶点统一编码：0..n_blocks-1 为模块，之后为固定端口。
    """

    nets: tuple[np.ndarray, ...]
    terminal_names: tuple[str, ...]
    terminal_x: np.ndarray
    terminal_y: np.ndarray
    n_blocks: int
    block_to_nets: tuple[np.ndarray, ...]

    @property
    def n_terminals(self) -> int:
        return len(self.terminal_names)

    @property
    def n_nets(self) -> int:
        return len(self.nets)

    @property
    def n_pins(self) -> int:
        return int(sum(len(net) for net in self.nets))

    def is_block(self, vertex: int) -> bool:
        return 0 <= int(vertex) < self.n_blocks

    def terminal_index(self, vertex: int) -> int:
        idx = int(vertex) - self.n_blocks
        if not (0 <= idx < self.n_terminals):
            raise IndexError(f"顶点 {vertex} 不是固定端口")
        return idx


def load_terminals(path: str | Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    path = Path(path)
    names: list[str] = []
    xs: list[float] = []
    ys: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"无法解析端口坐标行: {line}")
            names.append(fields[0])
            xs.append(float(fields[1]))
            ys.append(float(fields[2]))

    if not names:
        raise ValueError(f"未在 {path} 中解析到固定端口")
    if len(set(names)) != len(names):
        raise ValueError(f"{path} 中存在重复端口名")

    return (
        tuple(names),
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
    )


def load_netlist(
    nets_path: str | Path,
    pl_path: str | Path,
    blocks: BlockData,
) -> NetlistData:
    terminal_names, terminal_x, terminal_y = load_terminals(pl_path)
    block_index = {name: i for i, name in enumerate(blocks.names)}
    terminal_index = {name: i for i, name in enumerate(terminal_names)}

    declared_nets: int | None = None
    declared_pins: int | None = None
    raw_nets: list[list[str]] = []
    pending_degree: int | None = None
    pending_pins: list[str] = []

    with Path(nets_path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            header = _HEADER_RE.match(line)
            if header is not None:
                value = int(header.group("value"))
                if header.group("key") == "NumNets":
                    declared_nets = value
                else:
                    declared_pins = value
                continue

            degree = _DEGREE_RE.match(line)
            if degree is not None:
                if pending_degree is not None:
                    raise ValueError("遇到新 NetDegree 时上一网络尚未读完")
                pending_degree = int(degree.group("degree"))
                if pending_degree <= 0:
                    raise ValueError("NetDegree 必须为正")
                pending_pins = []
                continue

            if pending_degree is None:
                raise ValueError(f"网络引脚行之前缺少 NetDegree: {line}")
            pending_pins.append(line.split()[0])
            if len(pending_pins) == pending_degree:
                raw_nets.append(pending_pins)
                pending_degree = None
                pending_pins = []

    if pending_degree is not None:
        raise ValueError("文件结束时最后一个网络的引脚数不足")
    if declared_nets is not None and len(raw_nets) != declared_nets:
        raise ValueError(f"网络数不一致: 声明 {declared_nets}, 实际 {len(raw_nets)}")
    actual_pins = sum(len(net) for net in raw_nets)
    if declared_pins is not None and actual_pins != declared_pins:
        raise ValueError(f"引脚数不一致: 声明 {declared_pins}, 实际 {actual_pins}")

    encoded_nets: list[np.ndarray] = []
    block_to_nets: list[list[int]] = [[] for _ in range(blocks.n)]
    for net_id, pin_names in enumerate(raw_nets):
        vertices: list[int] = []
        for name in pin_names:
            if name in block_index:
                vertex = int(block_index[name])
                block_to_nets[vertex].append(net_id)
            elif name in terminal_index:
                vertex = blocks.n + int(terminal_index[name])
            else:
                raise ValueError(f"网络 {net_id} 引用了未知引脚 {name}")
            vertices.append(vertex)
        encoded_nets.append(np.asarray(vertices, dtype=np.int32))

    return NetlistData(
        nets=tuple(encoded_nets),
        terminal_names=terminal_names,
        terminal_x=terminal_x,
        terminal_y=terminal_y,
        n_blocks=blocks.n,
        block_to_nets=tuple(
            np.asarray(ids, dtype=np.int32) for ids in block_to_nets
        ),
    )
