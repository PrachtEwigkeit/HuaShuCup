from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bstar_pack import pack_bstar
from .data import BlockData
from .objective import annealing_delta, better, evaluate
from .operators import perturb
from .structures import BStarTreeState, Layout, SAHistory, Score


@dataclass(frozen=True)
class SAConfig:
    initial_temperature: float = 0.05
    final_temperature: float = 0.00005
    alpha: float = 0.90
    moves_per_temp_factor: float = 2.0
    max_stagnant_levels: int = 25
    shape_scale: float = 0.01
    p_rotate: float = 0.30
    p_swap: float = 0.35
    p_move: float = 0.35


def simulated_annealing(
    blocks: BlockData,
    init_state: BStarTreeState,
    config: SAConfig,
    rng: np.random.Generator,
) -> tuple[BStarTreeState, Layout, Score, SAHistory]:
    current = init_state.copy()
    current_layout = pack_bstar(blocks, current)
    current_score = evaluate(current_layout)

    best = current.copy()
    best_layout = current_layout
    best_score = current_score

    history = SAHistory([], [], [], [], [])
    iteration = 0
    T = float(config.initial_temperature)
    moves_per_temp = max(1, int(round(config.moves_per_temp_factor * blocks.n)))
    stagnant_levels = 0

    while T > config.final_temperature and stagnant_levels < config.max_stagnant_levels:
        improved_this_level = False

        for _ in range(moves_per_temp):
            iteration += 1
            cand = perturb(
                current,
                rng,
                config.p_rotate,
                config.p_swap,
                config.p_move,
            )
            cand_layout = pack_bstar(blocks, cand)
            cand_score = evaluate(cand_layout)

            delta = annealing_delta(
                current_score,
                cand_score,
                blocks.total_area,
                config.shape_scale,
            )

            if delta <= 0.0:
                accept = True
            else:
                exponent = -delta / max(T, 1e-15)
                accept = rng.random() < math.exp(max(exponent, -700.0))

            if accept:
                current = cand
                current_layout = cand_layout
                current_score = cand_score

            if better(current_score, best_score):
                best = current.copy()
                best_layout = current_layout
                best_score = current_score
                improved_this_level = True

            history.iteration.append(iteration)
            history.current_area.append(current_score.area)
            history.best_area.append(best_score.area)
            history.current_aspect.append(current_score.aspect)
            history.best_aspect.append(best_score.aspect)

        stagnant_levels = 0 if improved_this_level else stagnant_levels + 1
        T *= config.alpha

    return best, best_layout, best_score, history
