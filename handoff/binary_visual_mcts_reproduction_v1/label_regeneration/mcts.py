"""Unrestricted graph MCTS for complete binary visual-routing masks."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable


Mask = tuple[int, ...]
PartialKey = tuple[Mask, tuple[bool, ...]]


def mask_key(mask: Mask) -> str:
    return "".join(str(int(value)) for value in mask)


@dataclass(frozen=True)
class MCTSConfig:
    num_layers: int = 28
    exploration_constant: float = 1.8
    length_penalty: float = 3.0
    random_probability: float = 0.1
    rollout_off_probability: float = 0.5
    seed: int = 20260810


@dataclass
class GraphNode:
    node_id: int
    mask: Mask
    decided: tuple[bool, ...]
    visits: int = 0
    total_reward: float = 0.0
    children: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return all(self.decided)

    @property
    def mean_reward(self) -> float:
        return 0.0 if self.visits == 0 else self.total_reward / self.visits

    def available_actions(self) -> list[tuple[int, int]]:
        return [
            (layer, action)
            for layer, is_decided in enumerate(self.decided)
            if not is_decided
            for action in (0, 1)
        ]


class GraphMCTS:
    """Graph MCTS preserving the documented v2 unordered layer/action search."""

    def __init__(self, evaluate: Callable[[Mask, str], dict], config: MCTSConfig):
        self.evaluate = evaluate
        self.config = config
        self.rng = random.Random(config.seed)
        root_mask = (1,) * config.num_layers
        root_decided = (False,) * config.num_layers
        self.nodes: list[GraphNode] = [GraphNode(0, root_mask, root_decided)]
        self.transpositions: dict[PartialKey, int] = {(root_mask, root_decided): 0}
        self.evaluations: dict[Mask, dict] = {}
        self.simulations: list[dict] = []
        self.transposition_hits = 0

    def _evaluate_mask(self, mask: Mask, source: str) -> tuple[dict, bool]:
        if mask in self.evaluations:
            return self.evaluations[mask], True
        result = self.evaluate(mask, source)
        self.evaluations[mask] = result
        return result, False

    def evaluate_anchors(self) -> tuple[dict, dict]:
        root, _ = self._evaluate_mask((1,) * self.config.num_layers, "all_on_root")
        all_off, _ = self._evaluate_mask((0,) * self.config.num_layers, "all_off_anchor")
        return root, all_off

    def _ucb(self, parent: GraphNode, child: GraphNode) -> float:
        if child.visits == 0:
            return float("inf")
        exploration = self.config.exploration_constant * math.sqrt(
            math.log(max(1, parent.visits)) / child.visits
        )
        length = sum(child.mask) / self.config.num_layers
        return child.mean_reward + exploration - self.config.length_penalty * length

    def _select_child(self, node: GraphNode) -> GraphNode:
        child_ids = list(node.children.values())
        if self.rng.random() < self.config.random_probability:
            return self.nodes[self.rng.choice(child_ids)]
        return max(
            (self.nodes[index] for index in child_ids),
            key=lambda child: (self._ucb(node, child), -child.node_id),
        )

    def _expand(self, parent: GraphNode) -> tuple[GraphNode, tuple[int, int], list[dict], bool]:
        available = [action for action in parent.available_actions() if action not in parent.children]
        self.rng.shuffle(available)
        if not available:
            raise RuntimeError("attempted to expand a fully expanded node")
        action = self.rng.choice(available)
        layer, value = action
        next_mask = list(parent.mask)
        next_decided = list(parent.decided)
        next_mask[layer] = value
        next_decided[layer] = True
        key: PartialKey = (tuple(next_mask), tuple(next_decided))
        reused = key in self.transpositions
        if reused:
            child = self.nodes[self.transpositions[key]]
            self.transposition_hits += 1
        else:
            child = GraphNode(len(self.nodes), key[0], key[1])
            self.nodes.append(child)
            self.transpositions[key] = child.node_id
        parent.children[action] = child.node_id
        candidates = [
            {"layer_zero_based": int(layer_index), "visual_action": int(action_value)}
            for layer_index, action_value in available
        ]
        return child, action, candidates, reused

    def _rollout(self, node: GraphNode) -> tuple[Mask, list[int]]:
        mask = list(node.mask)
        removed = []
        for layer, decided in enumerate(node.decided):
            if not decided:
                mask[layer] = int(self.rng.random() >= self.config.rollout_off_probability)
                if mask[layer] == 0:
                    removed.append(layer)
        return tuple(mask), removed

    def run(self, total_simulations: int) -> None:
        if total_simulations < len(self.simulations):
            raise ValueError("cannot reduce completed simulation count")
        while len(self.simulations) < total_simulations:
            path = [self.nodes[0]]
            current = path[0]
            while not current.terminal:
                unexpanded = [
                    action for action in current.available_actions() if action not in current.children
                ]
                if unexpanded:
                    parent = current
                    current, action, candidates, transposition_reused = self._expand(parent)
                    path.append(current)
                    break
                current = self._select_child(current)
                path.append(current)
            else:
                parent = path[-2] if len(path) > 1 else current
                action = (-1, -1)
                candidates = []
                transposition_reused = False

            completed_mask, rollout_removed = self._rollout(current)
            result, evaluation_reused = self._evaluate_mask(completed_mask, "mcts_rollout")
            reward = float(result["reward"])
            for node in path:
                node.visits += 1
                node.total_reward += reward
            layer, value = action
            self.simulations.append(
                {
                    "simulation": len(self.simulations) + 1,
                    "selected_node_ids": [node.node_id for node in path],
                    "expanded_parent_node_id": parent.node_id,
                    "expanded_node_id": current.node_id,
                    "expanded_layer_zero_based": None if layer < 0 else layer,
                    "expanded_layer_one_based": None if layer < 0 else layer + 1,
                    "expanded_visual_action": None if value < 0 else value,
                    "expanded_visual_action_name": None
                    if value < 0
                    else ("visual_on" if value else "visual_off"),
                    "expansion_candidates": candidates,
                    "expansion_reused_transposition": transposition_reused,
                    "rollout_removed_layers_zero_based": rollout_removed,
                    "evaluated_mask": list(completed_mask),
                    "evaluated_mask_key": mask_key(completed_mask),
                    "num_visual_on_layers": int(sum(completed_mask)),
                    "evaluation_reused": evaluation_reused,
                    "reward": reward,
                }
            )

    def result(self, *, requested_simulations: int, extension_reason: str | None) -> dict:
        successful = sorted(
            (mask for mask, row in self.evaluations.items() if bool(row["result_correct"])),
            key=lambda mask: (sum(mask), mask_key(mask)),
        )
        evaluated = [
            {
                "visual_on_mask": list(mask),
                "mask_key": mask_key(mask),
                "reward": float(row["reward"]),
                "score": float(row["score"]),
                "result_correct": bool(row["result_correct"]),
                "route_id": row["route_id"],
            }
            for mask, row in self.evaluations.items()
        ]
        graph_nodes = [
            {
                "node_id": node.node_id,
                "partial_visual_on_mask": list(node.mask),
                "decided_layers_zero_based": [
                    index for index, decided in enumerate(node.decided) if decided
                ],
                "visits": node.visits,
                "total_reward": node.total_reward,
                "mean_reward": node.mean_reward,
                "children": [
                    {
                        "layer_zero_based": layer,
                        "visual_action": action,
                        "child_node_id": child_id,
                    }
                    for (layer, action), child_id in sorted(node.children.items())
                ],
            }
            for node in self.nodes
        ]
        root = self.evaluations[(1,) * self.config.num_layers]
        all_off = self.evaluations[(0,) * self.config.num_layers]
        return {
            "root_reward": float(root["reward"]),
            "all_off_reward": float(all_off["reward"]),
            "successful_masks": [list(mask) for mask in successful],
            "best_mask": None if not successful else list(successful[0]),
            "evaluated_masks": evaluated,
            "simulations": self.simulations,
            "graph_nodes": graph_nodes,
            "transposition_hits": self.transposition_hits,
            "expansion_policy": "choose_layer_and_visual_on_off_from_all_undecided_layers",
            "requested_simulations": requested_simulations,
            "completed_simulations": len(self.simulations),
            "extension_reason": extension_reason,
        }
