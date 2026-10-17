"""Routing data through a simulated network.

A rule set is a network topology (undirected edge list).  The puzzle asks for
the shortest route from a source to a destination; each hop is one scratchpad
step.  Changing the topology changes the routes ("the rules change").
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Tuple

from .base import Task


class RoutingTask(Task):
    name = "routing"
    header = "Find the route through the network.\n\n"
    input_label = "Route"

    def __init__(self, rules: Dict[str, List[list]], min_hops: int = 2, max_hops: int = 3):
        super().__init__(rules)
        self.min_hops, self.max_hops = int(min_hops), int(max_hops)
        self.adj: Dict[str, Dict[str, List[str]]] = {}
        self.nodes: Dict[str, List[str]] = {}
        for r, edges in rules.items():
            adj: Dict[str, List[str]] = {}
            for a, b in edges:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            for k in adj:
                adj[k].sort()
            self.adj[r] = adj
            self.nodes[r] = sorted(adj)

    def _path(self, rule: str, src: str, dst: str) -> List[str]:
        adj = self.adj[rule]
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if dst not in prev:
            return []
        path, cur = [], dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    def sample_input(self, rule: str, rng: random.Random) -> str:
        nodes = self.nodes[rule]
        for _ in range(1000):
            a, b = rng.sample(nodes, 2)
            p = self._path(rule, a, b)
            if self.min_hops <= len(p) - 1 <= self.max_hops:
                return f"{a} to {b}"
        raise RuntimeError("could not sample a route with the requested hop count")

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        a, b = input_str.split(" to ")
        p = self._path(rule, a, b)
        if not p:
            return ["unreachable"], "unreachable"
        steps = [f"{p[i]} -> {p[i + 1]}" for i in range(len(p) - 1)]
        return steps, " ".join(p)

    def n_steps(self, rule: str) -> int:
        return self.max_hops

    def describe(self, rule: str) -> str:
        links = ", ".join(f"{a}-{b}" for a, b in self.rules[rule])
        return (f"The network links are {links}. Each step moves along one link; "
                f"use the fewest hops. The answer lists every node on the route.")

    def max_answer_tokens(self) -> int:
        return 2 * (self.max_hops + 1) + 2
