from .base import Puzzle, Task
from .cipher import CipherTask
from .routing import RoutingTask
from .logic import FAMILIES, LogicTask
from .repoidiom import RepoIdiomTask
from .world import TradeWorld
from .workspace import WorkspaceResolutionTask, WorkspaceResolver
from .connect4 import ConnectFourValueTask
from .mutants import MutantTask
from .snippets import SnippetTask
from .composite import CompositeTask


def make_task(cfg: dict) -> Task:
    name = cfg["name"]
    if name == "cipher":
        return CipherTask(cfg["rules"], length=cfg.get("length", (4, 5)),
                          alphabet_size=cfg.get("alphabet_size", 26),
                          alphabet=cfg.get("alphabet", "letters"),
                          tagged=cfg.get("tagged", False),
                          tag_map=cfg.get("tag_map"))
    if name == "routing":
        return RoutingTask(cfg["rules"], min_hops=cfg.get("min_hops", 2),
                           max_hops=cfg.get("max_hops", 3))
    if name == "workspace":
        return WorkspaceResolutionTask(cfg["rules"],
                                       n_packages=cfg.get("n_packages", 6),
                                       n_overrides=cfg.get("n_overrides", 2),
                                       depth=cfg.get("depth", (1, 2)),
                                       latency_ms=cfg.get("latency_ms", 750.0),
                                       sleep=cfg.get("sleep", False),
                                       tagged=cfg.get("tagged", False),
                                       tag_map=cfg.get("tag_map"),
                                       exception_fraction=cfg.get("exception_fraction"))
    if name == "world":
        return TradeWorld(cfg["rules"], tagged=cfg.get("tagged", False),
                          tag_map=cfg.get("tag_map"),
                          baseline_n=cfg.get("baseline_n", 8),
                          split=cfg.get("split", "train"))
    if name == "repoidiom":
        return RepoIdiomTask(cfg["rules"], tagged=cfg.get("tagged", False),
                             tag_map=cfg.get("tag_map"),
                             split=cfg.get("split", "train"))
    if name == "logic":
        return LogicTask(cfg["rules"], tagged=cfg.get("tagged", False),
                         tag_map=cfg.get("tag_map"),
                         exception_fraction=cfg.get("exception_fraction"))
    if name == "connect4":
        return ConnectFourValueTask(cfg["rules"], width=cfg.get("width", 5),
                                    height=cfg.get("height", 4),
                                    plies=tuple(cfg.get("plies", (4, 14))),
                                    heldout_mod=cfg.get("heldout_mod", 4),
                                    split=cfg.get("split", "train"))
    if name == "mutants":
        return MutantTask(cfg["rules"], data_dir=cfg.get("data_dir", "results/mutants"),
                          modules=cfg.get("modules"), heldout_mod=cfg.get("heldout_mod", 4),
                          balance=cfg.get("balance", 0.5), split=cfg.get("split", "train"),
                          context=cfg.get("context", False))
    if name == "snippets":
        return SnippetTask(cfg["rules"], data=cfg.get("data", "results/snippets.jsonl"),
                           heldout_mod=cfg.get("heldout_mod", 4), balance=cfg.get("balance", 0.5),
                           split=cfg.get("split", "train"))
    if name == "composite":
        return CompositeTask(cfg["rules"], make_task)
    raise ValueError(f"unknown task {name}")


__all__ = ["Puzzle", "Task", "CipherTask", "RoutingTask", "WorkspaceResolutionTask",
           "WorkspaceResolver", "LogicTask", "FAMILIES", "RepoIdiomTask", "TradeWorld", "ConnectFourValueTask", "MutantTask", "SnippetTask", "CompositeTask", "make_task"]
