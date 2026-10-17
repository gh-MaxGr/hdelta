"""Connect-Four position VALUE on a small board: a task a reason cache cannot solve.

Every earlier task was a small rule table (10 digits, 24 circumstances, 32 workspace rules),
and a cache of the tool's reasons solved each with 24-32 tool calls and no model at all. So
"the reflex generalises to unseen inputs" has only ever meant "it learned to ignore the
irrelevant fields". This task is built so that shortcut does not exist:

  - the input space is large (millions of positions on a 5x4 board), and the train/held-out
    split is disjoint BY CONSTRUCTION (a stable hash of the mirror-canonical position), so a
    held-out position has never been seen, nor has its mirror image;
  - the answer - win / draw / loss for the side to move under perfect play - depends on the
    whole board. Only the tactical cases have a short checkable reason ("you can complete a
    line at column 3 now"); everything else is a search result with no compact condition.

The tool is an exact solver (negamax over bitboards, memoised). It is teacher and verifier.
Positions are shown from the MOVER's side: the player to move is always X, so the value is a
function of the pattern on the board alone.

    Position (X to move):
    | . . . . . |
    | . . O . . |
    | . X X O . |
    | O X O X . |
    Value: win
"""
from __future__ import annotations

import hashlib
import random
import sys
from typing import Dict, List, Tuple

from .base import Task

LABELS = ("win", "draw", "loss")


class Board:
    """Bitboards in Pascal Pons' layout: H+1 bits per column (one sentinel bit on top).
    `cur` = stones of the player to move, `mask` = all stones. Connect four only."""

    def __init__(self, width: int = 5, height: int = 4):
        self.W, self.H = int(width), int(height)
        H1 = self.H + 1
        self.bottom = [1 << (c * H1) for c in range(self.W)]
        self.top = [1 << (self.H - 1 + c * H1) for c in range(self.W)]
        self.col = [((1 << self.H) - 1) << (c * H1) for c in range(self.W)]
        centre = (self.W - 1) / 2
        self.order = sorted(range(self.W), key=lambda c: abs(c - centre))
        self.cells = self.W * self.H

    # ---- moves -----------------------------------------------------------------------
    def can_play(self, mask: int, c: int) -> bool:
        return not (mask & self.top[c])

    def play(self, cur: int, mask: int, c: int) -> Tuple[int, int]:
        """Returns the position with the OTHER player to move."""
        return cur ^ mask, mask | (mask + self.bottom[c])

    def aligned(self, p: int) -> bool:
        H = self.H
        for s in (1, H + 1, H, H + 2):              # vertical, horizontal, two diagonals
            m = p & (p >> s)
            if m & (m >> (2 * s)):
                return True
        return False

    def wins_by(self, cur: int, mask: int, c: int) -> bool:
        """Does the mover complete four by dropping into column c?"""
        stone = (mask + self.bottom[c]) & self.col[c]
        return self.aligned(cur | stone)

    def winning_moves(self, cur: int, mask: int) -> List[int]:
        return [c for c in range(self.W) if self.can_play(mask, c) and self.wins_by(cur, mask, c)]

    def n_stones(self, mask: int) -> int:
        return bin(mask).count("1")

    # ---- text --------------------------------------------------------------------------
    def to_str(self, cur: int, mask: int) -> str:
        """Rows top -> bottom joined by '/'; X = the mover, O = the opponent."""
        rows = []
        for r in range(self.H - 1, -1, -1):
            row = []
            for c in range(self.W):
                b = 1 << (c * (self.H + 1) + r)
                row.append("X" if cur & b else ("O" if mask & b else "."))
            rows.append("".join(row))
        return "/".join(rows)

    def from_str(self, s: str) -> Tuple[int, int]:
        rows = s.split("/")
        assert len(rows) == self.H and all(len(r) == self.W for r in rows), s
        cur = mask = 0
        for i, row in enumerate(rows):
            r = self.H - 1 - i
            for c, ch in enumerate(row):
                b = 1 << (c * (self.H + 1) + r)
                if ch != ".":
                    mask |= b
                    if ch == "X":
                        cur |= b
        return cur, mask

    @staticmethod
    def mirror(s: str) -> str:
        return "/".join(r[::-1] for r in s.split("/"))


class Solver:
    """Exact win/draw/loss for the side to move. Memo shared across queries; `calls` counts
    queries (the tool calls), `nodes` the positions it had to search."""

    def __init__(self, board: Board):
        self.b = board
        self.memo: Dict[int, int] = {}
        self.calls = 0
        self.nodes = 0
        sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

    def value(self, cur: int, mask: int) -> int:
        b = self.b
        key = cur + mask
        v = self.memo.get(key)
        if v is not None:
            return v
        self.nodes += 1
        if b.n_stones(mask) == b.cells:
            v = 0
        elif b.winning_moves(cur, mask):
            v = 1
        else:
            v = -1
            for c in b.order:
                if b.can_play(mask, c):
                    w = -self.value(*b.play(cur, mask, c))
                    if w > v:
                        v = w
                        if v == 1:
                            break
        self.memo[key] = v
        return v

    def solve(self, s: str) -> Tuple[List[str], str]:
        """The tool: an answer plus the reason it can give. Only the tactical cases have a
        reason that names checkable conditions; the rest is a search result."""
        self.calls += 1
        b = self.b
        cur, mask = b.from_str(s)
        v = self.value(cur, mask)
        label = LABELS[1 - v]                          # 1 -> win, 0 -> draw, -1 -> loss
        wins = b.winning_moves(cur, mask)
        if wins:
            return [f"X completes four in column {wins[0] + 1}"], label
        opp = b.winning_moves(cur ^ mask, mask)
        if len(opp) >= 2:
            return [f"O threatens columns {' and '.join(str(c + 1) for c in opp)}; "
                    f"X cannot block both"], label
        best = [c + 1 for c in b.order if b.can_play(mask, c)
                and -self.value(*b.play(cur, mask, c)) == v]
        return [f"search: best column {best[0]} -> {label}"], label


class ConnectFourValueTask(Task):
    name = "connect4"
    header = "Evaluate the Connect Four position for the player to move (X), assuming perfect play.\n\n"
    input_label = "Position"
    answer_label = "Value"
    rule_noun = "board"

    def __init__(self, rules: Dict[str, dict], width: int = 5, height: int = 4,
                 plies: Tuple[int, int] = (4, 14), heldout_mod: int = 4, split: str = "train"):
        super().__init__(rules)
        self.board = Board(width, height)
        self.solver = Solver(self.board)
        self.plies = (int(plies[0]), int(plies[1]))
        self.heldout_mod = int(heldout_mod)
        self.split = split          # the experiment flips this to 'heldout' for probe sets

    # ---- the split: disjoint by construction, mirror images kept together -------------
    def bucket(self, s: str) -> str:
        canon = min(s, Board.mirror(s))
        h = int(hashlib.md5(canon.encode()).hexdigest()[:8], 16)
        return "heldout" if h % self.heldout_mod == 0 else "train"

    def random_position(self, rng: random.Random) -> str:
        """Random legal play to a random depth; never a finished game."""
        b = self.board
        while True:
            n = rng.randint(*self.plies)
            cur = mask = 0
            ok = True
            for _ in range(n):
                cols = [c for c in range(b.W) if b.can_play(mask, c)]
                c = rng.choice(cols)
                if b.wins_by(cur, mask, c):
                    ok = False
                    break
                cur, mask = b.play(cur, mask, c)
            if ok and b.n_stones(mask) < b.cells:
                return b.to_str(cur, mask)

    # ---- Task interface ---------------------------------------------------------------
    def sample_input(self, rule: str, rng: random.Random) -> str:
        split = getattr(self, "split", None)
        for _ in range(10_000):
            s = self.random_position(rng)
            if split is None or self.bucket(s) == split:
                return s
        raise RuntimeError("could not draw a position in the requested split")

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        return self.solver.solve(input_str)

    def n_steps(self, rule: str) -> int:
        return 1

    def max_answer_tokens(self) -> int:
        return 4

    def describe(self, rule: str) -> str:
        return ("Players drop stones into columns; a stone falls to the lowest empty cell. "
                "Four in a row horizontally, vertically or diagonally wins. The value is win, "
                "draw or loss for X, who moves next, if both sides play perfectly.")

    def tool_calls(self) -> int:
        return self.solver.calls

    def is_valid_answer(self, rule: str, ans: str) -> bool:
        return ans.strip() in LABELS

    def is_exception(self, rule: str, input_str: str) -> bool:
        """'Exception' here = a TACTICAL position, the only kind whose reason is a checkable
        condition (an immediate win, or a double threat that cannot be blocked)."""
        b = self.board
        cur, mask = b.from_str(input_str)
        return bool(b.winning_moves(cur, mask)) or len(b.winning_moves(cur ^ mask, mask)) >= 2

    def _input_line(self, pz) -> str:
        # Bordered rows, so every cell is the SAME token in every column: unbordered, the first
        # cell lost its leading space ('X' vs ' X') and a last-column '.' fused with the newline.
        rows = pz.input_str.split("/")
        grid = "\n".join("| " + " ".join(r) + " |" for r in rows)
        return f"{self.input_label} (X to move):\n{grid}"
