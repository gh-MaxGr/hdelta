"""A minimal world whose engine scores OUTCOMES, not actions.

Every verifier in this project so far contained the answer. The logic families have a
`solve()` that returns the right label; the repo-idiom linter knew the command had to say
`Start-Process`. When the verifier already holds the answer, the model is learning to
reproduce it, which is why compiling it bought nothing a template would not.

Here the engine cannot produce the right move. It can only apply a move and score the state
that results. Verification is therefore RELATIVE: an action is accepted if it beats the BEST
of a small stratified sample of legal moves. Searching the ~400-move space stays expensive;
judging one candidate against 8 draws stays cheap. That gap is the entire reason to compile
anything, and it is why the verifier carries no policy of its own.

The three rules are DISPOSITIONS, not situations. The same offer has different right answers
for a trader, a diplomat and a hardliner, because each scores the resulting world
differently. A disposition cannot be stored as retrievable text - which is the claim this
task exists to test.

    trader     goods are what matter; standing is nearly irrelevant
    diplomat   standing is worth more than goods; takes mildly bad trades to keep the peace
    hardliner  will not trade with a hostile counterparty at all; otherwise like a trader
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Sequence, Tuple

from .base import Puzzle, Task

GOODS = ("grain", "iron", "cloth", "salt")
STANDING = ("hostile", "neutral", "friendly")
FAMILIES = ("trader", "diplomat", "hardliner")

_VALID_MOVE = re.compile(r"^(ACCEPT|REFUSE|WALK|COUNTER \d+ \w+ FOR \d+ \w+)$")

# The number to beat. Measured over 400 situations per disposition with the verifier below:
# the best CONSTANT policy, which is what a collapsed model would converge on. A compiled
# reflex scoring at or under its disposition's floor has learned nothing, however healthy
# the raw accuracy looks. The enumerated reference scores 1.00 everywhere.
CONSTANT_FLOOR = {"trader": 0.50, "diplomat": 0.70, "hardliner": 0.58}

# utility = sum(stock[g] * need[g]) + STANDING_WEIGHT * standing_index
WEIGHT = {"trader": 1.0, "diplomat": 9.0, "hardliner": 2.0}


class Situation:
    """One negotiation turn: what we hold, what we want, who is asking, and for what."""

    def __init__(self, stock, need, standing, give_n, give_g, want_n, want_g):
        self.stock, self.need, self.standing = stock, need, standing
        self.give_n, self.give_g = give_n, give_g      # what THEY give us
        self.want_n, self.want_g = want_n, want_g      # what THEY want from us

    def text(self) -> str:
        """Each good's two numbers sit ADJACENT, deliberately.

        The first encoding was `stock[grain:3 ...] need[grain:3 ...]` - two brackets with
        identical keys, requiring a cross-bracket lookup per good. Diagnosed on 72
        generations: the model consistently read the wrong bracket, saying "we value cloth
        at 3" when 3 was the stock and the need was 1. That made the task a test of parsing
        my encoding rather than of choosing a move.
        """
        holdings = "; ".join(f"{g} have {self.stock[g]} worth {self.need[g]}" for g in GOODS)
        return (f"{holdings} | standing {self.standing} | "
                f"they give {self.give_n} {self.give_g} for {self.want_n} {self.want_g}")

    @staticmethod
    def parse(s: str) -> "Situation":
        stock, need = {}, {}
        for g, have, worth in re.findall(r"(\w+) have (\d+) worth (\d+)", s):
            stock[g], need[g] = int(have), int(worth)
        st = re.search(r"standing (\w+)", s).group(1)
        m = re.search(r"they give (\d+) (\w+) for (\d+) (\w+)", s)
        return Situation(stock, need, st, int(m.group(1)), m.group(2),
                         int(m.group(3)), m.group(4))


def utility(disp: str, stock: Dict[str, int], need: Dict[str, int], standing: str) -> float:
    return (sum(stock[g] * need[g] for g in GOODS)
            + WEIGHT[disp] * STANDING.index(standing))


def legal_moves(sit: Situation) -> List[str]:
    """ACCEPT / REFUSE / WALK plus every counter-offer we could actually cover."""
    out = ["ACCEPT", "REFUSE", "WALK"]
    for g in GOODS:
        for n in (1, 2, 3):
            if sit.stock[g] >= n:
                for gn in (1, 2, 3):
                    out.append(f"COUNTER {n} {g} FOR {gn} {sit.give_g}")
    return out


def shift(standing: str, d: int) -> str:
    return STANDING[max(0, min(len(STANDING) - 1, STANDING.index(standing) + d))]


def apply_move(sit: Situation, move: str) -> Tuple[Dict[str, int], str, bool]:
    """Resolve a move deterministically -> (new stock, new standing, was it legal).

    The counterparty is a fixed rule, not a model: it accepts a counter whose value to it
    is at least as good as its own opening ask, and sours slightly when it is haggled.
    """
    stock = dict(sit.stock)
    m = move.strip().upper()
    if m.startswith("ACCEPT"):
        if stock[sit.want_g] < sit.want_n:
            return stock, shift(sit.standing, -1), False      # promised what we do not have
        stock[sit.give_g] += sit.give_n
        stock[sit.want_g] -= sit.want_n
        return stock, shift(sit.standing, +1), True
    if m.startswith("REFUSE"):
        return stock, shift(sit.standing, -1), True
    if m.startswith("WALK"):
        return stock, shift(sit.standing, -2), True
    mm = re.match(r"COUNTER (\d+) (\w+) FOR (\d+) (\w+)", m)
    if not mm:
        return stock, shift(sit.standing, -1), False
    give_n, give_g = int(mm.group(1)), mm.group(2).lower()
    want_n, want_g = int(mm.group(3)), mm.group(4).lower()
    if give_g not in GOODS or want_g not in GOODS or stock.get(give_g, 0) < give_n:
        return stock, shift(sit.standing, -1), False
    # the counterparty's own valuation: it opened wanting want_n of want_g for give_n of give_g
    theirs_open = sit.want_n
    theirs_now = give_n if give_g == sit.want_g else give_n * 0.5
    if theirs_now >= theirs_open * 0.75 and want_g == sit.give_g:
        stock[give_g] -= give_n
        stock[want_g] += want_n
        return stock, sit.standing, True
    return stock, shift(sit.standing, -1), True               # haggled and rebuffed


def score(disp: str, sit: Situation, move: str) -> float:
    stock, standing, ok = apply_move(sit, move)
    u = utility(disp, stock, sit.need, standing)
    if not ok:
        u -= 5.0                                              # illegal or undeliverable
    if disp == "hardliner" and sit.standing == "hostile" and move.strip().upper().startswith(
            ("ACCEPT", "COUNTER")):
        u -= 12.0                                             # will not deal with hostiles
    return u


class TradeWorld(Task):
    name = "world"
    header = "You are a faction negotiator. Reply with one move.\n\n"
    input_label = "Situation"
    answer_label = "Move"
    rule_noun = "disposition"

    def __init__(self, rules: Dict[str, dict], tagged: bool = False, tag_map=None,
                 baseline_n: int = 8, split: str = "train"):
        super().__init__(rules)
        self.tagged, self.tag_map = bool(tagged), dict(tag_map) if tag_map else None
        self.baseline_n, self.split = int(baseline_n), split
        self.spec = {}
        for r, s in (rules or {}).items():
            fam = (s or {}).get("family", "trader")
            if fam not in FAMILIES:
                raise ValueError(f"unknown disposition {fam}")
            self.spec[r] = {"family": fam}

    # ---- situations -----------------------------------------------------------
    def sample_input(self, rule: str, rng: random.Random) -> str:
        """Draw a situation.

        No train/held-out pool split here, deliberately. The situation space is ~6.7M
        combinations (stock 256 x need 81 x standing 3 x offer 108), so against 1080 stream
        puzzles a probe item is essentially never one the stream has seen - unlike the
        repo-idiom task, where 14 possible inputs against 360 puzzles made every probe a
        memorised one. Splitting the ranges instead (train 1-3, probe 2-4) would buy nothing
        and would confound generalisation with a distribution shift in stock levels.
        """
        stock = {g: rng.randint(1, 4) for g in GOODS}
        need = {g: rng.randint(1, 3) for g in GOODS}
        gg, wg = rng.sample(GOODS, 2)
        return Situation(stock, need, rng.choice(STANDING),
                         rng.randint(1, 3), gg, rng.randint(1, 3), wg).text()

    # ---- a reference policy, NOT ground truth ---------------------------------
    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        """The best move this disposition can find by enumeration.

        Used for few-shot demonstrations and for the deliberate path. It is deliberately
        NOT what `matches` scores against: if correctness meant 'equals this', the verifier
        would contain the answer and we would be back to reproducing a template.
        """
        disp = self.spec[rule]["family"]
        sit = Situation.parse(input_str)
        best = max(legal_moves(sit), key=lambda m: score(disp, sit, m))
        gain = sit.give_n * sit.need[sit.give_g]
        lose = sit.want_n * sit.need[sit.want_g]
        steps = [f"they give {sit.give_n} {sit.give_g} worth {sit.need[sit.give_g]} each "
                 f"= {gain}; we give {sit.want_n} {sit.want_g} worth "
                 f"{sit.need[sit.want_g]} each = {lose}",
                 f"goods change by {gain - lose:+d}, and standing is {sit.standing}",
                 f"as a {disp} a step of standing is worth {WEIGHT[disp]:g} goods"]
        return steps, best

    def n_steps(self, rule: str) -> int:
        return 3

    def max_answer_tokens(self) -> int:
        return 24

    def describe(self, rule: str) -> str:
        disp = self.spec[rule]["family"]
        common = ("Moves are ACCEPT, REFUSE, WALK, or COUNTER <n> <good> FOR <n> <good>. "
                  "Accepting trades away what they want and gains what they give, and "
                  "improves standing; refusing or walking worsens it.")
        if disp == "trader":
            return common + (" You care almost only about goods: take any trade whose "
                             "incoming value beats what you give up, and do not pay for "
                             "goodwill.")
        if disp == "diplomat":
            return common + (" Standing is worth far more to you than goods: keep the "
                             "relationship warm even at a mild loss, and avoid walking.")
        return common + (" You will not deal with a hostile counterparty at all - refuse "
                         "them outright. Otherwise weigh goods as a trader would.")

    # ---- outcome verification --------------------------------------------------
    def evaluate(self, rule: str, move: str, input_str: str,
                 rng: random.Random | None = None) -> Tuple[bool, float, str]:
        """Accept a move iff it beats the BEST of a small stratified sample of legal moves.

        The engine still never computes the best move - it ranks one candidate against a
        handful of draws. Judging stays cheap (8 evaluations) while searching the ~400-move
        space stays expensive, and that gap is the only reason compiling anything pays.

        Three details, each of which a weaker version got wrong:

        - The sample is STRATIFIED. Uniform draws are dominated by bad counter-offers, so
          the median of a uniform sample is a floor-level bar that `always REFUSE` cleared
          90% of the time. Forcing ACCEPT/REFUSE/WALK into the sample makes the bar the
          obvious moves rather than the junk ones.
        - The bar is the MAX, not the median. Measured against constant policies, max caps
          them at 0.41 / 0.65 / 0.54 for trader / diplomat / hardliner, against 1.00 for the
          enumerated reference - real headroom that collapsing cannot reach.
        - Unparseable output fails outright. A garbage string used to score 0.33 for the
          hardliner, because failing to act looks like refusing to act.

        Read any result against `CONSTANT_FLOOR`, never against random. A disposition that
        scores below its own best constant policy has collapsed, whatever the raw number.
        """
        disp = self.spec[rule]["family"]
        if not _VALID_MOVE.match(move.strip().upper()):
            return False, -99.0, "unparseable"
        sit = Situation.parse(input_str)
        rng = rng or random.Random(hash(input_str) & 0xFFFF)
        counters = [m for m in legal_moves(sit) if m.startswith("COUNTER")]
        sample = ["ACCEPT", "REFUSE", "WALK"]
        sample += [rng.choice(counters) for _ in range(max(0, self.baseline_n - 3))]
        bar = max(score(disp, sit, m) for m in sample)
        got = score(disp, sit, move)
        return got >= bar, got, f"{got:.1f} vs bar {bar:.1f}"

    def contrast(self, rule: str, move: str, input_str: str,
                 rng: random.Random | None = None):
        """(advantage, a better sampled move or None) - the signal a signed write needs.

        `advantage` is the move's score minus the mean of the sampled alternatives: positive
        means reinforce, negative means damage, near zero means the outcome was uninformative
        and should cost no capacity.

        The second element is the best of the SAMPLED alternatives when it beat this move.
        That is not oracle supervision - it is never the enumerated optimum, only something
        that happened to do better - so writing toward it stays contrastive. It matters
        because pure punishment says where not to go and not where to go, and mass pushed off
        a bad move lands on whatever was next most likely, which may be worse.
        """
        disp = self.spec[rule]["family"]
        if not _VALID_MOVE.match(move.strip().upper()):
            # (None, None) means DO NOT WRITE - not "write a large negative".
            #
            # Measured 2026-09-22: treating unparseable output as a catastrophically bad
            # decision (-99) created a runaway. Once the model slipped off the valid-move
            # manifold it took an enormous negative write, which pushed it further off, which
            # produced more unparseable output - both signed runs went from ~1 % invalid to
            # 100 % invalid within two fifths of the stream and trained on garbage thereafter.
            #
            # The advantage was NOT mis-centred before that: 38-46 % negative in the early
            # segments, median advantage +0.3 to +0.8. The collapse came from this penalty,
            # not from the baseline.
            #
            # And punishing it was never meaningful. An unparseable string is not a bad
            # choice, it is the absence of one, so the gradient lands on arbitrary tokens at
            # ~40x the magnitude of a real policy error. There is nothing there to learn from.
            return None, None
        sit = Situation.parse(input_str)
        rng = rng or random.Random(hash(input_str) & 0xFFFF)
        counters = [m for m in legal_moves(sit) if m.startswith("COUNTER")]
        sample = ["ACCEPT", "REFUSE", "WALK"]
        sample += [rng.choice(counters) for _ in range(max(0, self.baseline_n - 3))]
        scored = [(score(disp, sit, m), m) for m in sample]
        got = score(disp, sit, move)
        adv = got - sum(s for s, _ in scored) / len(scored)
        best_s, best_m = max(scored)
        return adv, (best_m if best_s > got else None)

    def matches(self, generated: str, answer: str, rule: str,
                input_str: str | None = None) -> bool:
        """Correctness is 'this move beat the baseline', not 'it equals the reference'.

        `answer` is ignored on purpose. It holds the enumerated best move, and scoring
        against it would put the policy back inside the verifier - the exact failure that
        made the repo-idiom result hollow.
        """
        if input_str is None:
            return generated.strip().upper() == answer.strip().upper()
        return self.evaluate(rule, generated, input_str)[0]

    def parse_generation(self, text: str) -> Tuple[str, int]:
        steps = 0
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                continue
            if s.startswith("Step"):
                steps += 1
                continue
            if s.startswith(f"{self.answer_label}:"):
                s = s[len(self.answer_label) + 1:].strip()
            m = re.match(r"(ACCEPT|REFUSE|WALK|COUNTER \d+ \w+ FOR \d+ \w+)", s.upper())
            return (m.group(1) if m else s), steps
        return "", steps
