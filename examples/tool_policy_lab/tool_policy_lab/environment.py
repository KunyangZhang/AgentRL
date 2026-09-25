"""A deterministic tool service with pagination-like sequential dependencies.

The benchmark is deliberately synthetic. It measures dispatch, ordering and
retry decisions, not mathematical reasoning or general agent intelligence.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import random

ACTIONS = ("fetch", "add", "subtract", "multiply", "submit")
OPS = ACTIONS[1:4]
SPLIT_SEEDS = {"train": 11001, "dev": 22002, "test": 33003, "stress": 44004}


@dataclass(frozen=True)
class Case:
    uid: str
    initial: int
    program: tuple[tuple[str, int], ...]
    failures: tuple[int, ...]

    @property
    def digest(self):
        # Excludes uid: detect actual content overlap, not merely ID overlap.
        raw = json.dumps([self.initial, self.program, self.failures])
        return hashlib.sha256(raw.encode()).hexdigest()


def make_cases(split, count=256):
    rng = random.Random(SPLIT_SEEDS[split])
    cases, seen = [], set()
    while len(cases) < count:
        n = rng.randint(4, 6) if split == "stress" else rng.randint(1, 3)
        case = Case(f"{split}-{len(cases):05d}", rng.randint(-10000, 10000),
                    tuple((rng.choice(OPS), rng.randint(1, 20)) for _ in range(n)),
                    tuple(rng.choice((0, 0, 1, 2) if split == "stress" else (0, 0, 0, 1))
                          for _ in range(n + 1)))
        if case.digest not in seen:
            cases.append(case)
            seen.add(case.digest)
    return cases


class ToolEnv:
    def __init__(self, case, shaped=True):
        self.case, self.shaped = case, shaped
        self.cursor, self.value, self.turn = 0, None, 0
        self.attempts = [0] * (len(case.program) + 1)
        self.done, self.success, self.conformant = False, False, True
        self.last_error = None
        self.trace = []
        self.max_turns = 3 * len(case.program) + 7

    def observe(self):
        return {"loaded": self.value is not None, "value": self.value,
                "step": self.cursor, "total_steps": len(self.case.program),
                "instruction": (dict(zip(("operation", "operand"), self.case.program[self.cursor]))
                                if self.cursor < len(self.case.program) else None),
                "last_error": self.last_error, "remaining_calls": self.max_turns - self.turn}

    def features(self):
        obs = self.observe()
        op = obs["instruction"]["operation"] if obs["instruction"] else None
        return [float(obs["loaded"]), float(op is None),
                *[float(op == x) for x in OPS],
                float(obs["last_error"] == "temporary_unavailable"),
                float(obs["last_error"] not in (None, "temporary_unavailable"))]

    def step(self, action):
        if self.done:
            raise RuntimeError("cannot step a terminated episode")
        before = self.observe()
        self.turn += 1
        self.last_error = None
        reward = -0.01 if self.shaped else 0.0
        progress = False
        if action not in ACTIONS:
            self.last_error = "unknown_tool"
        elif action == "submit":
            expected = self.case.initial
            for op, operand in self.case.program:
                expected = apply_op(op, expected, operand)
            self.success = (self.cursor == len(self.case.program) and self.conformant
                            and self.value == expected)
            reward += 1.0 if self.success else -0.3
            self.done = True
        elif action == "fetch" and self.value is not None:
            self.last_error = "already_loaded"
        elif action != "fetch" and self.value is None:
            self.last_error = "fetch_required"
        elif action != "fetch" and self.cursor == len(self.case.program):
            self.last_error = "program_complete"
        else:
            stage = 0 if action == "fetch" else self.cursor + 1
            # Failure happens BEFORE mutation. Retrying does not double-apply an op.
            if self.attempts[stage] < self.case.failures[stage]:
                self.attempts[stage] += 1
                self.last_error = "temporary_unavailable"
            elif action == "fetch":
                self.value = self.case.initial
                progress = True
            else:
                op, operand = self.case.program[self.cursor]
                self.value = apply_op(action, self.value, operand)
                self.cursor += 1
                progress = action == op
                self.conformant &= progress
                if not progress:
                    self.last_error = "wrong_operation"
        if self.shaped:
            reward += 0.2 / (len(self.case.program) + 1) if progress else 0
            # Exogenous service failures are not the policy's fault.
            if self.last_error and self.last_error != "temporary_unavailable":
                reward -= 0.1
        if self.turn >= self.max_turns and not self.done:
            self.done = True
            reward -= 0.3
            self.last_error = "call_budget_exhausted"
        self.trace.append({"observation": before, "action": action,
                           "reward": reward, "next_observation": self.observe(),
                           "done": self.done})
        return self.observe(), reward, self.done


def apply_op(op, a, b):
    return {"add": lambda: a + b, "subtract": lambda: a - b,
            "multiply": lambda: a * b}[op]()


def oracle_action(obs):
    if not obs["loaded"]:
        return "fetch"
    return obs["instruction"]["operation"] if obs["instruction"] else "submit"


def dataset_manifest(count=256):
    splits = {s: make_cases(s, count) for s in SPLIT_SEEDS}
    hashes = {s: {c.digest for c in cases} for s, cases in splits.items()}
    for a in hashes:
        for b in hashes:
            if a != b and hashes[a] & hashes[b]:
                raise ValueError(f"content leakage between {a} and {b}")
    return {s: {"count": len(cases), "seed": SPLIT_SEEDS[s],
                "cases": [{**asdict(c), "sha256": c.digest} for c in cases]}
            for s, cases in splits.items()}
