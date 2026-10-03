"""Example learning plugin: an epsilon-greedy bandit that tunes a threshold.

Shows the pattern for any reinforcement-learning style node:
  * the node keeps its learning state in ``self.state`` (persisted with the solution),
  * it receives a reward from the previous run (via a global variable or a direct link),
  * it emits an action (here: the threshold value) that downstream nodes consume.

Wire it like this in a flow:
  GetVariable("reward") -> BanditThresholdTuner.reward
  BanditThresholdTuner.threshold -> Threshold.thresh
  ... -> Blob Analysis -> Expression("1.0 if a == 3 else 0.0") -> SetVariable("reward")
"""
from __future__ import annotations

import random

from cvflow.core import DataType, Node, Param, Port


class BanditThresholdTuner(Node):
    type_id = "learning.bandit_threshold"
    category = "Learning"
    label = "Bandit Threshold Tuner"
    description = "Epsilon-greedy bandit choosing the threshold with the highest running reward."
    color = "#c62828"
    inputs = [Port("reward", DataType.FLOAT, optional=True, description="Reward for the previous action")]
    outputs = [Port("threshold", DataType.INT), Port("best", DataType.INT), Port("explore", DataType.BOOL),
               Port("q_values", DataType.DICT)]
    params = [Param("low", 60, "int", min=0, max=255), Param("high", 200, "int", min=0, max=255),
              Param("step", 10, "int", min=1, max=128),
              Param("epsilon", 0.1, "float", min=0, max=1, step=0.01),
              Param("learning_rate", 0.2, "float", min=0.001, max=1, step=0.01),
              Param("frozen", False, "bool", description="Stop learning, always use the best arm")]

    def _arms(self) -> list[int]:
        return list(range(int(self.get("low")), int(self.get("high")) + 1, int(self.get("step"))))

    def process(self, ctx, inputs):
        arms = self._arms()
        q: dict = self.state.setdefault("q", {})
        n: dict = self.state.setdefault("n", {})
        for a in arms:
            q.setdefault(str(a), 0.0)
            n.setdefault(str(a), 0)
        reward = inputs.get("reward")
        last = self.state.get("last_arm")
        if reward is not None and last is not None and not self.get("frozen"):
            k = str(last)
            n[k] += 1
            q[k] += float(self.get("learning_rate")) * (float(reward) - q[k])
        best = max(arms, key=lambda a: q[str(a)])
        explore = (not self.get("frozen")) and random.random() < float(self.get("epsilon"))
        action = random.choice(arms) if explore else best
        self.state["last_arm"] = action
        ctx.log(f"threshold={action} best={best} q={q[str(best)]:.2f} explore={explore}", "debug")
        return {"threshold": int(action), "best": int(best), "explore": explore, "q_values": dict(q)}
