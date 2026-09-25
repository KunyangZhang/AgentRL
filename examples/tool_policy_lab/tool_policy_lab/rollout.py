from dataclasses import dataclass
import random
import torch
from .environment import ACTIONS, ToolEnv, oracle_action


@dataclass
class Episode:
    uid: str
    observations: list
    actions: list
    old_log_probs: list
    rewards: list
    trace: list
    success: bool
    fault_assigned: bool = False


@torch.no_grad()
def collect(policy, cases, shaped=True, greedy=False, baseline=None, seed=0):
    """Vectorized across active environments; terminal episodes stop producing actions."""
    rng = random.Random(seed)
    envs = [ToolEnv(c, shaped=shaped) for c in cases]
    episodes = [Episode(c.uid, [], [], [], [], [], False, any(c.failures)) for c in cases]
    while active := [i for i, e in enumerate(envs) if not e.done]:
        observations = [envs[i].observe() for i in active]
        if baseline is None:
            dist = torch.distributions.Categorical(logits=policy(observations))
            actions = dist.probs.argmax(-1) if greedy else dist.sample()
            log_probs = dist.log_prob(actions).cpu().tolist()
            actions = actions.cpu().tolist()
        else:
            actions = [ACTIONS.index(oracle_action(x)) if baseline == "oracle"
                       else rng.randrange(len(ACTIONS)) for x in observations]
            log_probs = [0.0] * len(active)
        for i, obs, act, lp in zip(active, observations, actions, log_probs):
            _, reward, _ = envs[i].step(ACTIONS[act])
            ep = episodes[i]
            ep.observations.append(obs)
            ep.actions.append(act)
            ep.old_log_probs.append(lp)
            ep.rewards.append(reward)
    for ep, env in zip(episodes, envs):
        ep.success, ep.trace = env.success, env.trace
    return episodes


def metrics(episodes):
    n = len(episodes)
    faulted = [e for e in episodes if any(t["next_observation"]["last_error"] == "temporary_unavailable" for t in e.trace)]
    assigned = [e for e in episodes if e.fault_assigned]
    invalid = sum(t["next_observation"]["last_error"] not in (None, "temporary_unavailable")
                  for e in episodes for t in e.trace)
    steps = sum(len(e.actions) for e in episodes)
    p = sum(e.success for e in episodes) / n
    # Wilson interval for a single evaluated policy, not across-seed uncertainty.
    z = 1.96
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / (1+z*z/n)
    return {"episodes": n, "success_rate": p, "success_wilson95": [center-half, center+half],
            "mean_return": sum(sum(e.rewards) for e in episodes)/n,
            "mean_tool_calls": steps/n, "invalid_action_rate": invalid/max(steps, 1),
            "fault_encountered_episodes": len(faulted),
            "fault_assigned_episodes": len(assigned),
            "fault_assigned_success_rate": sum(e.success for e in assigned)/len(assigned) if assigned else None,
            "recovery_success_rate": sum(e.success for e in faulted)/len(faulted) if faulted else None}
