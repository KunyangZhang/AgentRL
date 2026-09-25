import asyncio
from dataclasses import replace
import json
import pytest
import torch
from agentrl.worker.task import Session
from agentrl.worker.typings import AgentOutput, AgentOutputStatus, SampleStatus, TaskOutput
from tool_policy_lab.environment import Case, ToolEnv, dataset_manifest, make_cases, oracle_action
from tool_policy_lab.policy import SmallPolicy
from tool_policy_lab.rollout import collect, metrics
from tool_policy_lab.task import ToolPolicyTask
from tool_policy_lab.training import Config, load_policy, prepare_batch, train


@pytest.mark.parametrize("split", ["train", "dev", "test", "stress"])
def test_oracle_completes_every_case(split):
    episodes = collect(None, make_cases(split, 128), baseline="oracle")
    assert all(e.success for e in episodes)


def test_content_hash_splits_are_disjoint_and_repeatable():
    assert dataset_manifest(64) == dataset_manifest(64)
    assert make_cases("train", 32) == make_cases("train", 64)[:32]


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_fault_is_before_mutation_and_retry_does_not_duplicate(failures):
    env = ToolEnv(Case("retry", 5, (("multiply", 3),), (failures, failures)))
    for _ in range(failures):
        env.step("fetch")
        assert env.value is None
    env.step("fetch")
    for _ in range(failures):
        env.step("multiply")
        assert env.value == 5 and env.cursor == 0
    env.step("multiply")
    assert env.value == 15 and env.cursor == 1
    env.step("submit")
    assert env.success
    with pytest.raises(RuntimeError):
        env.step("multiply")


@pytest.mark.parametrize("action,error", [("add", "fetch_required"), ("nope", "unknown_tool")])
def test_invalid_actions_have_no_side_effect(action, error):
    env = ToolEnv(make_cases("test", 1)[0])
    _, reward, _ = env.step(action)
    assert env.last_error == error and reward < 0
    assert env.value is None and env.cursor == 0


def test_wrong_op_cannot_game_zero_operand_answer():
    env = ToolEnv(Case("zero", 5, (("add", 0),), (0, 0)))
    for action in ("fetch", "subtract", "submit"):
        env.step(action)
    assert env.value == 5 and not env.success


def test_budget_terminates_stuck_policy():
    env = ToolEnv(make_cases("test", 1)[0])
    while not env.done:
        env.step("nope")
    assert env.turn == env.max_turns and not env.success


def test_failure_rate_denominator_includes_unreached_faults():
    cases = [Case("assigned", 2, (("add", 3),), (1, 0))]
    episodes = collect(None, cases, baseline="oracle")
    episodes[0].success = False
    episodes[0].trace = []  # models early exit before encountering its assigned fault
    report = metrics(episodes)
    assert report["fault_encountered_episodes"] == 0
    assert report["fault_assigned_episodes"] == 1
    assert report["fault_assigned_success_rate"] == 0


def test_no_reference_answer_in_observations():
    env = ToolEnv(Case("secret", 1234, (("add", 7),), (0, 0)))
    assert "1241" not in json.dumps(env.observe())
    assert "1234" not in json.dumps(env.observe())


def test_group_advantages_and_action_only_mask():
    policy = SmallPolicy()
    episodes = collect(policy, make_cases("train", 2) * 4)
    batch = prepare_batch(episodes)
    assert len(batch["actions"]) == sum(len(e.trace) for e in episodes)
    assert len(batch["advantages"]) == len(batch["observations"])
    with pytest.raises(ValueError):
        prepare_batch(episodes[:1])
    same = collect(None, [make_cases("train", 1)[0]] * 4, baseline="oracle")
    assert torch.equal(prepare_batch(same)["advantages"], torch.zeros(sum(len(e.actions) for e in same)))


def test_checkpoint_resume_is_parameter_exact(tmp_path):
    cfg = Config(updates=4, batch_size=2, group_size=3, eval_count=4, eval_every=2)
    continuous = train(cfg, tmp_path/"continuous")
    train(replace(cfg, updates=2), tmp_path/"resumed")
    resumed = train(cfg, tmp_path/"resumed", tmp_path/"resumed/last.pt")
    for a, b in zip(continuous.parameters(), resumed.parameters()):
        assert torch.equal(a, b)
    loaded, _ = load_policy(tmp_path/"resumed/last.pt")
    assert all(torch.equal(a, b) for a, b in zip(resumed.parameters(), loaded.parameters()))
    with pytest.raises(ValueError, match="mismatch"):
        train(replace(cfg, kl_coef=0.5), tmp_path/"resumed", tmp_path/"resumed/last.pt")


@pytest.mark.parametrize("mode", ["oracle", "invalid", "cancel", "parallel"])
def test_actual_agentrl_session_roundtrip(mode):
    async def run():
        task = ToolPolicyTask(count=1)
        session = Session(1)
        async def worker():
            result = await task.start_sample(0, session)
            await session.controller.env_finish(TaskOutput(status=result.status, result=result.result, history=session.history))
            return result
        worker_future = asyncio.create_task(worker())
        response = await session.controller.agent_pull()
        while response.status == SampleStatus.RUNNING:
            history = response.history
            obs = json.loads(next(x["content"] for x in reversed(history) if isinstance(x, dict) and x["role"] in ("user", "tool")))
            call = {"id": "c", "type": "function", "function": {"name": oracle_action(obs), "arguments": "{}"}}
            if mode == "invalid":
                call["function"]["arguments"] = '{"bad": 1}'
            agent = AgentOutput(status=AgentOutputStatus.CANCELLED) if mode == "cancel" else AgentOutput(messages=[{
                "role": "assistant", "tool_calls": [call, call] if mode == "parallel" else [call]}])
            response = await session.controller.agent_pull(agent)
        result = await worker_future
        expected = {"oracle": SampleStatus.COMPLETED, "invalid": SampleStatus.AGENT_INVALID_ACTION,
                    "parallel": SampleStatus.AGENT_INVALID_ACTION, "cancel": SampleStatus.CANCELLED}[mode]
        assert result.status == expected
        if mode == "oracle":
            assert result.result["success"]
    asyncio.run(asyncio.wait_for(run(), timeout=10))


@pytest.mark.parametrize("change", [{"group_size": 1}, {"updates": 0}, {"backend": "unknown"}])
def test_bad_config_fails_fast(change):
    with pytest.raises(ValueError):
        Config(**change).validate()
