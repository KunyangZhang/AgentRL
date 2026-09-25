"""Native AgentRL Task/Session bridge; no training-specific environment fork."""
import json
from agentrl.worker.task import Task
from agentrl.worker.typings import (AgentCancelledException, RewardHistoryItem,
                                    SampleStatus, TaskSampleExecutionResult)
from .environment import ACTIONS, ToolEnv, make_cases

TOOLS = [{"type": "function", "function": {
    "name": a, "description": {
        "fetch": "Load the task's starting value. Retry temporary_unavailable.",
        "add": "Add the current instruction operand to the accumulator.",
        "subtract": "Subtract the current instruction operand from the accumulator.",
        "multiply": "Multiply accumulator by the current instruction operand.",
        "submit": "Submit current accumulator after all instructions have executed."
    }[a], "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True}} for a in ACTIONS]


class ToolPolicyTask(Task):
    def __init__(self, name="tool-policy-train", split="train", count=256, shaped=True, **kwargs):
        super().__init__(name=name, tools=TOOLS, **kwargs)
        self.cases, self.shaped = make_cases(split, count), shaped

    def get_indices(self):
        return list(range(len(self.cases)))

    async def start_sample(self, index, session):
        env = ToolEnv(self.cases[int(index)], self.shaped)
        session.set_tools(TOOLS)
        session.inject({"role": "system", "content":
                        "Execute the arithmetic program. Call fetch first, follow each instruction, "
                        "then submit. Retry temporary failures. One tool call per turn; no arguments."})
        session.inject({"role": "user", "content": json.dumps(env.observe())})
        try:
            while not env.done:
                response = await session.action()
                calls = [c for m in response.messages or [] for c in (m.get("tool_calls") or [])]
                if len(calls) != 1:
                    session.inject(RewardHistoryItem(reward=-0.3))
                    return TaskSampleExecutionResult(status=SampleStatus.AGENT_INVALID_ACTION)
                call = calls[0]
                try:
                    args = json.loads(call["function"]["arguments"])
                    if args != {}:
                        raise ValueError("tools accept no arguments")
                    name = call["function"]["name"]
                    if name not in ACTIONS:
                        raise ValueError("unknown tool")
                except (KeyError, ValueError, TypeError):
                    session.inject(RewardHistoryItem(reward=-0.3))
                    return TaskSampleExecutionResult(status=SampleStatus.AGENT_INVALID_ACTION)
                obs, reward, _ = env.step(name)
                session.inject({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(obs)})
                session.inject(RewardHistoryItem(reward=reward))
            status = SampleStatus.TASK_LIMIT_REACHED if env.last_error == "call_budget_exhausted" else SampleStatus.COMPLETED
            return TaskSampleExecutionResult(status=status, result={"success": env.success, "trace": env.trace})
        except AgentCancelledException:
            return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)

    def calculate_overall(self, results):
        return {"success_rate": sum(bool(r.result and r.result.get("success")) for r in results) / max(len(results), 1),
                "count": len(results)}
