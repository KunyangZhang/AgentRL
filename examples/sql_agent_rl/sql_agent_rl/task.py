"""Native AgentRL Task/Session adapter for the same text-JSON SQL environment."""
import asyncio
import json
from pathlib import Path
from agentrl.worker.task import Task
from agentrl.worker.typings import AgentCancelledException, RewardHistoryItem, SampleStatus, TaskSampleExecutionResult
from .environment import SQLEnv


class SQLAgentTask(Task):
    def __init__(self, cases_path, databases, name="sql-agent", max_turns=3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cases = json.loads(Path(cases_path).read_text())
        self.databases, self.max_turns = databases, max_turns

    def get_indices(self):
        return list(range(len(self.cases)))

    async def start_sample(self, index, session):
        env = SQLEnv(self.cases[int(index)], self.databases, self.max_turns)
        session.inject(env.messages)
        try:
            while not env.done:
                response = await session.action()
                messages = response.messages or []
                if len(messages) != 1 or not isinstance(messages[0].get("content"), str) or messages[0].get("tool_calls"):
                    return TaskSampleExecutionResult(status=SampleStatus.AGENT_INVALID_ACTION)
                await asyncio.to_thread(env.step, messages[0]["content"])
                session.inject(env.messages[-1])
            result = await asyncio.to_thread(env.score)
            session.inject(RewardHistoryItem(reward=result["reward"]))
            return TaskSampleExecutionResult(
                status=SampleStatus.COMPLETED if result["submitted"] else SampleStatus.TASK_LIMIT_REACHED,
                result={**result, "trace": env.trace})
        except AgentCancelledException:
            return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)

    def calculate_overall(self, results):
        return {"count": len(results), "execution_match": sum(bool(r.result and r.result.get("success")) for r in results)/max(1, len(results))}
