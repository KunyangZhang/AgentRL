import json
from pathlib import Path
import sqlite3
import pytest
from sql_agent_rl.environment import SQLEnv, execute, parse_action, same_result


@pytest.fixture
def database(tmp_path):
    folder = tmp_path/"sales"
    folder.mkdir()
    db = folder/"sales.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE sales (region TEXT, amount INTEGER)")
        con.executemany("INSERT INTO sales VALUES (?,?)", [("east", 10), ("west", 20), ("east", 30)])
    return db


@pytest.mark.parametrize("sql", ["DELETE FROM sales", "DROP TABLE sales", "INSERT INTO sales VALUES ('x', 1)",
    "ATTACH DATABASE '/tmp/unwanted.sqlite' AS leak", "PRAGMA writable_schema=ON", "SELECT load_extension('bad')",
    "SELECT * FROM sales; DELETE FROM sales"])
def test_rejects_mutation_and_escape(database, sql):
    before = database.read_bytes()
    assert not execute(database, sql)["ok"]
    assert database.read_bytes() == before


def test_readonly_join_and_aggregation(database):
    r = execute(database, "SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY region")
    assert r["ok"] and r["rows"] == [["east", 40], ["west", 20]]


def test_row_limit_and_recursive_budget(database):
    assert not execute(database, "SELECT * FROM sales", row_limit=1)["ok"]
    result = execute(database, "WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x")
    assert not result["ok"]


def test_agent_can_fix_sql_after_tool_error_without_gold_leak(database):
    case = {"uid": "a", "db_id": "sales", "question": "What is total revenue?",
            "query": "SELECT sum(amount) FROM sales"}
    env = SQLEnv(case, database.parent.parent)
    assert case["query"] not in json.dumps(env.messages)
    env.step('{"tool":"query","sql":"SELECT sum(unknown) FROM sales"}')
    env.step('{"tool":"query","sql":"SELECT sum(amount) FROM sales"}')
    env.step('{"tool":"submit","sql":"SELECT sum(amount) FROM sales"}')
    assert env.score()["success"] and env.score()["error_recovered"]
    with pytest.raises(RuntimeError):
        env.step('{}')


def test_failed_submission_does_not_pass_on_executable_sql(database):
    env = SQLEnv({"uid": "a", "db_id": "sales", "question": "Total revenue", "query": "SELECT sum(amount) FROM sales"}, database.parent.parent)
    env.step('{"tool":"submit","sql":"SELECT count(*) FROM sales"}')
    assert env.score()["executable"] and not env.score()["success"]


@pytest.mark.parametrize("text", ['[]', '{}', '{"tool":"shell","sql":"ls"}', '{"tool":"query","sql":5}',
                                      '{"tool":"query","sql":"select 1","extra":true}'])
def test_invalid_action_contract(text):
    assert parse_action(text) is None


def test_multiset_duplicate_and_order_semantics():
    a = {"ok": True, "rows": [[1], [1], [2]]}
    b = {"ok": True, "rows": [[2], [1], [1]]}
    assert same_result(a, b)
    assert not same_result(a, b, ordered=True)
    assert not same_result(a, {"ok": True, "rows": [[1], [2]]})


def test_rejects_database_path_escape(database):
    with pytest.raises(ValueError, match="identifier"):
        SQLEnv({"db_id": "../sales"}, database.parent.parent)
    outside = database.parent.parent/'outside'
    outside.mkdir()
    (outside/'alias').symlink_to(database.parent)
    with pytest.raises(ValueError, match="escapes"):
        SQLEnv({"db_id": "alias"}, outside)


@pytest.mark.parametrize("mode", ["correct", "cancel", "budget", "invalid"])
def test_real_agentrl_session_lifecycle(database, tmp_path, mode):
    import asyncio
    from agentrl.worker.task import Session
    from agentrl.worker.typings import AgentOutput, AgentOutputStatus, SampleStatus, TaskOutput
    from sql_agent_rl.task import SQLAgentTask
    cases = tmp_path/'cases.json'
    cases.write_text(json.dumps([{"uid":"s", "db_id":"sales", "question":"Total revenue", "query":"SELECT sum(amount) FROM sales"}]))
    async def run():
        task = SQLAgentTask(cases, database.parent.parent)
        session = Session(1)
        async def worker():
            result = await task.start_sample(0, session)
            await session.controller.env_finish(TaskOutput(status=result.status, result=result.result, history=session.history))
            return result
        future = asyncio.create_task(worker())
        response = await session.controller.agent_pull()
        while response.status == SampleStatus.RUNNING:
            text = '{"tool":"submit","sql":"SELECT sum(amount) FROM sales"}' if mode == "correct" else '{}'
            agent = AgentOutput(messages=[{"role":"assistant", "content":text}])
            if mode == "cancel":
                agent = AgentOutput(status=AgentOutputStatus.CANCELLED)
            if mode == "invalid":
                agent = AgentOutput(messages=[{"role":"assistant", "content":"{}"}]*2)
            response = await session.controller.agent_pull(agent)
        result = await future
        expected = {"correct":SampleStatus.COMPLETED, "cancel":SampleStatus.CANCELLED,
                    "budget":SampleStatus.TASK_LIMIT_REACHED, "invalid":SampleStatus.AGENT_INVALID_ACTION}
        assert result.status == expected[mode]
        if mode == "correct":
            assert result.result["success"]
    asyncio.run(asyncio.wait_for(run(), timeout=5))
