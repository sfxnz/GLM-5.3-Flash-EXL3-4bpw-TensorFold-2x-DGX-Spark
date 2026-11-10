import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import agents  # noqa: E402
from fake_server import FakeServer  # noqa: E402


def run(args, **knobs):
    with FakeServer(**knobs) as s, tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
        rc = agents.main([*args, "--url", s.url, "--out", f"{d}/agents.json"])
        report = json.loads(Path(f"{d}/agents.json").read_text())
        bodies = list(s.requests)
    by_client = {}
    for b in bodies:
        by_client.setdefault(b["messages"][1]["content"], []).append(b)
    return rc, report, bodies, [sorted(v, key=lambda b: len(b["messages"])) for v in by_client.values()]


class AgentsTests(unittest.TestCase):
    def test_each_client_keeps_its_own_growing_history_on_one_shared_system_prompt(self):
        rc, report, bodies, clients = run(["--clients", "2", "--turns", "3", "--system-tokens", "600",
                                           "--context-tokens", "400"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(clients), 2)
        self.assertEqual(len({b["messages"][0]["content"] for b in bodies}), 1)
        self.assertGreater(len(bodies[0]["messages"][0]["content"]), 600 * 3)
        self.assertEqual({json.dumps(b["tools"]) for b in bodies}, {json.dumps(bodies[0]["tools"])})
        for sent in clients:
            system_and_user, user, reply, fake_tool_results = 2, 1, 1, 2
            per_turn = user + reply + fake_tool_results
            self.assertEqual([len(b["messages"]) for b in sent], [system_and_user + per_turn * n for n in range(3)])
            self.assertGreater(len(sent[0]["messages"][1]["content"]), 400 * 3)
            for earlier, later in zip(sent, sent[1:]):
                self.assertEqual(later["messages"][:len(earlier["messages"])], earlier["messages"])
        self.assertTrue(all(b["stream"] and b["chat_template_kwargs"] == {"enable_thinking": False}
                            and "temperature" not in b for b in bodies))
        self.assertEqual([(t["client"], t["turn"], t["kind"]) for t in report["turns"]],
                         [(c, n, "cold" if n == 1 else "resumed") for c in (0, 1) for n in (1, 2, 3)])
        self.assertTrue(all(t["cached_tokens"] is None and t["prompt_tokens"] and t["ttft_s"] is not None
                            for t in report["turns"]))
        self.assertEqual((report["summary"]["cold"]["turns"], report["summary"]["resumed"]["turns"]), (2, 4))

    def test_streamed_tool_calls_are_rebuilt_and_answered(self):
        rc, report, _, clients = run(["--clients", "1", "--turns", "2", "--system-tokens", "200"])
        second = clients[0][1]
        names = [t["function"]["name"] for t in second["tools"][:2]]
        self.assertEqual(rc, 0)
        self.assertEqual(second["messages"][2]["role"], "assistant")
        self.assertEqual(second["messages"][2]["tool_calls"], [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": names[i], "arguments": json.dumps({"path": f"src/mod_{i}.py"})}} for i in (0, 1)])
        self.assertEqual([(m["role"], m["tool_call_id"]) for m in second["messages"][3:5]],
                         [("tool", "call_0"), ("tool", "call_1")])
        self.assertEqual(second["messages"][5]["role"], "user")
        self.assertEqual([t["tool_calls"] for t in report["turns"]], [2, 2])

    def test_resumed_turns_that_hit_the_prefix_cache_show_a_lower_ttft(self):
        rc, report, bodies, _ = run(["--clients", "2", "--turns", "2", "--system-tokens", "1000",
                                     "--context-tokens", "2000", "--temperature", "0"],
                                    prefix_cache=True, prefill_s_per_ktok=0.2)
        s = report["summary"]
        cold = {t["client"]: t for t in report["turns"] if t["kind"] == "cold"}
        self.assertEqual(rc, 0)
        self.assertTrue(all(b["temperature"] == 0 for b in bodies))
        for t in report["turns"]:
            if t["kind"] == "resumed":
                self.assertGreaterEqual(t["cached_tokens"], cold[t["client"]]["prompt_tokens"] - 20)
        self.assertLess(s["resumed_over_cold"], 0.5)
        self.assertGreater(s["cold"]["median_ttft_s"], 0.3)

    def test_a_new_seed_gives_each_client_a_new_document_and_the_same_system_prompt(self):
        args = ["--clients", "2", "--turns", "1", "--system-tokens", "300", "--context-tokens", "300"]
        _, _, first, _ = run(args)
        _, _, again, _ = run(args)
        _, _, other, _ = run([*args, "--seed", "1"])
        docs = lambda bodies: sorted(b["messages"][1]["content"] for b in bodies)
        self.assertEqual(docs(first), docs(again))
        self.assertFalse(set(docs(first)) & set(docs(other)))
        self.assertEqual(first[0]["messages"][0], other[0]["messages"][0])

    def test_failed_turns_are_counted_and_fail_the_run(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            rc = agents.main(["--clients", "2", "--turns", "3", "--system-tokens", "100", "--url", "http://127.0.0.1:9",
                              "--out", f"{d}/agents.json"])
            report = json.loads(Path(f"{d}/agents.json").read_text())
        self.assertEqual(rc, 1)
        self.assertEqual(report["summary"]["cold"]["errors"], 2)
        self.assertTrue(all(t["error"] for t in report["turns"]))


if __name__ == "__main__":
    unittest.main()
