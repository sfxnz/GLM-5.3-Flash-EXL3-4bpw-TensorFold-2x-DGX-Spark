import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import bench  # noqa: E402
import needle  # noqa: E402
import recipe  # noqa: E402
from fake_server import FakeServer  # noqa: E402


def events(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


class NeedleTests(unittest.TestCase):
    def test_finds_the_code_inside_the_window(self):
        with FakeServer(window=200000) as s, redirect_stdout(io.StringIO()) as out:
            rc = needle.main(["20000", "--url", s.url])
        result = events(out.getvalue())[-1]
        self.assertEqual(rc, 0)
        self.assertTrue(result["found"])
        self.assertLessEqual(abs(result["prompt_tokens"] - 20000), 1000)

    def test_target_past_the_window_is_cut_to_fit(self):
        with FakeServer(window=30000) as s, redirect_stdout(io.StringIO()) as out:
            rc = needle.main(["90000", "--url", s.url])
        ev = events(out.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(ev[0]["event"], "rescaled")
        self.assertTrue(ev[-1]["found"])
        self.assertLessEqual(ev[-1]["prompt_tokens"], 30000)

    def test_context_length_refusal_rescales_and_retries(self):
        # The server claims a larger window than it admits, so only its refusal can bring the prompt down.
        with FakeServer(window=30000, reported_window=60000) as s, redirect_stdout(io.StringIO()) as out:
            rc = needle.main(["50000", "--url", s.url])
        ev = events(out.getvalue())
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("refused", [e["event"] for e in ev])
        self.assertTrue(ev[-1]["found"])
        self.assertLessEqual(ev[-1]["prompt_tokens"], 30000)


class BenchTests(unittest.TestCase):
    def test_records_raw_streams_and_medians(self):
        with FakeServer() as s, tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            rc = bench.main(["--concurrency", "1,4", "--runs", "2", "--max-tokens", "64", "--url", s.url,
                             "--out", f"{d}/bench.json"])
            report = json.loads(Path(f"{d}/bench.json").read_text())
            sent = list(s.requests)
        self.assertEqual(rc, 0)
        self.assertEqual([r["concurrency"] for r in report["runs"]], [1, 1, 4, 4])
        self.assertEqual([len(r["streams"]) for r in report["runs"]], [1, 1, 4, 4])
        self.assertEqual([x["concurrency"] for x in report["summary"]], [1, 4])
        stream = report["runs"][0]["streams"][0]
        self.assertEqual(stream["tokens_from"], "usage")
        self.assertIsNotNone(report["summary"][1]["median_ttft_s"])
        self.assertTrue(all(b["stream"] and b["chat_template_kwargs"] == {"enable_thinking": False} for b in sent))

    def test_warmup_runs_are_sent_but_not_recorded(self):
        with FakeServer() as s, tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            rc = bench.main(["--concurrency", "1,2", "--runs", "1", "--warmup", "2", "--max-tokens", "16",
                             "--url", s.url, "--out", f"{d}/bench.json"])
            report = json.loads(Path(f"{d}/bench.json").read_text())
            sent = len(s.requests)
        self.assertEqual(rc, 0)
        self.assertEqual(sent, (2 + 1) * (1 + 2))
        self.assertEqual([r["concurrency"] for r in report["runs"]], [1, 2])
        self.assertEqual([x["runs"] for x in report["summary"]], [1, 1])
        self.assertEqual(report["settings"]["warmup"], 2)

    def test_temperature_is_sent_only_when_set(self):
        for args, sent_temps, recorded in [([], set(), None), (["--temperature", "0"], {0.0}, 0.0)]:
            with FakeServer() as s, tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
                bench.main(["--concurrency", "2", "--runs", "1", "--max-tokens", "16", "--url", s.url,
                            "--out", f"{d}/bench.json", *args])
                report = json.loads(Path(f"{d}/bench.json").read_text())
                bodies = list(s.requests)
            self.assertEqual({b["temperature"] for b in bodies if "temperature" in b}, sent_temps)
            self.assertEqual(sum("temperature" in b for b in bodies), len(bodies) if sent_temps else 0)
            self.assertEqual(report["settings"]["temperature"], recorded)

    def test_failed_streams_are_counted_not_hidden(self):
        with redirect_stdout(io.StringIO()):
            rc = bench.main(["--concurrency", "2", "--runs", "1", "--url", "http://127.0.0.1:9"])
        self.assertEqual(rc, 1)


class RecipeSettingsTests(unittest.TestCase):
    def test_reads_the_values_bash_reads_and_the_environment_wins(self):
        names = ["SERVED_NAME", "PORT", "CONTEXT", "PARALLEL"]
        env = {k: v for k, v in os.environ.items() if k not in names}
        script = f"source {ROOT / 'recipe.conf'}; " + "; ".join(f'echo "${n}"' for n in names)
        bash = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual([recipe.setting(n) for n in names], bash.stdout.split())
        with mock.patch.dict(os.environ, {"PORT": "8123"}):
            self.assertEqual(recipe.default_url(), "http://127.0.0.1:8123")


if __name__ == "__main__":
    unittest.main()
