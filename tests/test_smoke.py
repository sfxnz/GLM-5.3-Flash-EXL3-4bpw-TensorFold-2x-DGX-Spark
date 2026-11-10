import io
import json
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke  # noqa: E402
from fake_server import FakeServer  # noqa: E402


def run_smoke(server, *extra):
    with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
        rc = smoke.main(["--url", server.url, "--out", f"{d}/smoke.json", *extra])
        return rc, json.loads(Path(f"{d}/smoke.json").read_text()), out.getvalue()


def pixel(png, x, y):
    w = struct.unpack(">I", png[16:20])[0]
    idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
    raw = zlib.decompress(idat)
    row = raw[y * (1 + 3 * w) + 1:]
    return tuple(row[3 * x:3 * x + 3])


class SmokeTests(unittest.TestCase):
    def test_healthy_server_passes_every_check(self):
        with FakeServer() as s:
            rc, results, out = run_smoke(s)
        self.assertEqual(rc, 0, out)
        self.assertEqual(set(results), set(smoke.Smoke.CHECKS))
        self.assertIn("smoke: 10/10 passed", out)
        self.assertNotIn("auth", results)

    def test_tool_arguments_must_stream_in_pieces_that_join_to_the_final_call(self):
        with FakeServer(whole_tool_args=True) as s:
            rc, results, _ = run_smoke(s)
        self.assertEqual(rc, 1)
        self.assertFalse(results["tool_stream"]["ok"])
        self.assertFalse(results["messages_stream"]["ok"])
        with FakeServer(drift_tool_args=True) as s:
            rc, results, _ = run_smoke(s)
        self.assertFalse(results["tool_stream"]["ok"])
        self.assertTrue(results["messages"]["ok"])

    def test_keyed_server_needs_the_key_and_the_tools_send_it(self):
        with FakeServer(api_key="sk-test") as s, mock.patch.dict("os.environ", {"API_KEY": "sk-test"}):
            rc, results, out = run_smoke(s)
        self.assertEqual(rc, 0, out)
        self.assertTrue(results["auth"]["ok"])
        self.assertIn("smoke: 11/11 passed", out)
        with FakeServer(api_key="sk-test") as s, mock.patch.dict("os.environ", {"API_KEY": "sk-wrong"}):
            rc, results, _ = run_smoke(s)
        self.assertEqual(rc, 1)
        self.assertFalse(results["models"]["ok"])

    def test_concurrent_check_sends_parallel_requests(self):
        with FakeServer() as s:
            rc, results, _ = run_smoke(s, "--parallel", "4")
        self.assertEqual(rc, 0)
        self.assertEqual(len(results["concurrent"]["detail"]["requests"]), 4)

    def test_think_tags_in_thinking_off_content_fail(self):
        with FakeServer(leak_think=True) as s:
            rc, results, _ = run_smoke(s)
        self.assertEqual(rc, 1)
        self.assertFalse(results["thinking_off"]["ok"])
        self.assertTrue(results["thinking_on"]["ok"])

    def test_no_accepted_drafts_fails(self):
        with FakeServer(accepted=0) as s:
            rc, results, _ = run_smoke(s)
        self.assertEqual(rc, 1)
        self.assertFalse(results["speculative"]["ok"])

    def test_swapped_shapes_fail_vision(self):
        swapped = '{"left": {"color": "blue", "shape": "circle"}, "right": {"color": "red", "shape": "square"}}'
        with FakeServer(vision=swapped) as s:
            rc, results, _ = run_smoke(s)
        self.assertEqual(rc, 1)
        self.assertFalse(results["vision"]["ok"])

    def test_vision_answer_inside_a_code_fence_passes(self):
        fenced = '```json\n{"left": {"color": "Red", "shape": "Square"}, "right": {"color": "Blue", "shape": "circle"}}\n```'
        with FakeServer(vision=fenced) as s:
            _, results, _ = run_smoke(s)
        self.assertTrue(results["vision"]["ok"])

    def test_wrong_window_fails_models(self):
        with FakeServer(window=131072) as s:
            rc, results, _ = run_smoke(s)
        self.assertEqual(rc, 1)
        self.assertFalse(results["models"]["ok"])

    def test_dead_server_fails_without_crashing(self):
        rc, results, _ = run_smoke(type("Dead", (), {"url": "http://127.0.0.1:9"})())
        self.assertEqual(rc, 1)
        self.assertFalse(any(r["ok"] for r in results.values()))

    def test_png_has_red_square_left_and_blue_circle_right(self):
        png = smoke.shapes_png()
        self.assertEqual(pixel(png, 64, 64), (255, 0, 0))
        self.assertEqual(pixel(png, 192, 64), (0, 0, 255))
        self.assertEqual(pixel(png, 225, 30), (255, 255, 255))
        self.assertEqual(pixel(png, 5, 5), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
