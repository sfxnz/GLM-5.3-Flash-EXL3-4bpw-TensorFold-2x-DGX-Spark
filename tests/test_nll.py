import hashlib
import io
import json
import random
import statistics
import sys
import tempfile
import unittest
import urllib.parse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import nll  # noqa: E402
from fake_server import FakeServer  # noqa: E402

WORDS = "the of river code ledger quietly seven north lantern broke under wire fold tensor amber".split()


def text(words, seed):
    rng = random.Random(seed)
    lines = [" ".join(rng.choice(WORDS) for _ in range(12)) for _ in range(words // 12)]
    return ("\n".join(lines) + "\n").encode()


class Synthetic:
    """A small corpus in a temp dir: 8 prose chunks, 6 code chunks and one long document past 16384 tokens."""

    def __init__(self, root):
        self.root = Path(root)
        files = {"https://example.test/prose.txt": text(4000, 1), "https://example.test/code.txt": text(3000, 2),
                 "https://example.test/long.txt": text(20000, 3)}
        self.sources = {}
        for url, raw in files.items():
            (self.root / urllib.parse.quote(url, safe="")).write_bytes(raw)
            self.sources[url] = hashlib.sha256(raw).hexdigest()
        self.corpus = ([("prose", "https://example.test/prose.txt", i * 2500, 2400) for i in range(8)]
                       + [("code", "https://example.test/code.txt", i * 2500, 2400) for i in range(6)]
                       + [("long", "https://example.test/long.txt", 0, 10 ** 9)])

    def __enter__(self):
        self.patches = [mock.patch.object(nll, "CORPUS", self.corpus), mock.patch.object(nll, "SOURCES", self.sources)]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = nll.main(argv)
    return rc, out.getvalue(), err.getvalue()


class NllGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "src").mkdir()
        self.corpus = Synthetic(self.dir / "src").__enter__()

    def tearDown(self):
        self.corpus.__exit__()
        self.tmp.cleanup()

    def capture(self, label, **knobs):
        with FakeServer(**knobs) as s:
            rc, out, err = run(["capture", "--label", label, "--out", str(self.dir / label), "--url", s.url,
                                "--corpus-root", str(self.dir / "src")])
            sent = list(s.requests)
        self.assertEqual(rc, 0, err)
        return sent

    def compare(self, a, b, *extra):
        report = self.dir / f"{a}-{b}.json"
        rc, out, err = run(["compare", str(self.dir / a), str(self.dir / b), "--json", str(report), *extra])
        return rc, (json.loads(report.read_text()) if report.exists() else None), out, err

    def test_capture_sends_each_chunk_once_as_a_scored_text_prompt(self):
        sent = self.capture("a")
        self.assertEqual(len(sent), len(self.corpus.corpus))
        self.assertTrue(all(b["prompt_logprobs"] == 1 and b["max_tokens"] == 1 and not b.get("stream") for b in sent))
        cap = json.loads((self.dir / "a" / "capture.json").read_text())
        for chunk, body in zip(cap["chunks"], sent):
            self.assertEqual(chunk["prompt_tokens"], len(body["prompt"].split()))
            self.assertEqual(len(chunk["ids"]), chunk["prompt_tokens"] - 1)

    def test_same_config_twice_pairs_to_exactly_zero_and_passes(self):
        self.capture("a")
        self.capture("a2")
        rc, rep, out, _ = self.compare("a", "a2")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["status"], "pass")
        self.assertEqual(rep["rows"][0]["delta"], 0.0)
        self.assertEqual(rep["rows"][0]["ci95"], [0.0, 0.0])
        self.assertEqual(rep["rows"][0]["identical"], 1.0)
        self.assertEqual(rep["rows"][0]["top1_agree"], 1.0)

    def test_boot_to_boot_jitter_gives_near_zero_with_a_ci_around_it(self):
        self.capture("a", jitter=1e-4)
        self.capture("a2", jitter=1.1e-4)
        rc, rep, out, _ = self.compare("a", "a2")
        overall = rep["rows"][0]
        self.assertEqual(rc, 0, out)
        self.assertLess(abs(overall["delta"]), 2e-5)
        self.assertLess(overall["ci95"][0], 0)
        self.assertGreater(overall["ci95"][1], 0)
        self.assertLess(overall["identical"], 0.01)

    def test_config_worse_than_the_margin_fails(self):
        self.capture("a")
        self.capture("b", nll_shift=0.01)
        rc, rep, out, _ = self.compare("a", "b")
        self.assertEqual(rc, 1)
        self.assertEqual(rep["status"], "worse")
        self.assertAlmostEqual(rep["rows"][0]["delta"], 0.01, places=9)
        self.assertIn("worse", out)

    def test_loss_inside_the_margin_passes(self):
        self.capture("a")
        self.capture("b", nll_shift=0.0005)
        rc, rep, _, _ = self.compare("a", "b")
        self.assertEqual(rc, 0)
        self.assertAlmostEqual(rep["rows"][0]["delta"], 0.0005, places=9)

    def test_too_noisy_to_bound_the_loss_is_inconclusive_and_fails(self):
        self.capture("a")
        self.capture("b", jitter=0.05)
        rc, rep, _, _ = self.compare("a", "b", "--margin", "0.0002")
        lo, hi = rep["rows"][0]["ci95"]
        self.assertEqual(rc, 1)
        self.assertEqual(rep["status"], "inconclusive")
        self.assertLess(lo, 0.0002)
        self.assertGreater(hi, 0.0002)

    def test_one_category_clearly_worse_fails_even_when_the_average_hides_it(self):
        self.capture("a")
        self.capture("b")
        cap = json.loads((self.dir / "b" / "capture.json").read_text())
        for c in cap["chunks"]:
            if c["category"] == "code":
                c["logprob"] = [x - 0.004 for x in c["logprob"]]
        (self.dir / "b" / "capture.json").write_text(json.dumps(cap))
        rc, rep, _, _ = self.compare("a", "b", "--margin", "0.003")
        rows = {r["group"]: r for r in rep["rows"]}
        self.assertLess(rows["all"]["ci95"][1], 0.003)
        self.assertGreater(rows["code"]["ci95"][0], 0.003)
        self.assertEqual(rep["status"], "worse")
        self.assertEqual(rc, 1)

    def test_long_document_reports_position_buckets(self):
        self.capture("a")
        self.capture("b", nll_shift=0.001)
        _, rep, _, _ = self.compare("a", "b")
        groups = [r["group"] for r in rep["rows"]]
        self.assertEqual(groups, ["all", "prose", "code", "long", "long@0-4096", "long@4096-16384", "long@16384-end"])
        late = rep["rows"][-1]
        self.assertEqual(late["tokens"], 20000 // 12 * 12 - 1 - 16384)
        self.assertAlmostEqual(late["delta"], 0.001, places=9)
        self.assertIsNotNone(late["ci95"])

    def test_captures_whose_token_ids_differ_are_refused(self):
        self.capture("a")
        self.capture("b", id_salt="other tokenizer")
        rc, rep, _, err = self.compare("a", "b")
        self.assertEqual(rc, 2)
        self.assertIsNone(rep)
        self.assertIn("do not pair", err)

    def test_a_changed_source_is_refused_before_any_request(self):
        path = self.dir / "src" / urllib.parse.quote("https://example.test/code.txt", safe="")
        path.write_bytes(path.read_bytes() + b"edited\n")
        with FakeServer() as s:
            rc, _, err = run(["capture", "--label", "a", "--out", str(self.dir / "a"), "--url", s.url,
                              "--corpus-root", str(self.dir / "src")])
            self.assertEqual(s.requests, [])
        self.assertEqual(rc, 2)
        self.assertIn("sha256", err)
        self.assertFalse((self.dir / "a" / "capture.json").exists())

    def test_an_existing_capture_is_never_overwritten(self):
        self.capture("a")
        before = (self.dir / "a" / "capture.json").read_bytes()
        with FakeServer(nll_shift=1.0) as s:
            rc, _, _ = run(["capture", "--label", "a", "--out", str(self.dir / "a"), "--url", s.url,
                            "--corpus-root", str(self.dir / "src")])
        self.assertEqual(rc, 2)
        self.assertEqual((self.dir / "a" / "capture.json").read_bytes(), before)

    def test_an_aa_floor_within_half_the_margin_refuses_the_gate(self):
        self.capture("a")
        self.capture("b")
        aa = self.dir / "aa.json"
        aa.write_text(json.dumps({"rows": [{"group": "all", "ci95": [-0.0004, 0.0011]}]}))
        rc, _, _, err = self.compare("a", "b", "--aa", str(aa))
        self.assertEqual(rc, 2)
        self.assertIn("A/A floor", err)
        aa.write_text(json.dumps({"rows": [{"group": "all", "ci95": [-0.0004, 0.0009]}]}))
        rc, rep, _, _ = self.compare("a", "b", "--aa", str(aa))
        self.assertEqual(rc, 0)
        self.assertEqual(rep["aa_floor"], 0.0009)


class PromptRowsTests(unittest.TestCase):
    def test_reads_tensorfolds_rows_with_the_prompt_token_first(self):
        # zig/src/server/logprobs_test.zig:399-416 at v1.2.2: prompt "abc", prompt_logprobs 1. "b" ranks second, so the
        # best token "z" follows it; "c" is the best, so it is the only entry.
        rows = [None,
                {"98": {"logprob": -0.125, "rank": 2, "decoded_token": "b"},
                 "122": {"logprob": -0.0625, "rank": 1, "decoded_token": "z"}},
                {"99": {"logprob": -0.01, "rank": 1, "decoded_token": "c"}}]
        self.assertEqual(nll.parse_rows(rows, 3), {"ids": [98, 99], "logprob": [-0.125, -0.01], "rank": [2, 1],
                                                   "top1": [122, 99]})

    def test_rows_that_do_not_cover_the_prompt_are_rejected(self):
        with self.assertRaises(ValueError):
            nll.parse_rows([None, {"5": {"logprob": -1.0, "rank": 1}}], 3)
        with self.assertRaises(ValueError):
            nll.parse_rows([{"5": {"logprob": -1.0, "rank": 1}}], 1)


class BootstrapTests(unittest.TestCase):
    def test_equal_units_have_a_zero_width_interval(self):
        self.assertEqual(nll.bootstrap([(0.5, 100)] * 10), (0.005, 0.005, 0.005))

    def test_mean_is_per_token_not_per_chunk(self):
        mean, _, _ = nll.bootstrap([(10.0, 10)] + [(0.0, 90)] * 4)
        self.assertAlmostEqual(mean, 10 / 370)

    def test_interval_matches_the_normal_approximation_on_iid_units(self):
        rng = random.Random(11)
        xs = [rng.gauss(0.01, 0.1) for _ in range(400)]
        mean, lo, hi = nll.bootstrap([(x, 1) for x in xs])
        half = 1.96 * statistics.stdev(xs) / 400 ** 0.5
        self.assertAlmostEqual(mean, statistics.fmean(xs), places=12)
        self.assertAlmostEqual(mean - lo, half, delta=0.1 * half)
        self.assertAlmostEqual(hi - mean, half, delta=0.1 * half)

    def test_fewer_than_five_units_get_no_interval(self):
        self.assertEqual(nll.bootstrap([(1.0, 10)] * 4), (0.1, None, None))


@unittest.skipUnless(nll.CORPUS_ROOT.is_dir(), f"needs the pinned corpus sources in {nll.CORPUS_ROOT}")
class RealCorpusTests(unittest.TestCase):
    def test_the_pinned_corpus_captures_whole_and_pairs_with_itself(self):
        with tempfile.TemporaryDirectory() as d, FakeServer() as s:
            for label in ("a", "a2"):
                rc, _, err = run(["capture", "--label", label, "--out", f"{d}/{label}", "--url", s.url])
                self.assertEqual(rc, 0, err)
            rc, out, err = run(["compare", f"{d}/a", f"{d}/a2", "--json", f"{d}/r.json"])
            rep = json.loads(Path(f"{d}/r.json").read_text())
            longest = max(len(b["prompt"]) for b in s.requests)
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(len(s.requests), 82)
        self.assertEqual(rep["corpus"]["version"], "nll-corpus-v1")
        self.assertEqual([r["group"] for r in rep["rows"][:6]], ["all", "prose", "multilingual", "code", "math", "long"])
        self.assertEqual(rep["rows"][0]["delta"], 0.0)
        self.assertGreater(longest, 130000)


if __name__ == "__main__":
    unittest.main()
