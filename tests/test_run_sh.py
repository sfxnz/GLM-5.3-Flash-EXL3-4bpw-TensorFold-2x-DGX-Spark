"""run.sh and stop.sh on a fake pair: stub docker, ssh, ip, rdma, free, ss, curl, hf and rsync on PATH, a fake HF cache.

The ssh stub runs the command locally as the worker, with the head's HF cache path pointing at the worker's own fake
cache, as the same absolute path on another machine would. Every stub call is logged, so a test can assert what ran.
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TR = "a" * 40
MODEL = "sfxnz/GLM-5.3-Flash-EXL3-4bpw"
TF_SHA = "17632c30f761e7f9051255744a17fc36f0b9c0d46343d47617bb34e7ec4e4128"
GIB = 1 << 30
SIDE_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "processor_config.json", "chat_template.jinja"]

STUBS = {
    "docker": r'''
echo "$FAKE_NODE docker $*" >> "$FAKE/calls"
d=$FAKE/$FAKE_NODE
case "$1" in
  ps)
    if [[ "$*" == *--filter* ]]; then
      if [[ "$*" == *'{{.State}}'* ]]; then cat "$d/ours"; else cut -d' ' -f1 "$d/ours"; fi
    else cut -d'|' -f1 "$d/inspect" | tr -d /; fi ;;
  inspect)
    if [[ "$2" == -f ]]; then grep -q "^$4 running$" "$d/ours" && echo true || echo false
    else grep "^/${@: -1}|" "$d/inspect"; fi ;;
  image) [[ -s "$d/image" ]] && cat "$d/image" || exit 1 ;;
  run) prev=""; for a in "$@"; do [[ $prev == --name ]] && echo "$a running" >> "$d/ours"; prev=$a; done ;;
  logs) echo "[tensorfold] serving on $FAKE_NODE" ;;
esac
exit 0''',
    "ssh": r'''
echo "ssh $*" >> "$FAKE/calls"
while [[ "$1" == -o ]]; do shift 2; done
shift
cmd="$*"
FAKE_NODE=worker bash -c "${cmd//$FAKE_HF/$FAKE/worker-hf}"''',
    "ip": r'''
d=$FAKE/$FAKE_NODE
if [[ "$*" == *" link "* ]]; then echo "4: x: <UP> mtu 1500 qdisc mq state $(cat "$d/link") mode DEFAULT"
else echo "4: x    inet $(cat "$d/ip")/24 brd 10.0.0.255 scope global"; fi''',
    "rdma": r'''for h in $(cat "$FAKE/$FAKE_NODE/hca"); do echo "link $h/1 state ACTIVE physical_state LINK_UP netdev x"; done''',
    "free": r'''echo "              total        used        free      shared  buff/cache   available"
echo "Mem: 130000000000 1 1 1 1 $(cat "$FAKE/$FAKE_NODE/mem")"''',
    "ss": r'''cat "$FAKE/$FAKE_NODE/ports"''',
    "curl": r'''echo "curl $*" >> "$FAKE/calls"; [[ -e "$FAKE/head/healthy" ]]''',
    "hf": r'''echo "hf $*" >> "$FAKE/calls"''',
    "rsync": r'''echo "rsync $*" >> "$FAKE/calls"''',
    "hostname": r'''if [[ $FAKE_NODE == worker ]]; then echo spark2; else echo spark-head; fi''',
}


class Pair:
    """A fake head and worker. Defaults are a clean pair: everything present and healthy, nothing running."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name, body in STUBS.items():
            p = self.bin / name
            p.write_text("#!/usr/bin/env bash\n" + body.lstrip() + "\n")
            p.chmod(0o755)
        self.hf = self.tmp / "hf"
        (self.tmp / "calls").touch()
        for node, ip in (("head", "10.9.9.1"), ("worker", "10.9.9.2")):
            self.node(node, link="UP", ip=ip, hca="rocep1s0f1 roceP2p1s0f1", mem=str(120 * GIB), ports="",
                      inspect="", ours="", image=f"sha256:{'1' * 64} {TF_SHA}")
        self.shard_bytes = self.snapshot(self.hf)
        self.snapshot(self.tmp / "worker-hf")

    def node(self, node, **files):
        d = self.tmp / node
        d.mkdir(exist_ok=True)
        for name, text in files.items():
            (d / name).write_text(text + ("\n" if text else ""))

    def snapshot(self, hf):
        snap = hf / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / TR
        snap.mkdir(parents=True)
        for f in SIDE_FILES:
            (snap / f).write_text("{}")
        shards = {"model-00001-of-00002.safetensors": 1000, "model-00002-of-00002.safetensors": 2345}
        for name, size in shards.items():
            (snap / name).write_bytes(b"\0" * size)
        index = {"weight_map": {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors",
                                "c.weight": "model-00002-of-00002.safetensors"}}
        (snap / "model.safetensors.index.json").write_text(json.dumps(index))
        return sum(shards.values())

    def run(self, script="run.sh", **env):
        base = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.tmp), "FAKE": str(self.tmp), "FAKE_NODE": "head",
                "FAKE_HF": str(self.hf), "HF_CACHE": str(self.hf), "REVISION": TR, "SHARDS": "2",
                "SHARD_BYTES": str(self.shard_bytes), "VALIDATE_ONLY": "1"}
        p = subprocess.run([str(ROOT / script)], env={**base, **env}, capture_output=True, text=True, timeout=60)
        return p.returncode, p.stdout + p.stderr

    def calls(self):
        return (self.tmp / "calls").read_text().splitlines()

    def close(self):
        shutil.rmtree(self.tmp)


def argvs(out):
    found = dict(re.findall(r"^argv rank(\d) \S+: (.*)$", out, re.M))
    return {int(r): shlex.split(a) for r, a in found.items()}


def conf(*names, env=None):
    script = f"source {ROOT / 'recipe.conf'}; " + "; ".join(f'echo "${n}"' for n in names)
    clean = {k: v for k, v in os.environ.items() if k not in names}
    out = subprocess.run(["bash", "-c", script], env={**clean, **(env or {})}, capture_output=True, text=True, check=True)
    return dict(zip(names, out.stdout.splitlines()))


def flag(argv, name):
    return argv[argv.index(name) + 1]


def env_of(argv):
    return dict(argv[i + 1].split("=", 1) for i, w in enumerate(argv) if w == "-e")


# The stubs log their own calls, so a docker write on the worker shows as "worker docker ...", not only inside an ssh line.
WRITE = re.compile(r"^(head|worker) docker (run|create|start|stop|kill|rm|build|load|save)\b|^(hf|rsync) |^ssh .*\bmkdir\b")


class ValidateOnly(unittest.TestCase):
    def setUp(self):
        self.pair = Pair()

    def tearDown(self):
        self.pair.close()

    def assertNoWrites(self):
        writes = [c for c in self.pair.calls() if WRITE.match(c)]
        self.assertEqual(writes, [])

    def test_clean_pair_passes_and_both_ranks_get_one_argv(self):
        rc, out = self.pair.run()
        self.assertEqual(rc, 0, out)
        self.assertIn("every check passes", out)
        a = argvs(out)
        self.assertEqual(set(a), {0, 1})
        strip = lambda argv, r: [w for w in argv if w not in (f"tf-glm53-rank{r}", f"TF_GLM_RANK={r}")]
        self.assertEqual(strip(a[0], 0), strip(a[1], 1))
        self.assertEqual(env_of(a[0])["TF_GLM_RANK"], "0")
        self.assertEqual(env_of(a[1])["TF_GLM_RANK"], "1")
        self.assertNoWrites()

    def test_argv_carries_the_label_names_mount_and_pins_from_recipe_conf(self):
        rc, out = self.pair.run()
        c = conf("IMAGE", "SERVED_NAME", "PORT", "CONTEXT", "PARALLEL", "LABEL", "NAME_PREFIX", "IFACE", "HCA", "ROCE_HCA",
                 "NCCL_PORT", "DENSE_BITS", "HEAD_BITS", "MEMORY_RESERVE_GIB")
        repo = f"{self.pair.hf}/hub/models--sfxnz--GLM-5.3-Flash-EXL3-4bpw"
        for r, argv in argvs(out).items():
            self.assertEqual(argv[:3], ["docker", "run", "-d"])
            self.assertEqual(flag(argv, "--name"), f"{c['NAME_PREFIX']}{r}")
            self.assertEqual(flag(argv, "--label"), c["LABEL"])
            self.assertEqual(flag(argv, "-v"), f"{repo}:{repo}:ro")
            for f in ("--gpus", "--network", "--ipc", "--device"):
                self.assertIn(f, argv)
            i = argv.index(c["IMAGE"])
            self.assertEqual(argv[i + 1:i + 3], ["tensorfold-native", "serve"])
            self.assertEqual(argv[i + 3], f"{repo}/snapshots/{TR}")
            self.assertEqual(flag(argv, "--context"), c["CONTEXT"])
            self.assertEqual(flag(argv, "--parallel"), c["PARALLEL"])
            self.assertEqual(flag(argv, "--port"), c["PORT"])
            self.assertEqual(argv[argv.index("--port") + 2:argv.index("--port") + 4], ["--name", c["SERVED_NAME"]])
            self.assertIn("--vision", argv)
            self.assertNotIn("--chat-template", argv)
            e = env_of(argv)
            self.assertEqual(e["TF_GLM_WORLD"], "2")
            self.assertEqual(e["NCCL_COMM_ID"], f"10.9.9.1:{c['NCCL_PORT']}")
            self.assertEqual(e["NCCL_SOCKET_IFNAME"], c["IFACE"])
            self.assertEqual(e["NCCL_IB_HCA"], c["HCA"])
            self.assertEqual(e["TF_ROCE_HCA"], c["ROCE_HCA"])
            self.assertEqual(e["TF_GLM_DENSE_BITS"], c["DENSE_BITS"])
            self.assertEqual(e["TF_GLM_HEAD_BITS"], c["HEAD_BITS"])
            self.assertEqual(e["TENSORFOLD_MEMORY_RESERVE_GIB"], c["MEMORY_RESERVE_GIB"])

    def test_environment_overrides_reach_both_ranks(self):
        rc, out = self.pair.run(CONTEXT="131072", PARALLEL="2", SERVED_NAME="glm-test", DENSE_BITS="16",
                                MEMORY_RESERVE_GIB="12.2")
        self.assertEqual(rc, 0, out)
        for argv in argvs(out).values():
            self.assertEqual(flag(argv, "--context"), "131072")
            self.assertEqual(flag(argv, "--parallel"), "2")
            self.assertIn("glm-test", argv)
            self.assertEqual(env_of(argv)["TF_GLM_DENSE_BITS"], "16")
            self.assertEqual(env_of(argv)["TENSORFOLD_MEMORY_RESERVE_GIB"], "12.2")

    def test_bad_pack_bits_and_reserve_are_refused(self):
        for k, v, msg in (("DENSE_BITS", "5", "DENSE_BITS=5 must be 4, 6, 8 or 16"),
                          ("HEAD_BITS", "bf16", "HEAD_BITS=bf16 must be 4, 6, 8 or 16"),
                          ("MEMORY_RESERVE_GIB", "10G", "MEMORY_RESERVE_GIB=10G is not a number of GiB")):
            rc, out = self.pair.run(**{k: v})
            self.assertNotEqual(rc, 0)
            self.assertIn(msg, out)

    def test_foreign_gpu_container_is_refused_and_never_touched(self):
        self.pair.node("worker", inspect='/glm53-quant-probe-b|glm53-quant:exl3-1.6.6||runc [{"Driver":"","Count":-1,"Capabilities":[["gpu"]]}] []')
        self.pair.node("head", inspect="/web|nginx:1||runc null []")
        rc, out = self.pair.run()
        self.assertEqual(rc, 1)
        self.assertRegex(out, r"REFUSE spark2 runs GPU container glm53-quant-probe-b glm53-quant:exl3-1.6.6")
        self.assertFalse([l for l in re.findall(r"^REFUSE.*$", out, re.M) if "nginx" in l])
        self.assertIn("ok     spark-head has no other GPU container", out)
        self.assertNoWrites()

    def test_real_run_refuses_a_foreign_gpu_container_before_any_write(self):
        self.pair.node("head", inspect='/other|img||nvidia null []')
        rc, out = self.pair.run(VALIDATE_ONLY="0")
        self.assertEqual(rc, 1)
        self.assertIn("error: spark-head runs GPU container other img", out)
        self.assertNoWrites()

    def test_unpinned_pack_is_refused(self):
        rc, out = self.pair.run(REVISION="unpublished")
        self.assertEqual(rc, 1)
        self.assertIn("REFUSE the pack is not pinned", out)
        (self.pair.tmp / "calls").write_text("")
        rc, out = self.pair.run(REVISION="unpublished", VALIDATE_ONLY="0")
        self.assertEqual(rc, 1)
        self.assertEqual(self.pair.calls(), [])

    def test_missing_worker_snapshot_plans_an_rsync_over_the_direct_link(self):
        shutil.rmtree(self.pair.tmp / "worker-hf")
        rc, out = self.pair.run()
        snap = f"{self.pair.hf}/hub/models--sfxnz--GLM-5.3-Flash-EXL3-4bpw/snapshots/{TR}"
        self.assertIn(f"WOULD  copy the snapshot to spark2 over enp1s0f1np1: rsync -aL --partial --info=progress2 {snap}/ 10.9.9.2:{snap}/", out)
        self.assertNotIn("WOULD  download", out)
        self.assertNoWrites()

    def test_short_head_snapshot_plans_a_download_with_one_include_per_pattern(self):
        rc, out = self.pair.run(SHARD_BYTES="999")
        self.assertIn("the shards hold 3345 bytes, want 999", out)
        line = next(l for l in out.splitlines() if l.startswith("WOULD  download"))
        words = shlex.split(line.split(": ", 1)[1])
        includes = [words[i + 1] for i, w in enumerate(words) if w == "--include"]
        self.assertEqual(includes, conf("HF_INCLUDE")["HF_INCLUDE"].split())
        self.assertEqual(flag(words, "--revision"), TR)

    def test_differing_worker_image_plans_a_copy(self):
        self.pair.node("worker", image=f"sha256:{'2' * 64} {TF_SHA}")
        rc, out = self.pair.run()
        self.assertIn("WOULD  copy tf-glm53:1.2.2 to spark2", out)
        self.assertNotIn("WOULD  build", out)

    def test_image_with_another_archive_is_rebuilt(self):
        self.pair.node("head", image=f"sha256:{'1' * 64} {'0' * 64}")
        rc, out = self.pair.run()
        self.assertIn("WOULD  build tf-glm53:1.2.2 from docker/", out)

    def test_memory_gate_refuses_and_force_overrides(self):
        self.pair.node("worker", mem=str(100 * GIB))
        rc, out = self.pair.run()
        self.assertEqual(rc, 1)
        self.assertIn(f"REFUSE MemAvailable below the {conf('MEM_GATE_GIB')['MEM_GATE_GIB']} GiB gate: spark2 100.0 GiB", out)
        self.assertIn("--- free -h on spark2", out)
        rc, out = self.pair.run(FORCE="1")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"FORCE=1: starting below the {conf('MEM_GATE_GIB')['MEM_GATE_GIB']} GiB gate", out)

    def test_busy_nccl_port_is_refused(self):
        self.pair.node("head", ports="LISTEN 0 4096 0.0.0.0:29584 0.0.0.0:*\nLISTEN 0 4096 0.0.0.0:22 0.0.0.0:*")
        rc, out = self.pair.run()
        self.assertEqual(rc, 1)
        self.assertIn("REFUSE spark-head: port(s) 29584 already listen", out)

    def test_inactive_hca_and_down_link_are_refused(self):
        self.pair.node("worker", hca="rocep1s0f1", link="DOWN")
        rc, out = self.pair.run()
        self.assertEqual(rc, 1)
        self.assertIn("REFUSE spark2: RoCE HCA not ACTIVE: roceP2p1s0f1", out)
        self.assertIn("REFUSE enp1s0f1np1 on spark2 is not up", out)

    def test_explicit_ip_that_is_not_on_the_interface_is_refused(self):
        rc, out = self.pair.run(HEAD_IP="10.9.9.7")
        self.assertEqual(rc, 1)
        self.assertIn("HEAD_IP=10.9.9.7, but enp1s0f1np1 has 10.9.9.1", out)

    def test_healthy_pair_of_ours_exits_zero_without_restarting(self):
        self.pair.node("head", ours="tf-glm53-rank0 running")
        self.pair.node("worker", ours="tf-glm53-rank1 running")
        (self.pair.tmp / "head" / "healthy").touch()
        rc, out = self.pair.run(VALIDATE_ONLY="0")
        self.assertEqual(rc, 0, out)
        self.assertIn("already serving", out)
        self.assertNoWrites()

    def test_leftover_rank_of_ours_is_cleared_before_a_fresh_start(self):
        self.pair.node("head", ours="tf-glm53-rank0 exited")
        rc, out = self.pair.run()
        self.assertIn("WOULD  stop and remove them", out)
        self.assertIn("WOULD  check ports", out)
        self.assertEqual(rc, 0, out)


class RealStart(unittest.TestCase):
    def setUp(self):
        self.pair = Pair()

    def tearDown(self):
        self.pair.close()

    def test_starts_rank1_then_rank0_with_the_planned_argvs(self):
        _, plan = self.pair.run()
        (self.pair.tmp / "head" / "healthy").touch()
        rc, out = self.pair.run(VALIDATE_ONLY="0")
        self.assertEqual(rc, 0, out)
        runs = [c for c in self.pair.calls() if re.match(r"\w+ docker run ", c)]
        planned = argvs(plan)
        self.assertEqual([c.split()[0] for c in runs], ["worker", "head"])
        # The ssh stub maps the head's cache path to the worker's fake cache; map it back to compare with the plan.
        worker_run = runs[0].replace(str(self.pair.tmp / "worker-hf"), str(self.pair.hf))
        self.assertEqual(shlex.split(worker_run)[1:], planned[1])
        self.assertEqual(shlex.split(runs[1])[1:], planned[0])
        self.assertIn("[tensorfold] serving on head", out)
        self.assertIn("endpoint  http://spark-head:8000/v1", out)

    def test_no_health_in_time_prints_both_logs_and_stops_ours(self):
        rc, out = self.pair.run(VALIDATE_ONLY="0", READY_TIMEOUT="1")
        self.assertEqual(rc, 1)
        self.assertIn("error: no /health after 1s", out)
        self.assertIn("--- rank 0 (spark-head), last 40 lines", out)
        self.assertIn("[tensorfold] serving on worker", out)
        stops = [c for c in self.pair.calls() if re.match(r"\w+ docker (stop|rm) ", c)]
        self.assertEqual(stops, ["head docker stop -t 60 tf-glm53-rank0", "head docker rm tf-glm53-rank0",
                                 "worker docker stop -t 60 tf-glm53-rank1", "worker docker rm tf-glm53-rank1"])


class Stop(unittest.TestCase):
    def setUp(self):
        self.pair = Pair()

    def tearDown(self):
        self.pair.close()

    def test_stops_rank0_then_rank1_and_removes_both(self):
        self.pair.node("head", ours="tf-glm53-rank0 running")
        self.pair.node("worker", ours="tf-glm53-rank1 running")
        rc, out = self.pair.run("stop.sh")
        self.assertEqual(rc, 0, out)
        acts = [c for c in self.pair.calls() if re.match(r"\w+ docker (stop|rm) ", c)]
        self.assertEqual(acts, ["head docker stop -t 60 tf-glm53-rank0", "head docker rm tf-glm53-rank0",
                                "worker docker stop -t 60 tf-glm53-rank1", "worker docker rm tf-glm53-rank1"])

    def test_nothing_running_is_a_quiet_success(self):
        rc, out = self.pair.run("stop.sh")
        self.assertEqual(rc, 0, out)
        self.assertIn("no tf-glm53-rank0 on spark-head", out)
        self.assertFalse([c for c in self.pair.calls() if WRITE.match(c)])

    def test_only_labelled_containers_are_asked_for(self):
        self.pair.run("stop.sh")
        listing = [c for c in self.pair.calls() if " docker ps" in c]
        self.assertEqual(len(listing), 2)
        label = conf("LABEL")["LABEL"]
        self.assertTrue(all(f"--filter label={label}" in c for c in listing))


class Static(unittest.TestCase):
    def test_shellcheck_is_clean(self):
        if not shutil.which("shellcheck"):
            self.skipTest("shellcheck not installed")
        p = subprocess.run(["shellcheck", "-x", "-S", "warning", "run.sh", "stop.sh"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout)

    def test_engine_pins_match_the_dockerfile(self):
        docker = (ROOT / "docker" / "Dockerfile").read_text()
        c = conf("TF_VERSION", "TF_SHA256")
        self.assertIn(f"ARG TF_VERSION={c['TF_VERSION']}\n", docker)
        self.assertIn(f"ARG TF_SHA256={c['TF_SHA256']}\n", docker)


if __name__ == "__main__":
    unittest.main()
