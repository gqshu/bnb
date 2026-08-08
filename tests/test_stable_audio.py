"""Rendering background specs with self-hosted Stable Audio 3.

The fast tests assert the invocation we build from a spec. The last test actually
generates audio and is marked ``slow`` (deselected by default, see pyproject):

    uv run pytest -m slow tests/test_stable_audio.py -s

It needs the sibling stable-audio-3 checkout synced. First run pulls the 2.3 GB
small-music checkpoint, a gated repo, so `hf auth login` must have run. If that
download crawls, disable the Xet backend — it measured ~10 KB/s here against
~1 MB/s for the plain CDN:

    HF_HUB_DISABLE_XET=1 hf download stabilityai/stable-audio-3-small-music \\
        model.safetensors model_config.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bnb import stable_audio
from bnb.background import build_signature

FAKE_CLI = "/opt/stable-audio-3/.venv/bin/stable-audio"
FAKE_MLX_CLI = "/opt/stable-audio-3/optimized/mlx/sa3"


@pytest.fixture
def spec():
    return build_signature("drone", "lofi", 60).spec()


def argv_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_command_carries_prompt_seed_and_duration(spec, tmp_path):
    out = tmp_path / "track.wav"
    cmd = stable_audio.build_command(spec, out, model="small-music", backend="torch", cli=FAKE_CLI)

    assert cmd[0] == FAKE_CLI
    assert argv_value(cmd, "--model") == "small-music"
    assert argv_value(cmd, "-p") == spec["prompt"]
    assert argv_value(cmd, "--seed") == str(spec["seed"])  # reproducible re-renders
    assert argv_value(cmd, "--duration") == "60"
    assert argv_value(cmd, "-o") == str(out)


def test_duration_override_beats_spec(spec, tmp_path):
    cmd = stable_audio.build_command(spec, tmp_path / "t.wav", duration_s=10, backend="torch", cli=FAKE_CLI)
    assert argv_value(cmd, "--duration") == "10"


def test_small_model_rejects_over_120s(spec, tmp_path):
    with pytest.raises(ValueError, match="at most 120s"):
        stable_audio.build_command(spec, tmp_path / "t.wav", model="small-music", duration_s=180, cli=FAKE_CLI)


def test_negative_prompt_and_steps_track_the_checkpoint_type(spec, tmp_path):
    # Post-trained: 8 distilled steps, guidance ignored -> no negative prompt sent.
    post = stable_audio.build_command(spec, tmp_path / "t.wav", backend="torch", cli=FAKE_CLI)
    assert argv_value(post, "--steps") == "8"
    assert "--negative-prompt" not in post

    # Base: needs many more steps and does respond to the spec's negative prompt.
    base = stable_audio.build_command(spec, tmp_path / "t.wav", model="small-music-base", backend="torch", cli=FAKE_CLI)
    assert argv_value(base, "--steps") == "50"
    assert argv_value(base, "--negative-prompt") == spec["negative_prompt"]


def test_explicit_steps_win(spec, tmp_path):
    cmd = stable_audio.build_command(spec, tmp_path / "t.wav", steps=16, backend="torch", cli=FAKE_CLI)
    assert argv_value(cmd, "--steps") == "16"


def test_cfg_turns_guidance_on_for_post_trained_too(spec, tmp_path):
    cmd = stable_audio.build_command(spec, tmp_path / "t.wav", cfg=7.0, backend="torch", cli=FAKE_CLI)
    assert argv_value(cmd, "--cfg-scale") == "7.0"
    assert argv_value(cmd, "--negative-prompt") == spec["negative_prompt"]


def test_mlx_command_renames_model_and_flags(spec, tmp_path):
    out = tmp_path / "track.wav"
    cmd = stable_audio.build_command(spec, out, model="medium", backend="mlx", cli=FAKE_MLX_CLI)

    assert cmd[0] == FAKE_MLX_CLI
    assert argv_value(cmd, "--dit") == "medium"
    assert argv_value(cmd, "--decoder") == "same-l"  # SAME-L is medium's codec
    assert argv_value(cmd, "--prompt") == spec["prompt"]
    assert argv_value(cmd, "--seconds") == "60"  # not --duration
    assert argv_value(cmd, "--out") == str(out)   # absolute, else it lands in optimized/mlx/output/
    assert "--model" not in cmd


def test_mlx_allows_medium_up_to_380s(spec, tmp_path):
    cmd = stable_audio.build_command(spec, tmp_path / "t.wav", model="medium", duration_s=300, backend="mlx", cli=FAKE_MLX_CLI)
    assert argv_value(cmd, "--seconds") == "300"

    with pytest.raises(ValueError, match="at most 380s"):
        stable_audio.build_command(spec, tmp_path / "t.wav", model="medium", duration_s=400, backend="mlx", cli=FAKE_MLX_CLI)


def test_mlx_omits_seed_when_random(spec, tmp_path):
    # The MLX CLI has no -1 sentinel — a random render means passing no --seed at all.
    cmd = stable_audio.build_command({**spec, "seed": -1}, tmp_path / "t.wav", backend="mlx", cli=FAKE_MLX_CLI)
    assert "--seed" not in cmd


def test_mlx_rejects_unconverted_checkpoints(spec, tmp_path):
    with pytest.raises(ValueError, match="no MLX conversion"):
        stable_audio.build_command(spec, tmp_path / "t.wav", model="small-music-base", backend="mlx", cli=FAKE_MLX_CLI)


# --- Worker: one model load, many renders ---------------------------------------
#
# Driven against a stub interpreter that speaks the protocol, so the client half is
# testable without torch or a checkpoint. The stub stands in for sa3_worker.py running
# inside the sibling venv; it logs every request so a test can assert what was sent
# and how many times the process booted.

STUB_WORKER = """#!/usr/bin/env python3
import json, sys, wave
LOG = {log!r}
def log(entry):
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\\n")

log({{"event": "boot", "argv": sys.argv[1:]}})
{startup}
print(json.dumps({{"event": "ready", "model": "stub"}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    log(request)
    {render}
"""

WRITE_WAV = """
    with wave.open(request["out"], "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes(b"\\x00\\x01" * 8820)
    print(json.dumps({"event": "rendered", "out": request["out"]}), flush=True)
"""
REPLY_ERROR = """
    print(json.dumps({"event": "error", "message": "kernel exploded"}), flush=True)
"""


@pytest.fixture
def stub_venv(tmp_path, monkeypatch):
    """A fake stable-audio-3 venv whose python speaks the worker protocol."""
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "stable-audio").write_text("#!/bin/sh\nexit 0\n")  # only has to exist
    log = tmp_path / "requests.jsonl"

    def install(*, render: str = WRITE_WAV, startup: str = "") -> Path:
        python = bin_dir / "python"
        python.write_text(STUB_WORKER.format(log=str(log), startup=startup, render=render))
        python.chmod(0o755)
        monkeypatch.setenv("BNB_SA3_CLI", str(bin_dir / "stable-audio"))
        return log

    return install


def requests_from(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_defaults_are_medium_on_mlx():
    # medium is the only checkpoint covering both music and SFX, and it needs MLX here.
    assert stable_audio.DEFAULT_MODEL == "medium"
    assert stable_audio.DEFAULT_BACKEND == "mlx"
    assert stable_audio.supports_sfx(stable_audio.DEFAULT_MODEL)
    assert stable_audio.DEFAULT_SFX_MODEL == stable_audio.DEFAULT_MODEL  # one load per run
    cmd = stable_audio.build_command(
        {"prompt": "x", "negative_prompt": "y", "seed": 1, "duration_s": 60},
        "/tmp/o.wav", cli=FAKE_MLX_CLI,
    )
    assert argv_value(cmd, "--dit") == "medium" and argv_value(cmd, "--decoder") == "same-l"


def test_worker_still_defaults_to_torch_since_it_is_the_only_resident_backend():
    assert stable_audio.Worker().backend == "torch"


def test_worker_is_torch_only():
    assert stable_audio.supports_worker("torch")
    assert not stable_audio.supports_worker("mlx")
    with pytest.raises(stable_audio.WorkerError, match="no worker"):
        stable_audio.Worker(backend="mlx")


def test_venv_python_sits_beside_the_cli(stub_venv, tmp_path):
    stub_venv()
    assert stable_audio.venv_python("torch") == tmp_path / ".venv" / "bin" / "python"


def test_worker_renders_from_the_spec(stub_venv, spec, tmp_path):
    log = stub_venv()
    out = tmp_path / "track.wav"

    with stable_audio.Worker(model="small-music") as worker:
        assert worker.render(spec, out) == out

    assert out.exists()
    boot, request = requests_from(log)
    script, *flags = boot["argv"]  # the stub stands in for the interpreter
    assert Path(script) == stable_audio.WORKER_SCRIPT
    assert flags == ["--model", "small-music"]
    assert request["prompt"] == spec["prompt"]
    assert request["seed"] == spec["seed"]  # same seed as the one-shot CLI path
    assert request["duration_s"] == 60
    assert request["steps"] == 8  # the post-trained default
    assert request["negative_prompt"] is None  # guidance off, so it can't steer


def test_worker_loads_the_model_once_for_a_batch(stub_venv, spec, tmp_path):
    log = stub_venv()
    with stable_audio.Worker() as worker:
        for i in range(3):
            worker.render({**spec, "seed": i}, tmp_path / f"{i}.wav")

    entries = requests_from(log)
    assert sum(e.get("event") == "boot" for e in entries) == 1  # one process, three renders
    assert [e["seed"] for e in entries if "seed" in e] == [0, 1, 2]


def test_worker_sends_the_negative_prompt_when_guidance_can_use_it(stub_venv, spec, tmp_path):
    log = stub_venv()
    with stable_audio.Worker(cfg=7.0) as worker:
        worker.render(spec, tmp_path / "t.wav")
    assert requests_from(log)[1]["negative_prompt"] == spec["negative_prompt"]


def test_worker_honours_explicit_steps(stub_venv, spec, tmp_path):
    log = stub_venv()
    with stable_audio.Worker(steps=16) as worker:
        worker.render(spec, tmp_path / "t.wav")
    assert requests_from(log)[1]["steps"] == 16


def test_worker_rejects_a_render_longer_than_the_model_allows(stub_venv, spec, tmp_path):
    stub_venv()
    with stable_audio.Worker(model="small-music") as worker:
        with pytest.raises(ValueError, match="at most 120s"):
            worker.render({**spec, "duration_s": 300}, tmp_path / "t.wav")


def test_worker_surfaces_a_render_error(stub_venv, spec, tmp_path):
    stub_venv(render=REPLY_ERROR)
    with stable_audio.Worker() as worker:
        with pytest.raises(stable_audio.WorkerError, match="kernel exploded"):
            worker.render(spec, tmp_path / "t.wav")


def test_worker_start_fails_when_the_model_cannot_load(stub_venv):
    stub_venv(startup='sys.exit("no checkpoint")')
    with pytest.raises(stable_audio.WorkerError, match="exited unexpectedly"):
        stable_audio.Worker().start()


def test_worker_without_a_venv_python_is_an_error(tmp_path, monkeypatch):
    cli = tmp_path / "stable-audio"  # a CLI with no interpreter beside it
    cli.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("BNB_SA3_CLI", str(cli))
    with pytest.raises(stable_audio.WorkerError, match="no python next to"):
        stable_audio.Worker().start()


@pytest.mark.slow
@pytest.mark.skipif(stable_audio.cli_path("torch") is None, reason="torch checkout not synced")
def test_worker_renders_two_tracks_off_one_model_load(spec, tmp_path):
    """The real thing: same audio as the CLI path, one load instead of two."""
    short = {**spec, "duration_s": 10}
    with stable_audio.Worker(model="small-music") as worker:
        first = worker.render(short, tmp_path / "a.wav")
        second = worker.render({**short, "seed": short["seed"] + 1}, tmp_path / "b.wav")

    for out in (first, second):
        info = sf.info(out)
        assert info.samplerate == 44100
        assert 9.5 <= info.duration <= 10.5
        audio, _ = sf.read(out, dtype="float32")
        assert float(np.sqrt(np.mean(audio**2))) > 0.001, "generated track is silent"


@pytest.mark.slow
@pytest.mark.skipif(stable_audio.cli_path() is None, reason="stable-audio-3 checkout not synced")
def test_renders_a_playable_background_track(spec, tmp_path):
    out = stable_audio.render(spec, tmp_path / f"{spec['track_id']}.wav", duration_s=10)

    assert out.exists()

    # Stable Audio writes float32 WAV via torchaudio, unlike the 16-bit PCM masters
    # the ElevenLabs path produces — so read with soundfile, not stdlib `wave`.
    info = sf.info(out)
    assert info.samplerate == 44100
    assert info.channels == 2
    assert 9.5 <= info.duration <= 10.5

    audio, _ = sf.read(out, dtype="float32")
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms > 0.001, "generated track is silent"
    assert np.abs(audio).max() < 1.0, "generated track is clipping"


@pytest.mark.slow
@pytest.mark.skipif(stable_audio.cli_path("mlx") is None, reason="optimized/mlx not installed")
def test_mlx_renders_medium_on_apple_silicon(spec, tmp_path):
    """The medium checkpoint, which the torch backend can't run without CUDA."""
    out = stable_audio.render(
        spec, tmp_path / f"{spec['track_id']}.wav", model="medium", backend="mlx", duration_s=10
    )

    info = sf.info(out)
    assert info.samplerate == 44100
    assert info.channels == 2
    assert 9.5 <= info.duration <= 10.5
    assert info.subtype == "PCM_16"  # unlike torchaudio's float32, this matches the ElevenLabs masters

    audio, _ = sf.read(out, dtype="float32")
    assert float(np.sqrt(np.mean(audio**2))) > 0.001, "generated track is silent"
