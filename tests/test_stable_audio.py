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
    cmd = stable_audio.build_command(spec, out, cli=FAKE_CLI)

    assert cmd[0] == FAKE_CLI
    assert argv_value(cmd, "--model") == "small-music"
    assert argv_value(cmd, "-p") == spec["prompt"]
    assert argv_value(cmd, "--seed") == str(spec["seed"])  # reproducible re-renders
    assert argv_value(cmd, "--duration") == "60"
    assert argv_value(cmd, "-o") == str(out)


def test_duration_override_beats_spec(spec, tmp_path):
    cmd = stable_audio.build_command(spec, tmp_path / "t.wav", duration_s=10, cli=FAKE_CLI)
    assert argv_value(cmd, "--duration") == "10"


def test_small_model_rejects_over_120s(spec, tmp_path):
    with pytest.raises(ValueError, match="at most 120s"):
        stable_audio.build_command(spec, tmp_path / "t.wav", duration_s=180, cli=FAKE_CLI)


def test_negative_prompt_and_steps_track_the_checkpoint_type(spec, tmp_path):
    # Post-trained: 8 distilled steps, guidance ignored -> no negative prompt sent.
    post = stable_audio.build_command(spec, tmp_path / "t.wav", cli=FAKE_CLI)
    assert argv_value(post, "--steps") == "8"
    assert "--negative-prompt" not in post

    # Base: needs many more steps and does respond to the spec's negative prompt.
    base = stable_audio.build_command(spec, tmp_path / "t.wav", model="small-music-base", cli=FAKE_CLI)
    assert argv_value(base, "--steps") == "50"
    assert argv_value(base, "--negative-prompt") == spec["negative_prompt"]


def test_explicit_steps_win(spec, tmp_path):
    cmd = stable_audio.build_command(spec, tmp_path / "t.wav", steps=16, cli=FAKE_CLI)
    assert argv_value(cmd, "--steps") == "16"


def test_cfg_turns_guidance_on_for_post_trained_too(spec, tmp_path):
    cmd = stable_audio.build_command(spec, tmp_path / "t.wav", cfg=7.0, cli=FAKE_CLI)
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
