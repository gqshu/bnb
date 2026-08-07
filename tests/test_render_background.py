import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_background as rb  # noqa: E402  (path hack must precede this import)

from bnb import stable_audio  # noqa: E402
from bnb.background import build_signature  # noqa: E402


def _args(**overrides):
    base = dict(
        provider="elevenlabs",
        model_id="music_v2",
        output_format="pcm_44100",
        sa3_model=stable_audio.DEFAULT_MODEL,
        sa3_backend=stable_audio.DEFAULT_BACKEND,
        sa3_steps=None,
        sa3_cfg=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_default_provider_is_stable_audio(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["render_background.py"])
    assert rb.parse_args().provider == "stable_audio"


def test_check_provider_ready_elevenlabs_requires_the_api_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ELEVENLABS_API_KEY"):
        rb.check_provider_ready(_args(provider="elevenlabs"))


def test_check_provider_ready_elevenlabs_passes_when_key_set(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-fake")
    readiness = rb.check_provider_ready(_args(provider="elevenlabs", model_id="music_v2"))
    assert "elevenlabs" in readiness and "music_v2" in readiness


def test_check_provider_ready_stable_audio_requires_the_cli(monkeypatch):
    def fake_require_cli(backend):
        raise RuntimeError(f"no {backend} CLI found")

    monkeypatch.setattr(stable_audio, "require_cli", fake_require_cli)
    with pytest.raises(SystemExit, match="no torch CLI found"):
        rb.check_provider_ready(_args(provider="stable_audio", sa3_backend="torch"))


def test_check_provider_ready_stable_audio_passes_when_cli_found(monkeypatch):
    monkeypatch.setattr(stable_audio, "require_cli", lambda backend: Path("/fake/stable-audio"))
    readiness = rb.check_provider_ready(_args(provider="stable_audio", sa3_model="small-music", sa3_backend="torch"))
    assert "small-music" in readiness and "torch" in readiness and "/fake/stable-audio" in readiness


def test_describe_grid_spec():
    spec = build_signature("drone", "buddhist_meditative", 60).spec()
    assert rb.describe(spec) == "drone x buddhist_meditative"


def test_describe_special_spec():
    from bnb.background import build_keyword_signature

    spec = build_keyword_signature("natural_sounds", "rain", 60).spec()
    assert rb.describe(spec) == "natural_sounds:rain"


def test_provider_kwargs_elevenlabs():
    kwargs = rb.provider_kwargs(_args(provider="elevenlabs", output_format="mp3_44100_192", model_id="music_v1"))
    assert kwargs == {"output_format": "mp3_44100_192", "model_id": "music_v1"}


def test_provider_kwargs_stable_audio():
    kwargs = rb.provider_kwargs(_args(provider="stable_audio", sa3_model="medium", sa3_backend="mlx", sa3_steps=16, sa3_cfg=5.0))
    assert kwargs == {"model": "medium", "backend": "mlx", "steps": 16, "cfg": 5.0}


def test_model_version_elevenlabs_is_the_model_id():
    assert rb.model_version(_args(provider="elevenlabs", model_id="music_v2")) == "music_v2"


def test_model_version_stable_audio_folds_in_backend():
    args = _args(provider="stable_audio", sa3_model="small-music", sa3_backend="torch")
    assert rb.model_version(args) == "small-music:torch"


def test_record_output_format_preserves_elevenlabs_format_string():
    args = _args(provider="elevenlabs", output_format="pcm_44100")
    assert rb.record_output_format(args) == "pcm_44100"


def test_record_output_format_stable_audio_is_always_wav():
    args = _args(provider="stable_audio")
    assert rb.record_output_format(args) == "wav"


def test_render_stable_audio_uses_the_audiosparx_adapted_prompt(monkeypatch, tmp_path):
    spec = build_signature("melodic_instrument", "neoclassical", 60).spec()  # felt_piano -> "Piano" tag
    captured = {}

    def fake_render(spec, out_path, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        out_path.write_bytes(b"fake")
        return out_path

    monkeypatch.setattr(stable_audio, "render", fake_render)

    result = rb.render_stable_audio(spec, model="small-music", backend="torch", steps=None, cfg=None)

    assert result.exists()
    assert captured["spec"]["prompt"] != spec["prompt"]  # adapted, not the bare stored prompt
    assert "TrackType: Music" in captured["spec"]["prompt"]
    assert "Instruments: Piano" in captured["spec"]["prompt"]
    assert captured["kwargs"] == {"model": "small-music", "backend": "torch", "steps": None, "cfg": None}
