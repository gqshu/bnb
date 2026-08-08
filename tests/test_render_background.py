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
        sa3_sfx_model=stable_audio.DEFAULT_SFX_MODEL,
        sa3_sfx_backend=None,
        sa3_steps=None,
        sa3_cfg=None,
        max_retry=rb.DEFAULT_MAX_RETRY,
        no_qc=False,
        no_worker=False,
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


def test_engine_routes_special_cells_to_an_sfx_checkpoint():
    from bnb.background import build_keyword_signature

    grid = build_signature("drone", "lofi", 60).spec()
    special = build_keyword_signature("natural_sounds", "rain", 60).spec()

    # Traded down to the small checkpoint: field recordings still need an SFX-capable
    # one, but they follow the run's backend rather than needing a second flag.
    args = _args(provider="stable_audio", sa3_model="small-music", sa3_backend="torch")
    assert rb.engine_for(grid, args) == rb.Engine("small-music", "torch")
    assert rb.engine_for(special, args) == rb.Engine("medium", "torch")

    # Default: one checkpoint covers both, so the run is a single engine group.
    args = _args(provider="stable_audio")
    assert rb.engine_for(grid, args) == rb.engine_for(special, args) == rb.Engine("medium", "mlx")

    # An explicit override still wins.
    args = _args(provider="stable_audio", sa3_backend="torch", sa3_sfx_backend="mlx")
    assert rb.engine_for(special, args) == rb.Engine("medium", "mlx")


def test_model_version_elevenlabs_is_the_model_id():
    engine = rb.Engine("elevenlabs", "music_v2")
    assert rb.model_version(_args(provider="elevenlabs", model_id="music_v2"), engine) == "music_v2"


def test_model_version_stable_audio_folds_in_the_engine_that_rendered_it():
    args = _args(provider="stable_audio")
    assert rb.model_version(args, rb.Engine("small-music", "torch")) == "small-music:torch"
    # A special cell in the same run is rendered by a different checkpoint, and says so.
    assert rb.model_version(args, rb.Engine("medium", "mlx")) == "medium:mlx"


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


# --- the quality gate: check every render, re-render what fails ------------------


def _write_wav(path: Path, *, healthy: bool) -> Path:
    """A 6s stereo file that either passes bnb.qc or fails it as silent."""
    import numpy as np
    import soundfile as sf

    frames = 44100 * 6
    if healthy:
        rng = np.random.default_rng(0)
        data = (rng.standard_normal((frames, 2)) * 0.1).astype("float32")
    else:
        data = np.zeros((frames, 2), dtype="float32")
    sf.write(path, data, 44100)
    return path


def _renderer(tmp_path, verdicts):
    """A fake render callable: one entry of ``verdicts`` consumed per attempt."""
    calls = []

    def render(spec):
        healthy = verdicts[len(calls)]
        calls.append(spec["seed"])
        return _write_wav(tmp_path / f"attempt{len(calls)}.wav", healthy=healthy)

    return render, calls


def test_a_healthy_render_is_kept_on_the_first_attempt(tmp_path):
    spec = build_signature("drone", "lofi", 60).spec()
    render, calls = _renderer(tmp_path, [True])

    path, quality = rb.render_checked(render, spec, max_retry=3, check=True)

    assert path is not None and path.exists()
    assert calls == [spec["seed"]]  # the spec's own seed, untouched
    assert quality["attempts"] == 1
    assert quality["verdict"] in {"ok", "warn"}
    assert quality["seed"] == spec["seed"]


def test_a_failed_render_is_redone_with_a_fresh_seed(tmp_path):
    spec = build_signature("drone", "lofi", 60).spec()
    render, calls = _renderer(tmp_path, [False, False, True])

    path, quality = rb.render_checked(render, spec, max_retry=3, check=True)

    assert path is not None and path.exists()
    assert quality["attempts"] == 3
    assert calls[0] == spec["seed"]
    assert len(set(calls)) == 3  # every retry moved the seed
    assert quality["seed"] == calls[-1]  # provenance records what actually rendered
    assert not (tmp_path / "attempt1.wav").exists()  # failures are cleaned up


def test_giving_up_after_max_retry_keeps_nothing(tmp_path):
    spec = build_signature("drone", "lofi", 60).spec()
    render, calls = _renderer(tmp_path, [False] * 4)

    path, quality = rb.render_checked(render, spec, max_retry=3, check=True)

    assert path is None
    assert len(calls) == 4  # the first attempt plus three retries
    assert quality["verdict"] == "fail"
    assert not list(tmp_path.glob("*.wav"))


def test_max_retry_zero_renders_once(tmp_path):
    spec = build_signature("drone", "lofi", 60).spec()
    render, calls = _renderer(tmp_path, [False])

    path, _ = rb.render_checked(render, spec, max_retry=0, check=True)

    assert path is None and len(calls) == 1


def test_no_qc_keeps_whatever_came_back(tmp_path):
    spec = build_signature("drone", "lofi", 60).spec()
    render, calls = _renderer(tmp_path, [False])

    path, quality = rb.render_checked(render, spec, max_retry=3, check=False)

    assert path is not None and path.exists()  # kept, silent though it is
    assert len(calls) == 1
    assert quality["verdict"] == "unchecked"


def test_retry_seeds_are_deterministic():
    from bnb.background import retry_seed

    spec = build_signature("drone", "lofi", 60).spec()
    assert retry_seed(spec["track_id"], 1) == retry_seed(spec["track_id"], 1)
    assert retry_seed(spec["track_id"], 1) != retry_seed(spec["track_id"], 2)
    assert retry_seed(spec["track_id"], 1) != spec["seed"]


# --- render_session: hold the model open where the backend allows ----------------


def test_session_uses_a_worker_for_the_torch_backend(monkeypatch, tmp_path):
    started = []

    class FakeWorker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)
            return self

        def render(self, spec, out_path):
            started.append(spec["prompt"])
            Path(out_path).write_bytes(b"fake")
            return Path(out_path)

        def close(self):
            started.append("closed")

    monkeypatch.setattr(stable_audio, "Worker", FakeWorker)

    with rb.render_session(_args(provider="stable_audio"), rb.Engine("small-music", "torch")) as render:
        out = render(build_signature("drone", "lofi", 60).spec())

    assert out.exists()
    assert started[0]["model"] == "small-music"
    assert "TrackType: Music" in started[1]  # the SA3-adapted prompt, as the CLI path sends
    assert started[-1] == "closed"  # the model is released at the end of the batch


def test_session_falls_back_to_one_process_per_track(monkeypatch, capsys):
    def unavailable(**kwargs):
        raise stable_audio.WorkerError("no python next to the torch CLI")

    monkeypatch.setattr(stable_audio, "Worker", unavailable)
    monkeypatch.setattr(rb, "render_stable_audio", lambda spec, **kw: Path("/tmp/x.wav"))

    with rb.render_session(_args(provider="stable_audio"), rb.Engine("small-music", "torch")) as render:
        assert render(build_signature("drone", "lofi", 60).spec()) == Path("/tmp/x.wav")
    assert "falling back" in capsys.readouterr().out


def test_session_never_starts_a_worker_for_mlx_or_elevenlabs(monkeypatch):
    monkeypatch.setattr(stable_audio, "Worker", lambda **kw: pytest.fail("should not start a worker"))
    monkeypatch.setattr(rb, "render_stable_audio", lambda spec, **kw: Path("/tmp/x.wav"))
    monkeypatch.setattr(rb, "render_elevenlabs", lambda spec, **kw: Path("/tmp/y.wav"))

    spec = build_signature("drone", "lofi", 60).spec()
    with rb.render_session(_args(provider="stable_audio"), rb.Engine("medium", "mlx")) as render:
        assert render(spec) == Path("/tmp/x.wav")
    with rb.render_session(_args(provider="stable_audio", no_worker=True), rb.Engine("small-music", "torch")) as render:
        assert render(spec) == Path("/tmp/x.wav")
    with rb.render_session(_args(provider="elevenlabs"), rb.Engine("elevenlabs", "music_v2")) as render:
        assert render(spec) == Path("/tmp/y.wav")
