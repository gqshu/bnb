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


# --- elevenlabs credential preflight ------------------------------------------
#
# check_elevenlabs_credential imports the SDK inside the function, so these stub the
# `elevenlabs` module's ElevenLabs class. None of them touch the network.

FAKE_KEY = "sk-fakefakefake9xyz"


class _FakeSubscription:
    tier = "creator"
    character_count = 12_345
    character_limit = 100_000


def _stub_sdk(monkeypatch, result):
    """Point `elevenlabs.ElevenLabs(...).user.subscription.get()` at ``result`` — an
    exception instance is raised, anything else is returned."""
    import elevenlabs

    class _Sub:
        def get(self):
            if isinstance(result, BaseException):
                raise result
            return result

    class _User:
        subscription = _Sub()

    class _Client:
        def __init__(self, api_key=None):
            self.user = _User()

    monkeypatch.setattr(elevenlabs, "ElevenLabs", _Client)


def test_elevenlabs_preflight_verifies_the_key_and_reports_the_account(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    _stub_sdk(monkeypatch, _FakeSubscription())
    readiness = rb.check_provider_ready(_args(provider="elevenlabs", model_id="music_v2"))
    assert "elevenlabs" in readiness and "music_v2" in readiness
    assert "verified" in readiness
    assert "creator" in readiness and "12,345" in readiness


def test_elevenlabs_preflight_never_echoes_the_whole_key(monkeypatch):
    # The readiness line is printed and often pasted into issues/logs.
    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    _stub_sdk(monkeypatch, _FakeSubscription())
    readiness = rb.check_provider_ready(_args(provider="elevenlabs"))
    assert FAKE_KEY not in readiness
    assert readiness.count(FAKE_KEY[-4:]) == 1  # enough to identify which key, no more


def test_elevenlabs_preflight_rejects_a_set_but_invalid_key(monkeypatch):
    # The gap this closes: a present-but-wrong key used to pass and die on the first
    # *paid* call, partway into a batch.
    from elevenlabs.core.api_error import ApiError

    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    _stub_sdk(monkeypatch, ApiError(status_code=401, body={"detail": "invalid api key"}))
    with pytest.raises(SystemExit, match="rejected by the API"):
        rb.check_provider_ready(_args(provider="elevenlabs"))


def test_elevenlabs_preflight_accepts_a_restricted_key(monkeypatch):
    # The false alarm this fixes: ElevenLabs keys can be restricted to chosen scopes, and
    # the account read (user_read) is not one rendering needs. The API can only name the
    # missing scope after recognising the key, so this response *proves* authentication —
    # reporting it as a bad key rejected a perfectly good music-generation key.
    from elevenlabs.core.api_error import ApiError

    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    _stub_sdk(
        monkeypatch,
        ApiError(
            status_code=401,
            body={"detail": {"status": "missing_permissions", "message": "missing permission user_read"}},
        ),
    )
    readiness = rb.check_provider_ready(_args(provider="elevenlabs", model_id="music_v2"))
    assert "accepted" in readiness and "music_v2" in readiness
    assert "quota is unknown" in readiness
    assert "missing permission user_read" in readiness


def test_elevenlabs_preflight_surfaces_the_apis_own_reason(monkeypatch):
    # The 401 branch used to discard the body and assert its own guess at the cause, which
    # is what made a restricted key look identical to a revoked one.
    from elevenlabs.core.api_error import ApiError

    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    _stub_sdk(
        monkeypatch,
        ApiError(status_code=401, body={"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}),
    )
    with pytest.raises(SystemExit, match="invalid_api_key") as excinfo:
        rb.check_provider_ready(_args(provider="elevenlabs"))
    assert "Invalid API key" in str(excinfo.value)


def test_elevenlabs_preflight_trims_a_trailing_newline_from_the_key(monkeypatch):
    # `export ELEVENLABS_API_KEY=$(cat key.txt)` is the common way to get one, and the
    # newline it leaves makes a good key fail as "invalid" with no hint why.
    seen = {}

    import elevenlabs

    class _Client:
        def __init__(self, api_key=None):
            seen["api_key"] = api_key
            self.user = type("U", (), {"subscription": type("S", (), {"get": lambda s: _FakeSubscription()})()})()

    monkeypatch.setattr(elevenlabs, "ElevenLabs", _Client)
    monkeypatch.setenv("ELEVENLABS_API_KEY", f"  {FAKE_KEY}\n")
    readiness = rb.check_provider_ready(_args(provider="elevenlabs"))
    assert seen["api_key"] == FAKE_KEY  # sent clean
    assert "whitespace was trimmed" in readiness  # ...and said so


def test_elevenlabs_preflight_distinguishes_other_api_errors(monkeypatch):
    from elevenlabs.core.api_error import ApiError

    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    _stub_sdk(monkeypatch, ApiError(status_code=500, body="upstream boom"))
    with pytest.raises(SystemExit, match="returned 500"):
        rb.check_provider_ready(_args(provider="elevenlabs"))


def test_elevenlabs_preflight_reports_an_unreachable_api(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    _stub_sdk(monkeypatch, ConnectionError("dns go boom"))
    with pytest.raises(SystemExit, match="could not reach"):
        rb.check_provider_ready(_args(provider="elevenlabs"))


def test_elevenlabs_preflight_runs_on_a_dry_run(monkeypatch):
    # The point of the whole check: --dry-run is when you want a bad key to surface,
    # before any credits are committed. A dry run must not be a way to skip it.
    from elevenlabs.core.api_error import ApiError

    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    _stub_sdk(monkeypatch, ApiError(status_code=401, body={"detail": "nope"}))
    with pytest.raises(SystemExit, match="rejected by the API"):
        rb.check_provider_ready(_args(provider="elevenlabs", dry_run=True))


def test_stable_audio_preflight_never_touches_the_elevenlabs_api(monkeypatch):
    # An offline SA3 run must not acquire a network dependency from this change.
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(rb, "check_elevenlabs_credential", _boom)
    monkeypatch.setattr(stable_audio, "require_cli", lambda backend: Path("/fake/stable-audio"))
    assert "stable_audio" in rb.check_provider_ready(_args(provider="stable_audio"))


def _boom(*_a, **_k):
    raise AssertionError("elevenlabs credential check must not run for stable_audio")


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
    spec = build_signature("drone", "buddhist_meditative", "relax", 60).spec()
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

    grid = build_signature("drone", "lofi", "relax", 60).spec()
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
    spec = build_signature("melodic_instrument", "neoclassical", "relax", 60).spec()  # felt_piano -> "Piano" tag
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
    spec = build_signature("drone", "lofi", "relax", 60).spec()
    render, calls = _renderer(tmp_path, [True])

    path, quality = rb.render_checked(render, spec, max_retry=3, check=True)

    assert path is not None and path.exists()
    assert calls == [spec["seed"]]  # the spec's own seed, untouched
    assert quality["attempts"] == 1
    assert quality["verdict"] in {"ok", "warn"}
    assert quality["seed"] == spec["seed"]


def test_a_failed_render_is_redone_with_a_fresh_seed(tmp_path):
    spec = build_signature("drone", "lofi", "relax", 60).spec()
    render, calls = _renderer(tmp_path, [False, False, True])

    path, quality = rb.render_checked(render, spec, max_retry=3, check=True)

    assert path is not None and path.exists()
    assert quality["attempts"] == 3
    assert calls[0] == spec["seed"]
    assert len(set(calls)) == 3  # every retry moved the seed
    assert quality["seed"] == calls[-1]  # provenance records what actually rendered
    assert not (tmp_path / "attempt1.wav").exists()  # failures are cleaned up


def test_giving_up_after_max_retry_keeps_nothing(tmp_path):
    spec = build_signature("drone", "lofi", "relax", 60).spec()
    render, calls = _renderer(tmp_path, [False] * 4)

    path, quality = rb.render_checked(render, spec, max_retry=3, check=True)

    assert path is None
    assert len(calls) == 4  # the first attempt plus three retries
    assert quality["verdict"] == "fail"
    assert not list(tmp_path.glob("*.wav"))


def test_max_retry_zero_renders_once(tmp_path):
    spec = build_signature("drone", "lofi", "relax", 60).spec()
    render, calls = _renderer(tmp_path, [False])

    path, _ = rb.render_checked(render, spec, max_retry=0, check=True)

    assert path is None and len(calls) == 1


def test_no_qc_keeps_whatever_came_back(tmp_path):
    spec = build_signature("drone", "lofi", "relax", 60).spec()
    render, calls = _renderer(tmp_path, [False])

    path, quality = rb.render_checked(render, spec, max_retry=3, check=False)

    assert path is not None and path.exists()  # kept, silent though it is
    assert len(calls) == 1
    assert quality["verdict"] == "unchecked"


def test_retry_seeds_are_deterministic():
    from bnb.background import retry_seed

    spec = build_signature("drone", "lofi", "relax", 60).spec()
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
        out = render(build_signature("drone", "lofi", "relax", 60).spec())

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
        assert render(build_signature("drone", "lofi", "relax", 60).spec()) == Path("/tmp/x.wav")
    assert "falling back" in capsys.readouterr().out


def test_session_never_starts_a_worker_for_mlx_or_elevenlabs(monkeypatch):
    monkeypatch.setattr(stable_audio, "Worker", lambda **kw: pytest.fail("should not start a worker"))
    monkeypatch.setattr(rb, "render_stable_audio", lambda spec, **kw: Path("/tmp/x.wav"))
    monkeypatch.setattr(rb, "render_elevenlabs", lambda spec, **kw: Path("/tmp/y.wav"))

    spec = build_signature("drone", "lofi", "relax", 60).spec()
    with rb.render_session(_args(provider="stable_audio"), rb.Engine("medium", "mlx")) as render:
        assert render(spec) == Path("/tmp/x.wav")
    with rb.render_session(_args(provider="stable_audio", no_worker=True), rb.Engine("small-music", "torch")) as render:
        assert render(spec) == Path("/tmp/x.wav")
    with rb.render_session(_args(provider="elevenlabs"), rb.Engine("elevenlabs", "music_v2")) as render:
        assert render(spec) == Path("/tmp/y.wav")


# --- skip decision: the catalog flag is a snapshot, the filesystem is the truth ---


def _rendered_spec(manager):
    """One spec with real audio on disk, and a catalog that knows about it."""
    from bnb.background import build_signature

    spec = build_signature("drone", "buddhist_meditative", "relax", 60).spec()
    manager.add_spec(spec, rebuild=False)
    manager.add_render(
        spec,
        b"\x00\x00" * 4410,
        output_format="pcm_44100",
        provider="elevenlabs",
        model_version="music_v1",
        license="x",
        generated_at="now",
    )
    return spec


def test_select_todo_skips_a_track_whose_audio_is_really_there(tmp_path):
    from bnb.catalog import CategoryManager

    manager = CategoryManager(tmp_path)
    spec = _rendered_spec(manager)
    todo, orphaned = rb.select_todo([spec["track_id"]], manager)
    assert todo == [] and orphaned == 0


def test_select_todo_re_renders_when_the_audio_file_is_gone(tmp_path):
    # The bug this fixes: the catalog's `rendered` flag is a snapshot from its last
    # rebuild. Trusting it means a master that was deleted or lost is skipped *forever* —
    # never re-rendered, while its spec still advertises a render whose file is missing.
    from bnb import assets
    from bnb.catalog import CategoryManager

    manager = CategoryManager(tmp_path)
    spec = _rendered_spec(manager)
    track_id = spec["track_id"]

    audio = assets.find_track(track_id, root=tmp_path)
    audio.unlink()
    # Deliberately do NOT rebuild: the catalog still claims this track is rendered.
    assert manager.search(rendered=True)[0]["track_id"] == track_id

    todo, orphaned = rb.select_todo([track_id], manager)
    assert todo == [track_id]
    assert orphaned == 1


def test_select_todo_reports_a_missing_master_distinctly(tmp_path, capsys):
    # An unrendered spec and an orphaned one both need rendering, but only one of them
    # means something is wrong on disk — worth saying so rather than silently queueing it.
    from bnb import assets
    from bnb.background import build_signature
    from bnb.catalog import CategoryManager

    manager = CategoryManager(tmp_path)
    orphan = _rendered_spec(manager)
    fresh = build_signature("drone", "lofi", "relax", 60).spec()
    manager.add_spec(fresh, rebuild=False)
    assets.find_track(orphan["track_id"], root=tmp_path).unlink()

    todo, orphaned = rb.select_todo([orphan["track_id"], fresh["track_id"]], manager)
    assert sorted(todo) == sorted([orphan["track_id"], fresh["track_id"]])
    assert orphaned == 1
    out = capsys.readouterr().out
    assert f"re-render      {orphan['track_id']}" in out
    assert "audio file is missing" in out
    assert fresh["track_id"] not in out  # a never-rendered spec is not a discrepancy


def test_select_todo_force_takes_everything_without_flagging_orphans(tmp_path):
    from bnb.catalog import CategoryManager

    manager = CategoryManager(tmp_path)
    spec = _rendered_spec(manager)
    todo, orphaned = rb.select_todo([spec["track_id"]], manager, force=True)
    assert todo == [spec["track_id"]]
    assert orphaned == 0  # --force re-renders by intent, not because anything is missing


def test_re_rendering_an_orphan_refills_the_spec_render_block(tmp_path):
    # The other half of the ask: the spec must stop advertising a file that isn't there.
    # No extra step is needed — record_render replaces the whole block.
    from bnb import assets
    from bnb.catalog import CategoryManager

    manager = CategoryManager(tmp_path)
    spec = _rendered_spec(manager)
    assets.find_track(spec["track_id"], root=tmp_path).unlink()

    stale = assets.load_spec(spec["track_id"], root=tmp_path)
    assert stale["render"]["generated_at"] == "now"

    manager.add_render(
        stale,
        b"\x00\x00" * 4410,
        output_format="pcm_44100",
        provider="stable_audio",
        model_version="medium",
        license="y",
        generated_at="later",
    )
    refilled = assets.load_spec(spec["track_id"], root=tmp_path)
    assert refilled["render"]["generated_at"] == "later"
    assert refilled["render"]["provider"] == "stable_audio"
    assert assets.find_track(spec["track_id"], root=tmp_path) is not None
    assert rb.select_todo([spec["track_id"]], manager) == ([], 0)
