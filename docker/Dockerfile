# syntax=docker/dockerfile:1

# bnb backend — the FastAPI stream/catalog service.
#
# Two properties of this image are dictated by how bnb finds things at runtime, and both
# look like clutter that wants tidying up. Neither does.
#
#   1. The asset repository is located *relative to the source tree*:
#      `assets.ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"`. With the
#      package living at /app/src/bnb/assets.py that resolves to /app/assets — which is
#      exactly where the host's assets directory gets mounted. Installing the package
#      into site-packages instead would drag ASSETS_DIR off to somewhere inside the
#      interpreter's lib directory and the service would find no tracks at all. So this
#      keeps a source layout at /app with an editable install pointing into it.
#
#   2. `bnb.server.main()` — what `scripts/serve.py` runs — binds 127.0.0.1, which inside
#      a container means "unreachable from anywhere". The CMD therefore drives uvicorn
#      directly with --host 0.0.0.0 rather than going through that entrypoint.

ARG PYTHON_VERSION=3.12

# ── build: resolve the locked dependency set into a self-contained venv ──────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS build

# Pin this to a specific uv release (e.g. ghcr.io/astral-sh/uv:0.9.2) once you've picked
# one — :latest keeps the build from breaking on a tag that doesn't exist yet, at the
# cost of the build not being reproducible across rebuilds.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, from the lockfile alone. With only pyproject.toml + uv.lock bound
# into this step, the layer stays cached across ordinary source edits; --no-install-project
# keeps bnb itself out of it so a change to src/ doesn't re-resolve the whole tree.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

# Then the source, and the project on top of the cached dependency layer. README.md is
# not documentation here — pyproject's `readme` field points at it, so the hatchling
# build fails without it.
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ── runtime: the venv and the source tree, nothing else ─────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# libsndfile is what soundfile binds to when reading the rendered background WAVs.
# Current soundfile wheels bundle their own copy, but the system library costs a few
# hundred KB and removes a whole class of "the wheel didn't ship one for this arch".
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Run as a normal user, defaulting to the first-login uid/gid Ubuntu hands out. Override
# at build time to match whoever owns the assets directory on the host — it matters as
# soon as that mount is read-write (see docker/compose.yaml).
ARG UID=1000
ARG GID=1000
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" bnb; fi; \
    if ! getent passwd "${UID}" >/dev/null; then \
        useradd -u "${UID}" -g "${GID}" -M -d /app -s /usr/sbin/nologin bnb; \
    fi

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Belt and braces for invariant (1) at the top of this file. `uv sync` installs the
# project editable, so /app/src is already on the path via a .pth file — but PYTHONPATH
# takes precedence over site-packages either way, which pins `import bnb` to the source
# tree even if the install mode ever changes. That's what keeps ASSETS_DIR at
# /app/assets instead of somewhere inside the interpreter's lib directory.
ENV PYTHONPATH=/app/src

# Serving-only by default: the tagging endpoints are the one API that writes back into
# the asset repository, and they pair with the read-only mount in docker/compose.yaml.
ENV BNB_ENABLE_TAGGING=0

WORKDIR /app

COPY --from=build --chown=${UID}:${GID} /app/.venv /app/.venv
COPY --from=build --chown=${UID}:${GID} /app/src /app/src

# The mount point for the host's asset repository. Created here so that an accidental
# run with no volume attached starts cleanly and fails per-request with a 404, rather
# than blowing up somewhere less legible.
RUN mkdir -p /app/assets && chown "${UID}:${GID}" /app/assets

USER ${UID}:${GID}

EXPOSE 8000

# /api/stream is the cheapest honest liveness probe: it reads the engine snapshot and
# touches neither the asset repository nor the audio path. Written in Python because
# the slim image has no curl and this doesn't warrant installing one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/stream', timeout=4)"]

# Single worker, deliberately. The service keeps its state in process — the module-level
# `engine` and the `sessions` registry of per-session StreamEngines — so a second worker
# would answer control calls for sessions it has never heard of. Scaling this out means
# moving that state somewhere shared, not adding --workers.
#
# --timeout-graceful-shutdown matters more than usual here: the stream generators loop
# for as long as their engine runs, so on SIGTERM uvicorn would otherwise wait for
# connections that are designed never to end.
CMD ["uvicorn", "bnb.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-graceful-shutdown", "5"]
