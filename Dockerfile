# The lab in a container: one image, one service. The Inspector is a path on
# the lab (`/inspector`), not a second process, so there is nothing left for a
# second command to point at.
#
# The image carries no corpus state: `.runs/`, `databases/` and `.datasets/`
# are the durable artifacts and arrive as mounts, never as layers. See
# .dockerignore, which keeps them (and `.env`) out of the build context.
FROM python:3.13-slim

# uv, because that is how this project is run everywhere else. Pinned rather
# than `latest` so a rebuild months from now resolves the same way, and pinned
# to *this* version because uv.lock is `revision = 3` — a uv old enough not to
# know that revision refuses the lock instead of reading it.
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /usr/local/bin/uv

# Byte-compiled once at build time rather than on every cold start, and copied
# rather than linked because the venv does not share a filesystem with a cache
# mount here.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Dependencies before source, so editing a Python file does not re-resolve or
# re-download roughly a gigabyte of torch. `--frozen` holds uv.lock to its word:
# a build that would need to change the lock fails instead of drifting.
#
# Both embedding extras, not just the default embedder's. `semantic` carries
# fastembed, and the cross-encoder reranker imports its class from inside it
# (retrieve_fuse_rerank_grade.cross_encoder_available), so an image without
# that extra reports two capabilities unavailable rather than one.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra local-embeddings --extra semantic

# ── the two CLI backends, off by default ─────────────────────────────────────
# `RAGLAB_LLM=claude` and `RAGLAB_LLM=codex` shell out to a command
# (llm_backends/cli_subprocess_chat.py), and the lab offers a CLI model only
# when that command is on PATH *in here* — a host binary is Mach-O and cannot
# run on this image, so copying one in is not an option. The CLIs publish npm
# packages, which can. Off by default because Node is weight the Ollama and
# OpenRouter backends never need:
#
#     docker compose build --build-arg INSTALL_CLI=true
#
# Installing the command is half the job — a logged-in credential still has to
# arrive from outside. compose.yaml carries the mounts, commented, and
# .env.example says what each backend needs.
ARG INSTALL_CLI=false
RUN if [ "$INSTALL_CLI" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends nodejs npm \
      && rm -rf /var/lib/apt/lists/* \
      && npm install -g @openai/codex @anthropic-ai/claude-code; \
    fi

COPY . .
RUN uv sync --frozen --extra local-embeddings --extra semantic

# The mount points, created here so they exist and are writable even when a run
# starts without the volumes attached.
RUN mkdir -p /app/.runs /app/databases /app/.datasets

# Documentation only — publishing is compose.yaml's job. The number is owned
# by src/raglab/dashboard/cli/serve.py (PANEL_PORT); this line and
# compose.yaml's are the only places outside it that repeat it.
EXPOSE 9002

# Not the `raglab` entry point, and that is deliberate: serve.py calls
# uvicorn.run() without a host, so uvicorn's default 127.0.0.1 applies — inside
# a container that is the container's own loopback and no published port would
# ever reach it. Binding every interface is safe here only because compose.yaml
# publishes to 127.0.0.1 on the host side.
#
# `served_lab:app`, not `panel_server:app`: the composed app is what mounts the
# Inspector at /inspector. Pointing this at the bare panel app would still
# answer on :9002, but /inspector — and the widget's shared static files it
# needs — would 404.
CMD ["uv", "run", "uvicorn", "raglab.dashboard.served_lab:app", \
     "--host", "0.0.0.0", "--port", "9002"]
