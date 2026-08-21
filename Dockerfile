# The lab in a container. Two services come out of this one image — the panel
# and the read-only Inspector — because they differ only in which app uvicorn
# is pointed at; compose.yaml names both.
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
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra local-embeddings

COPY . .
RUN uv sync --frozen --extra local-embeddings

# The mount points, created here so they exist and are writable even when a run
# starts without the volumes attached.
RUN mkdir -p /app/.runs /app/databases /app/.datasets

# Documentation only — publishing is compose.yaml's job. The numbers are owned
# by src/raglab/dashboard/cli/serve.py (PANEL_PORT, INSPECTOR_PORT); these two
# lines and compose.yaml's are the only places outside it that repeat them.
EXPOSE 9002 9003

# Not the `raglab` entry point, and that is deliberate: serve.py calls
# uvicorn.run() without a host, so uvicorn's default 127.0.0.1 applies — inside
# a container that is the container's own loopback and no published port would
# ever reach it. Binding every interface is safe here only because compose.yaml
# publishes to 127.0.0.1 on the host side.
CMD ["uv", "run", "uvicorn", "raglab.dashboard.panel_server:app", \
     "--host", "0.0.0.0", "--port", "9002"]
