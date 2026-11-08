FROM python:3.12-slim

# uv binary from the official image (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv

WORKDIR /app

# Install runtime dependencies only (no dev group: no jupyter/plotting), exactly
# as pinned in uv.lock - the same versions the model was trained with.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Service code and the trained model
COPY common.py predict.py ./
COPY models/model.bin models/model.bin

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
