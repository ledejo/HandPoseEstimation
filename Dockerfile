FROM python:3.11-slim
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without dev dependencies, no virtualenv in container)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

COPY src ./src
CMD ["python", "-m", "handpose.frontend.app"]
