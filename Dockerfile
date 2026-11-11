FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu . \
 && useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
# role is chosen by the command: node or gateway
CMD ["python", "-m", "scllm.transport.node_server"]
