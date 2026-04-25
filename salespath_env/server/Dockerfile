ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY salespath_env/ /app/salespath_env/

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "salespath_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]