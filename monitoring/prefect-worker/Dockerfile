FROM python:3.10-slim
WORKDIR /opt/prefect

# Installe Prefect 2.x
RUN pip install --no-cache-dir prefect==2.14.10

CMD ["prefect", "agent", "start", "-q", "default"]