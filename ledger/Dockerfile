# 원장 연동 전용 컨테이너(3장 8.2, 2026-10-25 결정: web3py-ext를 학습 환경과 분리)
FROM python:3.12-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY service.py /app/service.py
ENTRYPOINT ["python", "/app/service.py"]
