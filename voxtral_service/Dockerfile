
# Use a modern CUDA version for the latest PyTorch and Transformers
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg libsndfile1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
