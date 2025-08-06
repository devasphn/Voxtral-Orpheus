FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install git, ffmpeg, and other build tools first
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    git \
    ffmpeg \
    libsndfile1
    
COPY requirements.txt .
# The git+https... dependency requires git to be installed
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
