# Delivery Stride -- fast bowling analysis from one side-on video.
FROM python:3.11-slim

# OpenCV and MediaPipe need these; the -headless wheel still wants libGL.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as uid 1000 and expects a writable home.
RUN useradd -m -u 1000 app
WORKDIR /home/app
ENV PATH=/home/app/.local/bin:$PATH PYTHONUNBUFFERED=1

COPY --chown=app:app requirements.txt .
USER app
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app:app analyze.py measure_deliveries.py ./
COPY --chown=app:app models ./models
COPY --chown=app:app web ./web
RUN mkdir -p web/uploads

# Spaces serves on 7860. One worker: each request holds a MediaPipe graph and
# a 300-tree forest, so parallel workers would multiply memory for no gain.
ENV PORT=7860
EXPOSE 7860
CMD ["gunicorn","--chdir","web","--bind","0.0.0.0:7860","--workers","1",\
     "--threads","4","--timeout","300","app:app"]
