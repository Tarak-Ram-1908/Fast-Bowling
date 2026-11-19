# Delivery Stride -- fast bowling analysis from one side-on video.
FROM python:3.13-slim

# OpenCV needs libGL and glib; mediapipe imports sounddevice, which needs
# PortAudio.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libportaudio2 ffmpeg \
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

# Spaces serves on 7860. Exactly one worker: jobs and the analysis queue live
# in that process's memory, and a second worker would neither see them nor
# share the queue. Threads only serve pages; analysis runs one at a time.
ENV PORT=7860
EXPOSE 7860
CMD ["gunicorn","--chdir","web","--bind","0.0.0.0:7860","--workers","1",\
     "--threads","4","--timeout","300","app:app"]
