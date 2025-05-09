FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
# naistaluje tesseract a libg, co je kniznica ktore je potrebna pre openCV, nie je tu kvoli base imagu ktory je slim, teda nie je to cely python
RUN apt-get update && apt-get -y install \
  tesseract-ocr \
  libgl1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip 

# Inštalácia CPU-only PyTorch a potom Ultralytics (aby sa zabránilo stiahnutiu GPU verzie)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir ultralytics

EXPOSE 8080
ENV FLASK_APP=webApp.py
#CMD ["python", "./webApp.py"]
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]