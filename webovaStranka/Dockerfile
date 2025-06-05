# pouzije sa lahsia verzia Pythonu
FROM python:3.12-slim     

# Zakaze bufferovanie vystupu Pythonu - teda vystuo neuklada do medzipamate, vypisuje ihned
ENV PYTHONUNBUFFERED=1
# naistaluje tesseract a libg, co je kniznica ktore je potrebna pre openCV, nie je tu kvoli base imagu ktory je slim, teda nie je to cely python
RUN apt-get update && apt-get -y install \
  tesseract-ocr \
  libgl1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# nastavi pracovny adresar v kontajneri na /app
WORKDIR /app
# skopiruje vsetky subory z lokalneho adresara do /app v kontajneri
COPY . /app

# Aktualizuje pip na najnovsiu verziu
RUN pip install --upgrade pip 

# instalacia CPU-only PyTorch a potom Ultralytics (zabrani to stiahnutie GPU verzie)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir ultralytics

# Oznamuje, ze kontajner bude pocuvat na porte 8080
EXPOSE 8080
# Nastavi premennu prostredia pre Flask na subor webApp.py
ENV FLASK_APP=webApp.py
# Spusti Flask aplikaciu na vsetkych rozhraniach (0.0.0.0) na porte 8080
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]