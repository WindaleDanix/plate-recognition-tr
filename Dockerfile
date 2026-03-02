# Stabil ve hafif bir imaj
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Paketleri yeni versiyon isimleriyle kuruyoruz
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları kopyala
COPY requirements.txt .

# pip güncellemesi ve paket kurulumu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Flask yapılandırması
ENV FLASK_APP=app.py
EXPOSE 7860

# Uygulamayı başlat
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
