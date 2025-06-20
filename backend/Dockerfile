# FROM python:3.10-slim

# # Instala dependencias del sistema necesarias para ML (PyTorch, etc)
# RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# WORKDIR /backend

# COPY backend/app ./app
# COPY models ./app/models  
# COPY src ./src
# COPY ../../models ./models 
# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8000
# ENV PYTHONPATH=/backend/app:/backend/src

# EXPOSE 8000
# ENV PYTHONPATH=/backend/app:/src

# CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.10-slim

# Instala dependencias del sistema necesarias para ML (PyTorch, etc)
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /backend

# Copia la app (todo el código de backend/app)
COPY backend/app ./app

# Copia la carpeta src de la raíz a /backend/src
COPY src ./src

# Copia la carpeta models de la raíz a /backend/models
COPY models ./models

# Copia requirements.txt de la raíz
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# PYTHONPATH para que Python encuentre los módulos de app y src
ENV PYTHONPATH=/backend/app:/backend/src

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
