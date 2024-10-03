# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install the necessary packages
RUN pip install --no-cache-dir fastapi[standard] uvicorn transformers torch accelerate

# Install additional system dependencies for PyTorch
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

COPY . /app

EXPOSE 8000

# Run the FastAPI server
CMD ["fastapi", "run", "app.py", "--port", "8000"]
# uvicorn app:app --host 0.0.0.0 --port 8000
