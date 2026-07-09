# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /code
WORKDIR /code

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install dependencies (use CPU-only PyTorch to save massive space in Hugging Face Spaces)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . /code

# Hugging Face Spaces requires setting up a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860 which is required by Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI app via Uvicorn on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
