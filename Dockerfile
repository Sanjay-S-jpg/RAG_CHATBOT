# Use a lightweight Python Linux environment
FROM python:3.10-slim

# Create a folder inside the cloud server
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the cloud
COPY . .

# Hugging Face Spaces ALWAYS listens on port 7860. We force Chainlit to use it.
EXPOSE 7860
CMD ["chainlit", "run", "app_ui.py", "--host", "0.0.0.0", "--port", "7860"]