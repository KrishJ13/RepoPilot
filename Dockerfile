FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
# CPU wheels avoid downloading CUDA libraries for this small local model.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY repopilot repopilot
EXPOSE 8000 8501
CMD ["uvicorn", "repopilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
