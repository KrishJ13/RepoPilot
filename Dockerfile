FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY repopilot repopilot
EXPOSE 8000 8501
CMD ["uvicorn", "repopilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
