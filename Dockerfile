FROM quay.io/unstructured-io/unstructured:latest

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt
RUN pip install -e .

EXPOSE 10000
CMD ["uvicorn", "lumos.server.app:app", "--host", "0.0.0.0", "--port", "10000"]
