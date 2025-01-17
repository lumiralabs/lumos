FROM quay.io/unstructured-io/unstructured:latest

ENV UNSTRUCTURED_API_URL="https://api.unstructuredapp.io/general/v0/general"
ENV UNSTRUCTURED_API_KEY="5821gQyHOpVdAzXr92OsQOWSOJAsQ9"

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt
RUN pip install -e .

EXPOSE 10000
CMD ["uvicorn", "lumos.server.app:app", "--host", "0.0.0.0", "--port", "10000"]
