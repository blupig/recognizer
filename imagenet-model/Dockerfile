FROM python:3-slim
COPY . /app
# RUN apk add --no-cache --virtual build-base && \
#     pip3 install -r /app/requirements.txt && \
#     apk del build-base
WORKDIR /app
RUN pip3 install --no-cache-dir -r /app/requirements.txt
EXPOSE 8000
CMD ["python3", "api_server.py"]
