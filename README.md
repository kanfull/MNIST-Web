Go backend

docker build --platform=linux/amd64 -t go-mnist .
docker run -v ./history:/app/history -p 8081:8081 go-mnist
