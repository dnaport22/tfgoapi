build:
	rm -rf bin/*
	go build main.go && mv ./main bin && cp -r data/ bin/data