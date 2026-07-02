CGO_ENABLED=1 go build -ldflags="-extldflags '-Wl,-rpath,/mnt/d/Code/glim/bin'" -o build/glim ./cmd/glim

./build/run.sh
