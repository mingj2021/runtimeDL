# runtimeDL
```
git clone https://github.com/mingj2021/runtimeDL.git
docker build -t dev:runtimeDL -f ./Dockerfile.dev .
docker run -it --rm --gpus all -v /yourdirs/runtimeDL:/workspace/runtimeDL  dev:runtimeDL
# attach running container by vscode
# build
# run sampleAsyscTRTYolo samples
sampleAsyscTRTYolo -is_seg=1|0
```
