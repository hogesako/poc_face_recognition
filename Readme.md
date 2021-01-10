# Jetson Nano(jetpack4.3)でのdlib build
コンパイル自体は通るが`Illegal instruction (core dumped)` が出るのでいったん諦めている
## default runtimeをnvidiaに変更
/etc/docker/daemon.json
```
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
```
systemd restart docker
```
## libcnn.hをコピーしてからビルド
```
cp /usr/include/libcnn.h ./include
docker build -f Dockerfile.jetson43 -t face_reco .
```