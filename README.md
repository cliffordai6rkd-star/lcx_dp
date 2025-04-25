# HIROLRobotPlatform
A unified framework enabling cross-vendor robot collaboration through hardware abstraction and centralized task planning.

```
cd docker/[some specific robot]
docker compose up
```

```
docker exec -it [docker container id] /bin/bash
```

FR3
使用[panda_py](https://jeanelsner.github.io/panda-py/panda_py.html)进行控制
在控制主机上切换到实时内核，
```
pro attach
pro enable realtime-kernel
```

然后

```
pip install -r requirements.txt
```

