Install deps: follow `https://github.com/JakubAndrysek/PySpaceMouse`.

如果在docker container中使用，尝试
1.在host和container中都进行如下操作

```
sudo echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"' > /etc/udev/rules.d/99-hidraw-permissions.rules
sudo usermod -aG plugdev $USER
newgrp plugdev
```

插拔usb线后，重新启动container