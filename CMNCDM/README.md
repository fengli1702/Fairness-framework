
使用 screen 会话管理工具
screen 是一个会话管理器，允许你创建一个虚拟终端，程序在这个虚拟终端里运行。即使 SSH 断开，screen 会话依然会保留。

启动 screen：

bash
复制代码
screen -S my_session


运行程序：

在 screen 会话内运行你的程序：

bash
复制代码
python your_script.py

分离 screen 会话：

按下 Ctrl+A，然后按 D 键，将当前 screen 会话分离。此时你可以安全地断开 SSH 连接。

重新连接 screen 会话：

重新连接服务器后，使用以下命令重新连接到 screen 会话：

bash
复制代码
screen -r my_session