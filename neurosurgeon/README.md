# 使用
只在边缘端推理
tc qdisc add dev eth0 root tbf rate 5mbit burst 12kbit latency 400ms; python edge_inference.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net --partition 22


云端推理命令：
python3 cloud_inference.py -i 0.0.0.0 -p 99 
# 测试数据
云端推理:

传输时间: 