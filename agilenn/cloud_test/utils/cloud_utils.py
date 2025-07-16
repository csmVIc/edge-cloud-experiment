import socket
import pickle
import time
import argparse

class Linklab:
    
    def __init__(self):
        self.ip = None
        self.port = None
        self.server = None
        self.client = None
        self.client_address = None
        self.args = None
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ip', '-i', default="127.0.0.1", type=str)
        parser.add_argument('--port', '-p', default=99, type=int)
        self.args = parser.parse_args()
        
        self.ip = self.args.ip
        self.port = self.args.port
        
        return self.args
    
    def start_server(self):
        # socket.AF_INET: 使用IPv4协议，socket.SOCK_STREAM: 使用TCP协议
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置地址重用选项，避免"Address already in use"错误
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.ip, self.port))
        self.server.listen(1)  # 监听队列大小为1
        print(f"Server started on {self.ip}:{self.port}")
        return self.server
    
    def accept_connection(self):
        self.client, self.client_address = self.server.accept()
        print(f"Connection from {self.client_address}")
        return self.client
    
    def initialize_server(self):
        """一键初始化：解析参数并启动服务器"""
        self.parse_args()
        self.start_server()
        self.accept_connection()
        return self
    
    def inference(self, model, inputs, device):
        inputs = inputs.to(device)
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(inputs)
        inference_latency = 0
        end_time = time.perf_counter()
        inference_latency = (end_time - start_time) * 1000  # 转换为毫秒
        return outputs, inference_latency
    # 发送数据
    def send_data(self, data):
        self.client.sendall(pickle.dumps(data))
    # 接收数据
    def recv_data(self):
        return pickle.loads(self.client.recv(1024))
    # 发送特征
    def send_features(self, features):
        data = pickle.dumps(features)
        # 先发送数据的长度
        data_size = len(data)
        # pickle.dumps()会将数据转化为字节流，这个序列化后的数据长度通常很小，5000字节的数据可能相当于几十字节
        self.client.sendall(pickle.dumps(data_size))    # 发送数据长度
        self.client.recv(1024).decode()

        self.client.sendall(data)
    def recv_features(self):
        # self.client.recv(1024)最多接收1024字节的数据长度，如果数据长度超过1024字节，则需要分多次接收
        # pickle.dumps()序列化 -> pickle.loads()反序列化
        data_size = pickle.loads(self.client.recv(1024))    # 接收数据长度
        self.client.sendall('ready'.encode())
        
        data = b''
        trans_latency = 0   # 传输时延
        while len(data) < data_size:
            start_time = time.perf_counter()
            chunk = self.client.recv(4096)
            end_time = time.perf_counter()
            if not chunk:
                break
            trans_latency += (end_time - start_time) * 1000
            data += chunk
        # print(f"Transmission latency: {trans_latency:.3f} ms")
        return pickle.loads(data), trans_latency
    
    def close(self):
        if self.client:
            self.client.close()
        if self.server:
            self.server.close()
    
