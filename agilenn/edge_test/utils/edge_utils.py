import socket
import pickle
import time
import argparse
import os

class Linklab:
    def __init__(self, partition=None):
        self.client = None
        self.partition = partition
        self.args = None
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--partition','-pt',default=None,type=int)
        self.args = parser.parse_args()
        self.partition = parser.parse_args().partition
        return self.args
        
    def connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        cloud_test_service_host = os.environ.get('CLOUD_TEST_SERVICE_HOST')
        cloud_test_service_port = os.environ.get('CLOUD_TEST_SERVICE_PORT')

        ip = cloud_test_service_host
        port = int(cloud_test_service_port)
        # ip = "127.0.0.1"
        # port = int(5000)
        print(f"等待 Connecting to {ip}:{port}...")
        # 设置连接超时时间为1分钟
        self.client.settimeout(60)
        while True:
            try:
                self.client.connect((ip, port))
                break
            except socket.timeout:
                print(f"Connection to {ip}:{port} timed out after 60 seconds")
                time.sleep(10)
                # raise
            except Exception as e:
                print(f"Failed to connect to {ip}:{port}: {e}")
                time.sleep(10)
            
        # 连接成功后恢复默认的阻塞模式
        self.client.settimeout(None)
        
        return self.client

    def initialize_connection(self):
        """一键初始化：解析参数并建立连接"""
        self.parse_args()
        self.connect()
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
    # 结束推理
    def finish_inference(self):
        self.send_features('END')
    # 关闭连接
    def close(self):
        if self.client:
            self.client.close()
