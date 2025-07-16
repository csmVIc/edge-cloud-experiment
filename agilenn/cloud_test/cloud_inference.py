import tensorflow as tf
import numpy as np
import time
from utils.cloud_utils import *
from decompress import decompress_features_accurate

if __name__ == '__main__':
    # 一行代码初始化服务器
    linklab = Linklab().initialize_server()

    # 云端模型地址
    cloud_model_path = 'cloud_model_savedmodel'
    # 正确加载AgileNN边缘模型组件
    print("Loading AgileNN cloud model")
    # 加载模型
    cloud_model = tf.saved_model.load(cloud_model_path)
    print("Load complete！")
    remote_inference_times = []
    trans_latencys = []
    decompression_times = []
    flag = 0
    # 循环处理推理请求
    while True:
        # 接收边缘数据
        compressed_features,trans_latency = linklab.recv_features()
            
        if isinstance(compressed_features,str) and compressed_features == 'END':
            print("Received END signal, stopping inference...")
            break
        else:
            decompressed_features, decompression_time = decompress_features_accurate(compressed_features)
            # 云端推理
            start_time = time.time()
            remote_logits = cloud_model(decompressed_features)
            remote_inference_time = (time.time() - start_time) * 1000
            # 推理结束 - 传输时间，云端推理时间
            if flag == 1:
                trans_latencys.append(trans_latency)
                remote_inference_times.append(remote_inference_time)
                decompression_times.append(decompression_time)
            print(f"云端推理:       {remote_inference_time:5.2f} | "
                  f"解压缩：        {decompression_time:5.2f} | "
                  f"传输时延：       {trans_latency:5.2f} ")
            # 发送云端预测结果 - 612B
            linklab.send_data(remote_logits)
        flag = 1
            
    print("推理延迟统计 (ms/sample):")
    print(f"云端推理:     {np.mean(remote_inference_times):6.2f} ± {np.std(remote_inference_times):5.2f}")
    print(f"传输时间:     {np.mean(trans_latencys):6.2f} ± {np.std(trans_latencys):5.2f}")
    print(f"解压缩时间:   {np.mean(decompression_times):6.2f} ± {np.std(decompression_times):5.2f}")

    # 关闭连接
    linklab.close()