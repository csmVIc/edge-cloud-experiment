import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
from utils.cloud_utils import *


def main():

    linklab = Linklab().initialize_server()
    # 边缘模型地址
    cloud_model_path = "saved_models/mobilenetv2_cifar100"
    # 正确加载AgileNN边缘模型组件
    print("Loading AgileNN edge model")
    # 加载模型
    loaded_model = tf.saved_model.load(cloud_model_path)
    cloud_model = loaded_model.signatures["serving_default"]
    print("Load complete！")
    
    # 推理
    remote_inference_times = []
    trans_latencys = []
    flag = 0
    while True:
        features, trans_latency = linklab.recv_features()
        if isinstance(features,str) and features == 'END':
            print("Received END signal, stopping inference...")
            break
        else:
            start_time = time.time()
            predictions = cloud_model(features)
            remote_inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            if flag == 1:
                remote_inference_times.append(remote_inference_time)
                trans_latencys.append(trans_latency)
            linklab.send_data(predictions)
            print(f"云端推理:       {remote_inference_time:5.2f} | "
                f"传输时延：       {trans_latency:5.2f} ")
        flag = 1

    
    print("推理延迟统计 (ms/sample):")
    print(f"云端推理:     {np.mean(remote_inference_times):6.2f} ± {np.std(remote_inference_times):5.2f}")
    print(f"传输时间:     {np.mean(trans_latencys):6.2f} ± {np.std(trans_latencys):5.2f}")
if __name__ == '__main__':
    main()


