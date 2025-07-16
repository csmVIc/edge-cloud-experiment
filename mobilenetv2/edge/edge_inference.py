import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
import argparse
from utils.edge_utils import *

def main():
    linklab = Linklab().initialize_connection()
    only_cloud = linklab.only_cloud
    # 数据集目录
    data_dir = "data/tensorflow_datasets"
    # 边缘模型地址
    edge_model_path = "saved_models/mobilenetv2_cifar100"
    # 正确加载AgileNN边缘模型组件
    print("Loading AgileNN edge model")
    # 加载模型
    loaded_model = tf.saved_model.load(edge_model_path)
    edge_model = loaded_model.signatures["serving_default"]
    print("Load complete！")
    
    # 3. 数据预处理 - 只选择100张图片
    ds = tfds.load('cifar100', 
               data_dir=data_dir, 
               download=False, 
               as_supervised=True)
    
    def valid_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        return x, y
    
    # 只取100张图片，逐张处理
    ds_test = ds['test'].take(100)\
                        .cache()\
                        .map(valid_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                        .batch(1)
    
    # 推理
    correct_count = 0
    total_count = 0
    end_to_end_times = []
    flag = 0
    for sample_idx, (x, y) in enumerate(ds_test):
        true_label = int(y.numpy()[0])
        
        # 推理时间测量
        start_time = time.time()
        if only_cloud:
            linklab.send_features(x)
            predictions = linklab.recv_data()
        else:
            predictions = edge_model(x)
        end_time = time.time()
        end_to_end_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 获取预测结果
        output_keys = list(predictions.keys())
        logits = predictions[output_keys[0]]
        predicted_label = int(tf.argmax(logits, axis=1).numpy()[0])
        
        is_correct = predicted_label == true_label
        if is_correct:
            correct_count += 1
        total_count += 1
        if(flag == 1):
            end_to_end_times.append(end_to_end_time)
        flag = 1
        # 打印每张图片的推理结果
        print(f"样本 {sample_idx:3d}: "
              f"预测={predicted_label:2d} | "
              f"真实={true_label:2d} | "
              f"{'✓' if is_correct else '✗'} | "
              f"端到端推理时间={end_to_end_time:6.2f}ms | "
              f"传输数据大小={f'{len(pickle.dumps(x)):6d} bytes' if only_cloud else 0}")

    # 输出最终准确率
    accuracy = correct_count / total_count
    print(f"\n总准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"正确数量: {correct_count}/{total_count}")
    print(f"端到端推理时间: {np.mean(end_to_end_times):5.2f}ms ± {np.std(end_to_end_times):5.2f}ms")

    # 如果使用云端推理，发送结束信号
    if only_cloud:
        print("Sending END signal to cloud...")
        linklab.send_features('END')
        linklab.close()
if __name__ == '__main__':
    main()


