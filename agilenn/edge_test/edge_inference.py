import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
from utils.edge_utils import *
from compress import compress_features_accurate

if __name__ == '__main__':

    # 一行代码初始化通信
    linklab = Linklab().initialize_connection()
    # 边缘模型地址
    edge_model_path = 'edge_model_savedmodel'
    # 正确加载AgileNN边缘模型组件
    print("Loading AgileNN edge model")
    # 加载模型
    edge_model = tf.saved_model.load(edge_model_path)
    print("Load complete！")


    # 3. 数据预处理 - 只选择100张图片
    data_dir = './data'  # 指定本地数据目录
    ds = tfds.load('cifar100', 
               data_dir=data_dir, 
               download=False, 
               as_supervised=True)
    std = tf.reshape((0.267, 0.256, 0.276), shape=(1, 1, 3))
    mean = tf.reshape((0.507, 0.487, 0.441), shape=(1, 1, 3))
    
    def valid_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = (x - mean) / std
        x = tf.image.resize(x, [96, 96])
        return x, y
    
    # 只取100张图片，逐张处理
    ds_test = ds['test'].take(100)\
                        .cache()\
                        .map(valid_prep, num_parallel_calls=tf.data.AUTOTUNE)\
                        .batch(1)
    
    # 4. 测试指标和推理延迟统计
    correct_predictions = 0
    total_samples = 0
    
    # 时间统计
    local_inference_times = []
    remote_inference_times = []
    fusion_times = []
    end_to_end_times = []
    compression_times = []
    # 压缩统计
    compressed_sizes = []
    compression_ratios = []

    # CIFAR-100类别名称
    cifar100_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                       'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                       'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                       'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                       'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                       'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                       'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                       'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                       'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                       'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                       'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                       'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                       'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                       'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                       'worm']
    
    print("开始逐张推理测试...")
    flag = 0
    # 5. 推理过程：逐张处理
    for batch_idx, (x, y) in enumerate(ds_test):
        sample_idx = batch_idx + 1
        true_label = int(y[0])
        
        # 端到端推理开始
        start_total = time.time()
        
        # 5.1 边 - 特征提取+特征分割+推理+量化云端特征
        start_time = time.time()
        edge_result = edge_model(x)
        local_logits = edge_result['local_logits']
        compressed_features = edge_result['compressed_features']
        local_inference_time = (time.time() - start_time) * 1000
        
        # 特征压缩
        compressed_data, compression_stats = compress_features_accurate(compressed_features)
        
        # 发送远程特征
        linklab.send_features(compressed_data)
        # 接受云端预测结果 ，并计算时间
        remote_logits = linklab.recv_data()

        # 5.7 结果融合
        start_time = time.time()
        final_logits = local_logits + remote_logits
        fusion_time = (time.time() - start_time) * 1000

        # 端到端推理结束, 本地推理时间，融合时间，端到端时间
        end_to_end_time = (time.time() - start_total) * 1000
        if flag == 1:
            local_inference_times.append(local_inference_time)
            fusion_times.append(fusion_time)
            end_to_end_times.append(end_to_end_time)
        
        # 获取压缩时间，压缩率，和压缩后大小
        compression_times.append(compression_stats['compression_time_ms'])
        compressed_sizes.append(compression_stats['compressed_size_bytes'])
        compression_ratios.append(compression_stats['compression_ratio'])

        # 预测结果
        predicted_label = int(tf.argmax(final_logits[0]))
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct_predictions += 1
        total_samples += 1
        
        # 计算特征大小
        original_size = compressed_features.shape[1] * compressed_features.shape[2] * compressed_features.shape[3] * 4

        # 打印每张图片的推理结果
        print(f"样本 {sample_idx:3d}: "
              f"预测={predicted_label:2d} | "
              f"真实={true_label:2d} | "
              f"{'✓' if is_correct else '✗'} | "
              f"端到端={end_to_end_time:6.2f}ms | "
              f"压缩时间={compression_stats['compression_time_ms']:5.2f}ms | "
              f"压缩前={original_size:6.0f}B | "
              f"压缩后={compression_stats['compressed_size_bytes']:5d}B | "
              f"压缩比={compression_stats['compression_ratio']:5.1f}x ")
        flag = 1
    linklab.finish_inference()

    # 关闭连接
    linklab.close()
    # 6. 输出结果
    accuracy = correct_predictions / total_samples
    
    print("AgileNN 测试结果 ")
    print(f"总样本数: {total_samples}")
    print(f"推理正确率: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    print()
    print("推理延迟统计 (ms/sample):")
    print(f"端到端时间:     {np.mean(end_to_end_times):6.2f} ± {np.std(end_to_end_times):5.2f}")
    print(f"本地推理:       {np.mean(local_inference_times):6.2f} ± {np.std(local_inference_times):5.2f}")
    print(f"压缩时间：       {np.mean(compression_times):6.2f} ± {np.std(compression_times):5.2f}")
    print(f"结果融合:       {np.mean(fusion_times):6.2f} ± {np.std(fusion_times):5.2f}")
    print("压缩前后统计 :")
    print(f"压缩前={original_size:6.0f}B ")
    print(f"压缩后={np.mean(compressed_sizes):6.2f} ± {np.std(compressed_sizes):5.2f}B")
    print(f"压缩比={np.mean(compression_ratios):5.1f} ± {np.std(compression_ratios):5.1f}x")

    