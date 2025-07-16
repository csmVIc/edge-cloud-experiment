import tensorflow as tf
import numpy as np
import time
import zlib
import pickle


def compress_features_accurate(quantized_features):
    """
    精度保持的压缩方案 - 直接压缩量化后的特征值
    Args:
        quantized_features: 量化后的特征 (batch, H, W, C)
    Returns:
        compressed_data: 压缩后的数据
        compression_stats: 压缩统计信息
    """
    start_time = time.time()
    
    # 将量化特征转换为bytes并压缩
    features_bytes = quantized_features.numpy().tobytes()
    compressed_bytes = zlib.compress(features_bytes, level=3)
    
    compressed_data = {
        'method': 'accurate_lossless',
        'data': compressed_bytes,
        'shape': quantized_features.shape.as_list(),
        'dtype': str(quantized_features.dtype)
    }
    
    compressed_data_bytes = pickle.dumps(compressed_data)
    compression_time = (time.time() - start_time) * 1000
    
    # 计算压缩统计
    original_size = quantized_features.numpy().nbytes
    compressed_size = len(compressed_data_bytes)
    
    stats = {
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'compression_ratio': original_size / compressed_size,
        'space_saved_percent': (1 - compressed_size/original_size) * 100,
        'compression_time_ms': compression_time
    }
    
    return compressed_data_bytes, stats


def decompress_features_accurate(compressed_data):
    """
    精确解压缩特征
    Args:
        compressed_data: 压缩后的数据
    Returns:
        reconstructed_features: 精确重建的特征张量
        decompression_time: 解压时间
    """
    start_time = time.time()
    
    data = pickle.loads(compressed_data)
    
    # 解压缩
    decompressed_bytes = zlib.decompress(data['data'])
    
    # 重建张量 - 保持完全精度
    features_np = np.frombuffer(decompressed_bytes, dtype=np.float32)
    features_np = features_np.reshape(data['shape'])
    
    reconstructed = tf.constant(features_np, dtype=tf.float32)
    
    decompression_time = (time.time() - start_time) * 1000
    
    return reconstructed, decompression_time