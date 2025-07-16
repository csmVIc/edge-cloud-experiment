import tensorflow as tf
from tensorflow.keras import layers, Model


def _make_divisible(v, divisor, min_value=None):
    """确保通道数能被divisor整除"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保舍入后的值不会减少超过10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """倒残差块"""
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # 扩展层
        x = layers.Conv2D(expansion * in_channels,
                         kernel_size=1,
                         padding='same',
                         use_bias=False,
                         activation=None,
                         name=prefix + 'expand')(x)
        x = layers.BatchNormalization(name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # 深度可分离卷积
    x = layers.DepthwiseConv2D(kernel_size=3,
                              strides=stride,
                              activation=None,
                              use_bias=False,
                              padding='same',
                              name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(name=prefix + 'depthwise_BN')(x)
    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # 投影层
    x = layers.Conv2D(pointwise_filters,
                     kernel_size=1,
                     padding='same',
                     use_bias=False,
                     activation=None,
                     name=prefix + 'project')(x)
    x = layers.BatchNormalization(name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def MobileNetV2(input_shape=(32, 32, 3), alpha=1.0, num_classes=100):
    """
    MobileNetV2模型
    
    Args:
        input_shape: 输入图像形状
        alpha: 宽度乘数，控制网络宽度
        num_classes: 分类数量
    
    Returns:
        Keras模型
    """
    img_input = layers.Input(shape=input_shape)
    
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(first_block_filters,
                     kernel_size=3,
                     strides=(1, 1),  # 对于CIFAR-100，第一层stride设为1
                     padding='same',
                     use_bias=False,
                     name='Conv1')(img_input)
    x = layers.BatchNormalization(name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    # MobileNetV2的倒残差块配置
    # [expansion, output_channels, num_blocks, stride]
    inverted_residual_setting = [
        [1, 16, 1, 1],
        [6, 24, 2, 1],  # 对于CIFAR-100，stride改为1
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    # 构建倒残差块
    for i, (expansion, output_channels, num_blocks, stride) in enumerate(inverted_residual_setting):
        for j in range(num_blocks):
            if j == 0:
                x = _inverted_res_block(x, expansion, stride, alpha, output_channels, i * 10 + j)
            else:
                x = _inverted_res_block(x, expansion, 1, alpha, output_channels, i * 10 + j)

    # 最后的卷积层
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters,
                     kernel_size=1,
                     use_bias=False,
                     name='Conv_1')(x)
    x = layers.BatchNormalization(name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)

    # 全局平均池化
    x = layers.GlobalAveragePooling2D()(x)
    
    # 分类层
    x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # 创建模型
    model = Model(img_input, x, name='mobilenetv2')
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = MobileNetV2()
    model.summary()
    print(f"总参数数量: {model.count_params():,}")