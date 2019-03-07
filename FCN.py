# -*- coding:utf-8 -*-

from __future__ import print_function # 兼容Python2和Python3
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

# 设置超参数
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224


# vgg网络部分， weights 是vgg网络各层的权重集合， image是被预测的图像的向量
def vgg_net(weights, image):
    # FCN的前五层就是vgg网络
    layers = (
        # 前两组卷积的形式都是：conv-relu-conv-relu-pool
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        # 后三组卷积的形式都是：conv-relu-conv-relu-conv-relu-pool
        # 多出的relu对网络中层进一步压榨提炼特征
        # 有种用更多的非线性变换对已有的特征去做变换，产生更多的特征的意味
        # 本身多了relu特征变换就加剧（权力释放）
        # 那么再用一个conv去控制（权力回收），也在指导网络中层的收敛
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]：实现用于计算机视觉领域的卷积神经网络(CNN)的MATLAB工具箱
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)# 前向传播结果 current
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug: # 是否开启debug模式 true / false
                utils.add_activation_summary(current)# 画图

        # vgg 的前5层的stride都是2，也就是前5层的size依次减小1倍
        # 这里处理了前4层的stride，用的是平均池化
        # 第5层的pool在下文的外部处理了，用的是最大池化
        # pool1 size缩小2倍
        # pool2 size缩小4倍
        # pool3 size缩小8倍
        # pool4 size缩小16倍
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current) # 平均池化
        net[name] = current #每层前向传播结果放在net中， 是一个字典

    return net # 保存了vgg每层的结果


# 预测流程，image是输入图像的向量，keep_prob是dropout rate
def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob 这里的keep_prob是保留概率，即我们要保留的结果所占比例
    它作为一个placeholder，在run时传入
    当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用
    训练中keep_prob=1时，模型对训练数据的适应性优于测试数据，就可以暴露出overfitting问题
    keep_prob=0.5时，dropout就发挥了作用

    :return:
    """

    # 获取训练好的vgg部分的model
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL) #返回VGG19模型中内容
    # model_dir Model_zoo/
    # MODEL_URL 下载VGG19网址

    mean = model_data['normalization'][0][0][0] # 数据归一化，获取图像均值
    mean_pixel = np.mean(mean, axis=(0, 1)) # 计算沿指定轴的算术平均值，RGB

    weights = np.squeeze(model_data['layers']) # 将表示向量的数组转换为秩为1的数组，压缩VGG网络中参数，把维度是1的维度去掉 剩下的就是权重

    # 将图像的向量值都减去平均像素值，进行 normalization
    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"): # 命名作用域

        # 传入权重参数和预测图像,计算前五层vgg网络的输出结果
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        # pool1 size缩小2倍
        # pool2 size缩小4倍
        # pool3 size缩小8倍
        # pool4 size缩小16倍
        # pool5 size缩小32倍
        pool5 = utils.max_pool_2x2(conv_final_layer)

        # w * x + b
        # # 初始化第6层的w、b
        # 7*7 卷积核的视野很大
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        # 在第六层没有进行池化，所以经过第六层后size缩小仍为32倍

        #初始化第七层的w、b
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        # 在第七层没有进行池化，所以经过第七层后size缩小仍为32倍

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8) #第八层卷积层，分类151类
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape() # 将pool4 1/16结果尺寸拿出来 做融合 [b,h,w,c]
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        #  输入为conv8特征图，对第8层的结果进行反卷积(上采样)，使得其特征图大小扩大两倍，并且特征图个数由NUM_OF_CLASSESS变为pool4的通道数
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        # 对应论文原文中的"2× upsampled prediction + pool4 prediction"
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1") # 进行融合 逐像素相加

        deconv_shape2 = image_net["pool3"].get_shape()
        # 对上一层上采样的结果进行反卷积(上采样),通道数也由上一层的通道数变为第3层的通道数
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        # 将上一层融合结果fuse_1再扩大两倍，输出尺寸和pool3相同
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        # 融合操作deconv(fuse_1) + pool3，对应论文原文中的"2× upsampled prediction + pool3 prediction"
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image) # 获得原始图像的大小，height、width和通道数
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])  #堆叠列表，反卷积输出尺寸，[b，原图H，原图W，类别个数]
        # 建立反卷积w[8倍扩大需要ks=16, 输出通道数为类别个数， 输入通道数pool3通道数]
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        # 反卷积，fuse_2反卷积，将上一层的结果转化为和原始图像相同size、通道数为分类数的形式数据。输出尺寸为 [b，原图H，原图W，类别个数]
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        # 目前conv_t3的形式为size为和原始图像相同的size，通道数与分类数相同
        # 这句我的理解是对于每个像素位置，根据第3维度（通道数）通过argmax能计算出这个像素点属于哪个分类
        # 也就是对于每个像素而言，NUM_OF_CLASSESS个通道中哪个数值最大，这个像素就属于哪个分类
        # 每个像素点有21个值，哪个值最大就属于那一类
        # 返回一张图，每一个点对于其来别信息shape=[b,h,w]
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
    # 从第三维度扩展 形成[b,h,w,c] 其中c=1, conv_t3最后具有21深度的特征图
    return tf.expand_dims(annotation_pred, dim=3), conv_t3


# 训练
# 下面是参照tf api
# Compute gradients of loss_val for the variables in var_list.
# This is the first part of minimize().
# loss: A Tensor containing the value to minimize.
# var_list: Optional list of tf.Variable to update to minimize loss.
#   Defaults to the list of variables collected in the graph under the key GraphKey.TRAINABLE_VARIABLES.
def train(loss_val, var_list):
    """
    :param loss_val:  损失函数
    :param var_list:  需要优化的值
    :return:
    """
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads) #返回迭代梯度


def main(argv=None):
    # dropout保留率
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    # 图像占坑，原始图像的向量
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    # 原始图像对应的标注图像的向量
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    # 输入原始图像向量、保留率，得到预测的标注图像和随后一层的网络输出。获得预测图[b,h,w,c=1]  结果特征图[b,h,w,c=151]
    pred_annotation, logits = inference(image, keep_probability)
    # 查看图像预处理的效果
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    # 计算预测标注图像和真实标注图像的交叉熵。空间交叉熵损失函数[b,h,w,c=151]  和labels[b,h,w]    每一张图分别对比。
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    # 返回需要训练的变量列表
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    # 定义损失函数。传入损失函数和需要训练的变量列表
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    # 定义合并变量操作，一次性生成所有摘要绘图数据
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    # data_dir = Data_zoo/MIT_SceneParsing/
    # training: [{image: 图片全路径， annotation:标签全路径， filename:图片名字}] [{}][{}]
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    # 将训练数据集、验证数据集的格式转换为网络需要的格式
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        # 读取图片 产生类对象 其中包含所有图片信息
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    # 加载之前的checkpoint
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir) # 训练断点回复
    if ckpt and ckpt.model_checkpoint_path: #如果存在checkpoint文件，则恢复sess
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            # 读取训练集的一个batch
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            # 迭代优化需要训练的变量，计算损失，网络开始运行
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                # 迭代10次打印显示
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                # 迭代500 次验证
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                # 保存模型
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)


    elif FLAGS.mode == "visualize":
        #可视化
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        # pred_annotation预测结果图
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()