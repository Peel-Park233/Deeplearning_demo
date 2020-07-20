import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yad2k.models.keras_yolo import yolo_head, yolo_head, yolo_boxes_to_corners,  preprocess_true_boxes, yolo_loss, yolo_body
import yolo_utils
import imageio
#  测试
# a = np.random.randn(19 * 19, 5, 1)
# b = np.random.randn(19 * 19, 5, 80)
# c = a * b
# print(c)
tf.disable_v2_behavior()


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    # 第一步：计算锚框的得分
    box_scores = box_confidence * box_class_probs
    # 第二步：找到最大值的锚框的索引以及对应的最大值的锚框的分数
    box_classes = K.argmax(box_scores, axis=-1)  # 在pyhton中，-1代表倒数第一个，就是i选最后一个维度的最大值
    box_class_scores = K.max(box_scores, axis=-1)
    # 第三步：根据阈值创建掩码
    filtering_mask = (box_class_scores >= threshold)
    # print(filtering_mask)
    # 对scores, boxes以及classes使用掩码
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf. boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


# with tf.Session() as test_a:
#     box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
#     boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
#     box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
#     scores, boxes, classes, filtering_mask = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)
#
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.shape))
#     print("boxes.shape = " + str(boxes.shape))
#     print("classes.shape = =" +str(classes.shape))
#     # print(filtering_mask)
#
#     test_a.close()

def iou(box1, box2):

    # 计算香蕉的区域面积
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi1 - xi2) * (yi1 - yi2)

    #   计算并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    #   计算交并比
    iou = inter_area / union_area

    return iou


# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4)
# print("iou = " + str(iou(box1, box2)))

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype="int32")    # 用于tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # 初始化变量max_boxes_tensor

    # 使用使用tf.image.non_max_suppression()来获取与我们保留的框相对应的索引列表
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


# with tf.Session() as test_b:
#     scores = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
#     boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed=1)
#     class_1 = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
#     scores, boxes, classes = yolo_non_max_suppression(scores, boxes, class_1)
#
#     print("scores[2] = " + str(scores[2].eval()))   # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))
#
#     test_b.close()

def yolo_eval(yolo_outputs, image_shape=(720., 1280.),
              max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """

    参数：
        yolo_outputs - 编码模型的输出（对于维度（608，608，3）的图片的维度，这里是（608，608）
        max_boxes - 整数，预测的框锚数量的最大值
        score_threshold - 实数， 可能行阈值
        iou_threshold - 实数， 交并比的阈值

    返回：
        scores - tensor类型，维度（，None）,每个锚框的预测可能值
        boxes - tensor类型， 维度（4， None）,预测的锚框的坐标
        classes - tensor类型，维度（， None），每个锚框预测分类
    """
    # 获取YOLO模型的输出
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    # 中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    # 可信度分支过滤
    scores, boxes, classes= yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    #   缩放框锚，以适应原图像
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    #使用非最大意志
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes





# with tf.Session() as test_c:
#     yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
#                     tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
#     scores, boxes, classes = yolo_eval(yolo_outputs)
#
#     print("scores[2] = " + str(scores[1].eval()))
#     print("boxes[2] = " + str(boxes[1].eval()))
#     print("classes[2] = " + str(classes[1].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))
#
#     test_c.close()

sess = K.get_session()
class_names = yolo_utils.read_classes("model_data/coco_classes.txt")
anchors = yolo_utils.read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)  # 我真的吐了，少个点数据类型不一样了
yolo_model = load_model("model_data/yolov2.h5")
# yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
print(str(yolo_outputs))
# box_confidence, box_xy, box_wh,, box_class_probs = yolo_outputs
# yolo_outputs = (box_confidence, box_xy, box_wh, box_class_probs)
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

# image, image_data = yolo_utils.preprocess_image("images/0002.jpg", model_image_size=(608, 608))


def predict(sess, image_file, is_show_info=True, is_plot=True):
    # 图像预处理
    image, image_data = yolo_utils.preprocess_image("images/" + image_file, model_image_size=(608, 608))
    #   运行并选择占位符
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input:image_data, K.learning_phase(): 0})
    # 打印预测信息
    if is_show_info:
        print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框。")

    # 指定要绘制的边界框的颜色
    colors = yolo_utils.generate_colors(class_names)

    # 在图中绘制边界框
    yolo_utils.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    # 保存已经绘制了边界框的图
    image.save(os.path.join("out", image_file), quality=100)

    # 打印出已经绘制了边界框的图
    if is_plot:
        # output_image = scipy.misc.imread(os.path.join("out", image_file))
        output_image = imageio.imread(os.path.join("out", image_file))
        plt.imshow(output_image)

    return out_scores, out_boxes, out_classes


out_scores, out_boxes, out_classes = predict(sess, "0077.jpg")















