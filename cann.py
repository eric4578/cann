#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2  # 图片处理三方库，用于对图片进行前后处理
import numpy as np  # 用于对多维数组进行计算
from albumentations.augmentations import transforms  # 数据增强库，用于对图片进行变换

import acl  # acl 推理文件库


def sigmoid(x):
    y = 1.0 / (1 + np.exp(-x))  # 对矩阵的每个元素执行 1/(1+e^(-x))
    return y


def plot_mask(img, msk):
    """ 将推理得到的 mask 覆盖到原图上 """
    msk = msk + 0.5  # 将像素值范围变换到 0.5~1.5, 有利于下面转为二值图
    msk = cv2.resize(msk, (img.shape[1], img.shape[0]))  # 将 mask 缩放到原图大小
    msk = np.array(msk, np.uint8)  # 转为二值图, 只包含 0 和 1

    # 从 mask 中找到轮廓线, 其中第二个参数为轮廓检测的模式, 第三个参数为轮廓的近似方法
    # cv2.RETR_EXTERNAL 表示只检测外轮廓,  cv2.CHAIN_APPROX_SIMPLE 表示压缩水平方向、
    # 垂直方向、对角线方向的元素, 只保留该方向的终点坐标, 例如一个矩形轮廓只需要4个点来保存轮廓信息
    # contours 为返回的轮廓（list）
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上画出轮廓, 其中 img 为原图, contours 为检测到的轮廓列表
    # 第三个参数表示绘制 contours 中的哪条轮廓, -1 表示绘制所有轮廓
    # 第四个参数表示颜色, （0, 0, 255）表示红色, 第五个参数表示轮廓线的宽度
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1) 

    # 将轮廓线以内（即分割区域）覆盖上一层红色
    img[..., 2] = np.where(msk == 1, 255, img[..., 2])

    return img


###1、补全该处代码，初始化变量，包括图片输入，模型路径，类别数量，指定运算的设备
#----------------------****************-----------------------

# 初始化变量
pic_input = './image.png'  # 单张图片
model_path = "./unet_best.om"  # 模型路径
num_class = 2   # 类别数量, 需要根据模型结构、任务类别进行改变; 
device_id = 0   # 指定运算的Device
#----------------------****************-----------------------

print("init resource stage:")
##/2、补全该处代码，acl初始化部分，指定运算设备，创建context、stream
#----------------------****************-----------------------
# acl初始化
ret = acl.init()
ret = acl.rt.set_device(device_id)       # 指定运算的Device
context, ret = acl.rt.create_context(device_id)  # 显式创建一个Context，用于管理Stream对象 
stream, ret = acl.rt.create_stream()    # 显式创建一个Stream, 用于维护一些异步操作的执行顺序，确保按照应用程序中的代码调用顺序执行任务
#----------------------****************-----------------------
print("Init resource success")

##/3、补全该处代码，用acl加载离线模型，初始化模型信息，获取模型的描述信息
#----------------------****************-----------------------
# 加载模型
model_id, ret =  acl.mdl.load_from_file(model_path) # 加载离线模型文件, 返回标识模型的ID
model_desc = acl.mdl.create_desc()     # 初始化模型描述信息, 包括模型输入个数、输入维度、输出个数、输出维度等信息
ret = acl.mdl.get_desc(model_desc, model_id)      # 根据加载成功的模型的ID, 获取该模型的描述信息
#----------------------****************-----------------------
print("Init model resource success")

####/4、补全该处代码，前处理部分，用opencv读取图片，图像裁切，标准化，通道转换
#----------------------****************-----------------------
img_bgr = cv2.imread(pic_input)  # 读入图片
img = cv2.resize(img_bgr, (960, 960))# 将原图缩放到 960*960 大小

mean = img.mean()
stddev = img.std()

img = (img- mean) / stddev # 将像素值标准化（减去均值除以方差）
img = img.astype('float32') / 255  # 将像素值缩放到 0~1 范围内
img = img.transpose(2, 0, 1)  # 将形状转换为 channel first (3, 96, 96)
#----------------------****************-----------------------


# 准备输入数据集
input_list = [img, ]  # 初始化输入数据列表
input_num = acl.mdl.get_num_inputs(model_desc)  # 得到模型输入个数

####/5、补全该处代码，创建输入数据
#----------------------****************-----------------------
input_dataset = acl.mdl.create_dataset()    # 创建输入数据
#----------------------****************-----------------------

for i in range(input_num):
    input_data = input_list[i]  # 得到每个输入数据

    # 得到每个输入数据流的指针(input_ptr)和所占字节数(size)
    size = input_data.size * input_data.itemsize  # 得到所占字节数
    bytes_data=input_data.tobytes()  # 将每个输入数据转换为字节流
    input_ptr=acl.util.bytes_to_ptr(bytes_data)  # 得到输入数据指针

    model_size = acl.mdl.get_input_size_by_index(model_desc, i)  # 从模型信息中得到输入所占字节数
    # if size != model_size:  # 判断所分配的内存是否和模型的输入大小相符
    #     print(" Input[%d] size: %d not equal om size: %d" % (i, size, model_size) + ", may cause inference result error, please check model input")

    dataset_buffer = acl.create_data_buffer(input_ptr, size)  # 为每个输入创建 buffer
    _, ret = acl.mdl.add_dataset_buffer(input_dataset, dataset_buffer)  # 将每个 buffer 添加到输入数据中
print("Create model input dataset success")



# 准备输出数据集
output_size = acl.mdl.get_num_outputs(model_desc)  # 得到模型输出个数
output_dataset = acl.mdl.create_dataset()  # 创建输出数据

####/6、补全该处代码，根据模型输出个数，计算输出内存大小，分配内存，创建buffer，将buffer添加到输出数据中，并设计判断错误释放内存指令
#----------------------****************-----------------------
for i in range(output_size):
    size = acl.mdl.get_output_size_by_index(model_desc, i)  # 得到每个输出所占内存大小
    buf, ret = acl.rt.malloc(size, 2)  # 为输出分配内存。
    dataset_buffer = acl.create_data_buffer(buf, size)  # 为每个输出创建 buffer
    _, ret = acl.mdl.add_dataset_buffer(output_dataset, dataset_buffer)  # 将每个 buffer 添加到输出数据中
    if ret:  # 若分配出现错误, 则释放内存
        acl.rt.free(buf)
        acl.destroy_data_buffer(dataset_buffer)
#----------------------****************-----------------------
print("Create model output dataset success")


####/7、补全该处代码，使用acl命令推理得到输出，并将结果写入到output_dataset中
#----------------------****************-----------------------
# 模型推理, 得到的输出将写入 output_dataset 中
ret = acl.mdl.execute(model_id, input_dataset, output_dataset)

#----------------------****************-----------------------


# 解析 output_dataset, 得到模型输出列表
model_output = [] # 模型输出列表
for i in range(output_size):
    buf = acl.mdl.get_dataset_buffer(output_dataset, i)  # 获取每个输出buffer
    data_addr = acl.get_data_buffer_addr(buf)  # 获取输出buffer的地址
    size = int(acl.get_data_buffer_size(buf))  # 获取输出buffer的字节数
    byte_data = acl.util.ptr_to_bytes(data_addr, size)  # 将指针转为字节流数据
    dims = tuple(acl.mdl.get_output_dims(model_desc, i)[0]["dims"])  # 从模型信息中得到每个输出的维度信息
    output_data = np.frombuffer(byte_data, dtype=np.float32).reshape(dims)  # 将 output_data 以流的形式读入转化成 ndarray 对象
    model_output.append(output_data) # 添加到模型输出列表


x0 = 2200  # w:2200~4000; h:1000~2800
y0 = 1000
x1 = 4000
y1 = 2800
ori_w = x1 - x0
ori_h = y1 - y0


####/8、补全该处代码，定义_process_mask函数，通过传入mask_path，使用opencv读取mask图片
#----------------------****************-----------------------
def _process_mask(mask_path):
    # 手动裁剪
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # [y0:y1, x0:x1]
    mask= mask[y0:y1, x0:x1]
    return mask
#----------------------****************-----------------------


####/9、补全该处代码，取出模型推理结果，推理形状为（batchsize, num_class, height, width），读取mask数据，将处理后的输出画在原图上
#----------------------****************-----------------------
# 后处理
model_out_msk = model_output[0]  # 取出模型推理结果, 推理结果形状为 (1, 1, 96, 96),即（batchsize, num_class, height, width）
model_out_msk =  _process_mask('./mask.png')  # 抠图后的shape， hw
model_out_msk = sigmoid(model_out_msk[0][0])  # 将模型输出变换到 0~1 范围内
img_to_save = plot_mask(img_bgr, model_out_msk)  # 将处理后的输出画在原图上, 并返回
#----------------------****************-----------------------


# 保存图片到文件
cv2.imwrite('result.png', img_to_save)  



####/10、补全该处代码，释放输出资源，包括包括数据结构和内存，获取输出个数，获取每个输出的buffer，获取每个buffer的地址，手动释放acl.rt.malloc 所分配的内存，销毁每个输入buffer，销毁输入数据。
#----------------------****************-----------------------

# 释放输出资源, 包括数据结构和内存
num = acl.mdl.get_num_outputs(output_dataset)  # 获取输出个数
for i in range(num):
    data_buf = acl.get_dataset_buffer(output_dataset, i)   # 获取每个输出buffer
    if data_buf:
        data_addr = acl.get_data_buffer_addr(data_buf)     # 获取buffer的地址
        acl.free(data_addr)  # 手动释放 acl.rt.malloc 所分配的内存
        ret = acl.destory_data_buffer(data_buf)  # 销毁每个输出buffer (销毁 aclDataBuffer 类型)
ret = acl.mdl.destroy_dataset(output_dataset)  # 销毁输出数据 (销毁 aclmdlDataset类型的数据)
#----------------------****************-----------------------


# 卸载模型
if model_id:
    ret = acl.mdl.unload(model_id)

# 释放模型描述信息
if model_desc:
    ret = acl.mdl.destroy_desc(model_desc)

# 释放 stream
if stream:
    ret = acl.rt.destroy_stream(stream)

# 释放 Context
if context:
    ret = acl.rt.destroy_context(context)

# 释放Device
acl.rt.reset_device(device_id)
acl.finalize()
print("Release acl resource success")