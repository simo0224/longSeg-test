import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import torch
import nibabel as nib
import SimpleITK as sitk

def save_loss_curve(loss, path_folder, isTrain=True):
    title = 'Train Loss Curve' if isTrain else 'Val Loss Curve'
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    pic_name = 'train_loss.png' if isTrain else 'val_loss.png'
    path = os.path.join(path_folder, pic_name)
    plt.savefig(path)
    plt.close()

def save_acc_curve(acc, path_folder, isTrain=True):
    title = 'Train Accuracy Curve' if isTrain else 'Val Accuracy Curve'
    plt.figure()
    acc_arr = np.array(acc)
    acc_arr = acc_arr.T ## [class, epochs]
    for i in range(acc_arr.shape[0]):
        plt.plot(acc_arr[i,:], label=f"Class {i}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    pic_name = 'train_acc.png' if isTrain else 'val_acc.png'
    path = os.path.join(path_folder, pic_name)
    plt.legend([f"Class {i}" for i in range(acc_arr.shape[0])])
    plt.savefig(path)
    plt.close()

def save_acc_loss_curves(acc_train, acc_val, loss_train, loss_val, folder):
    save_loss_curve(loss_train, folder, isTrain=True)
    save_loss_curve(loss_val, folder, isTrain=False)
    save_acc_curve(acc_train, folder, isTrain=True)
    save_acc_curve(acc_val, folder, isTrain=False)


def get_closest_multiple_of_16(shape):
    """
    找到给定形状中每个维度的接近但不超过的16的倍数。
    :param shape: 当前数组的shape (D, H, W)
    :return: 裁剪后的shape (D', H', W')
    """
    # 计算每个维度裁剪后的大小，使其为16的倍数
    cropped_shape = [dim - (dim % 16) for dim in shape]
    
    return tuple(cropped_shape)

def crop_to_multiple_of_16(array, mod='single_seg'):
    """
    change the shape to fit UNet pool and upconv
    array:  (B, D, H, W)
    ouput:  (B, D', H', W')
    """
    original_shape = array.shape[1:]  # 忽略batch size
    # cropped_shape = get_closest_multiple_of_16(original_shape)
    # cropped_shape = tuple([160,160,160])
    if mod == 'long_seg':
        cropped_shape = tuple([160,160,160])
    elif mod == 'single_seg':
        cropped_shape = tuple([144,160,160])
        # cropped_shape = tuple([96,96,96])
    else:
        raise ValueError("mod should be 'long_seg' or 'single_seg'")
    # cropped_shape = tuple([144, 160, 160])
    # cropped_shape = tuple([96,96,96])

    ## calculate the difference between the original shape and the cropped shape
    diff = np.array(original_shape) - np.array(cropped_shape)
    crop_offsets = [diff[i] // 2 for i in range(3)]  # 每个维度的起始裁剪坐标
    D_interval = slice(crop_offsets[0], crop_offsets[0]+ cropped_shape[0])
    H_interval = slice(crop_offsets[1], crop_offsets[1]+ cropped_shape[1])
    W_interval = slice(crop_offsets[2], crop_offsets[2]+ cropped_shape[2])

    cropped_array = array[:, D_interval, H_interval, W_interval]
    
    return cropped_array, crop_offsets


def set_logging(path_file):
    # 获取当前时间，并格式化为适合文件名的字符串
    log_filename = os.path.join(path_file, "logging.txt")

    # 创建一个logger，如果已经存在处理器则不重复添加
    logger = logging.getLogger()

    # 检查是否已经有文件处理器，如果没有再添加
    if not logger.handlers:  # 确保处理器不重复添加
        logger.setLevel(logging.INFO)  # 设置全局日志级别

        # 创建一个文件处理器，将日志写入文件
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        # 创建一个控制台处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 定义日志格式
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理器添加到 logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def logging_recording(logger, opt):
    """
    Logs each configuration option and its value from the given `opt` object.

    Parameters:
    - logger: A configured logger to write logs.
    - opt: An object containing the parsed configuration options.
    """
    logger.info("---------------Training Configuration-------------")
    # 获取所有键的最大长度
    max_key_len = max(len(key) for key in vars(opt))
    
    # 遍历 opt 的所有属性，并记录每个参数的名称和值，确保 ":" 对齐
    for key, value in vars(opt).items():
        # 使用 ljust 格式化，使得所有的 ":" 对齐
        logger.info(f"{key.ljust(max_key_len)} : {value}")

def model_info_recording(logger, model):
    logger.info(f"-------------------TRAINING MODEL------------------")
    logger.info(model)
    logger.info(f"------------------------Over----------------------")


def cal_diceMetric_new(out, gt, num_class):
    '''
    使用 numpy 计算多类别 Dice 系数
    Arg: 
        out: (B, C=num_class, D, H, W) - 模型的输出 (torch.Tensor)
        gt: (B, D, H, W) - 真实标签掩码 (torch.Tensor)
        num_class: int - 类别数
    Return:
        dice_scores: (num_class,) - 每个类别的 Dice 系数
    '''
    # tensor to numpy
    with torch.no_grad():
        out_logits = out.cpu().numpy()
        mask = gt.cpu().numpy()

    pred_mask = np.argmax(out_logits, axis=1)  # [B, C, D, H, W] -> [B, D, H, W]

    dice_scores = np.zeros(num_class)

    for b in range(pred_mask.shape[0]): 
        pred_sample = pred_mask[b]
        mask_sample = mask[b]  # [B=b, C=1, D, H, W] -> [D, H, W]

        for cls in range(num_class):
            pred_i = (pred_sample == cls).astype(float)
            mask_i = (mask_sample == cls).astype(float)

            # 计算交集和 Dice 系数
            inter_area = np.sum(pred_i * mask_i)
            den = np.sum(pred_i) + np.sum(mask_i) + 1e-8
            num = 2 * inter_area + 1e-8
            dice_i = num / den

            dice_scores[cls] += dice_i

    # average Dice score
    dice_scores /= pred_mask.shape[0] 
    return dice_scores 
    
def cal_diceMetric(out_logits, mask, num_class):
    '''
        Here we ban the construction of gradient
        
        out_logits: (B=2, C=num_class, D, H, W). ## from the UNet3D model
        mask: (B=1, 1, D, H, W). ## the mask of latter timepoint
    '''
    with torch.no_grad(): ## ban the construction of gradient
        ## Here we count Background: 0
        _, pred_mask = torch.max(out_logits, dim=1) ## indices: (B=1, D, H, W)
        pred_mask = pred_mask.squeeze(0)
        mask = mask.squeeze(0).squeeze(0)
        dices = []
        for i in range(num_class):
            pred_i = (pred_mask==i).float()
            mask_i = (mask==i).float()

            interArea_i = torch.sum(mask_i * pred_i)
            dice_i = 2*interArea_i/(torch.sum(pred_i)+torch.sum(mask_i)+(1e-8))
            
            dices.append(dice_i.item())
        return np.array(dices)

def get_avgAcc(dice_tot, iter_num):
    # print(dice_tot)
    train_accuracy = dice_tot / iter_num
    for i in range(train_accuracy.shape[0]):
        train_accuracy[i] = np.around(train_accuracy[i], decimals=4)

    # train_accuracy = np.around(train_accuracy, decimals=2)  # 保留两位小数
    return train_accuracy

def print_classRatio(ratios, iter_num):
    ratios = ratios / iter_num
    log_message = "class ratio: [" + ", ".join(f"{ratios[i]:.4f}" for i in range(ratios.shape[0])) + "]"
    logging.info(log_message)

    print("calss 0 ratio:", np.round(ratios[0].item()*10000)/10000, end=';  ')
    print("calss 1 ratio:", np.round(ratios[1].item()*10000)/10000, end=';  ')
    print("calss 2 ratio:", np.round(ratios[2].item()*10000)/10000, end=';  ')
    print("calss 3 ratio:", np.round(ratios[3].item()*10000)/10000)



def save_pred_nii(pred, epoch, path_folder, source_nii_path):
    # 1. 转换 logits 为分类标签 (假设 argmax 方式进行多分类)
    pred_labels = torch.argmax(pred, dim=1)  # 对 C 维度执行 argmax，得到 (B, D, H, W)
    pred_labels = pred_labels.squeeze(0)  # 去掉 batch 维度，得到 (D, H, W)

    # 2. 将张量转换为 NumPy 数组，并确保它在 CPU 上
    pred_numpy = pred_labels.cpu().numpy().astype(np.uint8)  # 转为 uint8 标签值

    # 3. 加载源 NIfTI 文件以获取 affine 矩阵
    source_img = nib.load(source_nii_path)  # 加载源 NIfTI 文件
    original_affine = source_img.affine  # 提取源文件的 affine 矩阵
    original_shape = source_img.shape  # 获取源文件的形状

    # 4. 调整 affine 矩阵以适应裁剪后的图像
    cropped_shape = (160, 160, 160)  # 你的裁剪目标形状 (D, H, W)
    # 计算裁剪的起始点
    crop_offsets = [ (original_shape[i] - cropped_shape[i]) // 2 for i in range(3) ]

    # 调整 affine 的平移项 (affine 矩阵的第四列)
    adjusted_affine = original_affine.copy()
    for i in range(3):
        adjusted_affine[i, 3] += crop_offsets[i] * original_affine[i, i]

    # 5. 创建保存路径，文件名为 "pred_{epoch}.nii.gz"
    file_name = f"pred_{epoch}.nii.gz"
    file_path = os.path.join(path_folder, file_name)

    # 6. 使用 nibabel 保存为 .nii.gz 文件
    nii_img = nib.Nifti1Image(pred_numpy, adjusted_affine)  # 使用调整后的 affine
    nib.save(nii_img, file_path)  # 保存文件

    print(f"Prediction saved to: {file_path}")

def save_pred_nii_simple(pred, epoch, path_folder, affine=None, crop_offsets=None, mod='single_seg'):

    if mod == 'single_seg':
        if affine is not None:
            spacing = torch.stack(affine['Spacing'])
            transpose_spacing = spacing.T
            spacing = [tuple(row.tolist()) for row in transpose_spacing]

            origin = torch.stack(affine['Origin'])
            transpose_origin = origin.T
            origin = [tuple(row.tolist()) for row in transpose_origin]

            direction = torch.stack(affine['Direction'])
            transpose_direction = direction.T
            direction = [tuple(row.tolist()) for row in transpose_direction]


        crop_offset = torch.stack(crop_offsets)
        transpose_crop_offset = crop_offset.T
        crop_offset = [tuple(row.tolist()) for row in transpose_crop_offset]

        for i in range(pred.shape[0]):
            # 1. 转换 logits 为分类标签 (假设 argmax 方式进行多分类)
            # 假设 pred 大小为 (B=1, C=1, D, H, W)，我们去掉 batch 维度和 channel 维度
            pred_i = pred[i]  # 得到 (C, D, H, W)
            pred_labels = torch.argmax(torch.softmax(pred_i, dim=0), dim=0)  # 对 C 维度执行 argmax，得到 (D, H, W)
            
            # 去掉 batch 维度，得到 (D, H, W)
            
            # 2. 将张量转换为 NumPy 数组，并确保它在 CPU 上
            pred_numpy = pred_labels.cpu().numpy().astype(np.int32)
            
            # 3. 创建保存路径，文件名为 "pred_{epoch}.nii.gz"
            # file_name = f"pred_{epoch}.nii.gz"
            file_name = f"pred.nii.gz"
            file_path = os.path.join(path_folder[i], file_name)
            
            if affine is None:
                affine_i = np.eye(4) 
            else:
                affine_i = {}
                affine_i['Spacing'] = spacing[i]
                affine_i['Origin'] = origin[i]
                affine_i['Direction'] = direction[i]
            
            

            crop_offsets_i = crop_offset[i]
            save_nii(pred_numpy, affine_i, file_path, crop_offsets_i)
    else:
        
        # 1. 转换 logits 为分类标签 (假设 argmax 方式进行多分类)
        # 假设 pred 大小为 (B=1, C=1, D, H, W)，我们去掉 batch 维度和 channel 维度
        pred_labels = torch.argmax(torch.softmax(pred, dim=1), dim=1)  # 对 C 维度执行 argmax，得到 (D, H, W)
        
        # 去掉 batch 维度，得到 (D, H, W)
        pred_labels = pred_labels.squeeze(0)
        
        # 2. 将张量转换为 NumPy 数组，并确保它在 CPU 上
        pred_numpy = pred_labels.cpu().numpy().astype(np.int32)
        
        # 3. 创建保存路径，文件名为 "pred_{epoch}.nii.gz"
        file_name = f"pred_{epoch}.nii.gz"
        file_path = os.path.join(path_folder, file_name)
        
        if affine is None:
            affine_i = np.eye(4) 
            
        
        save_nii(pred_numpy, affine, file_path, crop_offsets)



def save_nii_simple(pred, path_folder, isMask=False, affine=None, crop_offsets=None):
    
    # 得到 (D, H, W)
    pred_labels = pred.squeeze(0)
    
    pred_numpy = pred_labels.cpu().numpy() 
    
    if isMask:
        file_name = f"mask.nii.gz"
    else:
        file_name = f"image.nii.gz"
    file_path = os.path.join(path_folder, file_name)
    
    if affine is None:
        affine = np.eye(4) 
    
    save_nii(pred_numpy, affine, file_path, crop_offsets)



def normalize_img(array, percentile=99.99, zero_centered=True, verbose=False):
    min_ = np.min(array)
    # max_ = np.percentile(array, percentile)
    max_ = np.max(array)
    #print(max_, min_)
    if verbose:
        print('original range: {},{}'.format( min_, max_))

    if max_ - min_ > 0:
        array = (array - min_)/ (max_ - min_)  # [0,1] normalized
    if zero_centered: # [-1,1] normalized
        array = array * 2 - 1
    if verbose:
        print('normalized to range {}, {}'.format(np.min(array), np.max(array)))
    return array


def get_nii(input_filename):
    image = sitk.ReadImage(input_filename)
    affine = {}
    affine['Spacing'] = image.GetSpacing()
    affine['Origin'] = image.GetOrigin()
    affine['Direction'] = image.GetDirection()
    image_data = sitk.GetArrayFromImage(image)
    return image_data, affine

def save_nii(image_data, affine, output_filename, crop_offsets=None):

    # 创建新的图像
    new_image = sitk.GetImageFromArray(image_data)
    new_image.SetSpacing(affine['Spacing'])      # 设置间距
    new_image.SetDirection(affine['Direction'])         # 设置方向（仿射矩阵）
    if crop_offsets is None:
        crop_offsets = [0, 0, 0]
        new_origin = np.array(affine['Origin']) + np.array(crop_offsets) * np.array(affine['Spacing'])
        new_image.SetOrigin(new_origin.tolist())        # 设置原点
    else:
        new_image.SetOrigin(affine['Origin'])        # 设置原点
    # 保存为新的 .nii.gz 文件
    sitk.WriteImage(new_image, output_filename)

