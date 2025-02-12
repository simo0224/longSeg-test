from torchvision import transforms
from src.dataset.longData import *
from my_models.resUNet3D import *
from torch.utils.data import DataLoader
import torch
from src.tools.utils import *
import datetime
import os
from src.tools.loss_new import myDiceLoss
import random
import my_configs.train_config as train_config

## get config options
train_config = train_config.trainConfig()
parser = train_config.initialize(train_config.parser)
opt = parser.parse_args()

## set logger and get config
current_time = datetime.datetime.now().strftime("%m%d_%H-%M")
path_log = os.path.join(opt.output_dir, f"log_{current_time}") ## output log path
if not os.path.exists(path_log):
    os.mkdir(path_log)
logger = set_logging(path_log)
logging_recording(logger, opt)


train_data = CustomDataset(opt.train_path, path_log=path_log, isTrain=True, mod=opt.mod)
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=opt.isDrop)
val_data = CustomDataset(opt.val_path, path_log=path_log, isTrain=False, mod=opt.mod)
val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=opt.isDrop)


device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')

model = myResUNet3D(1, opt.classes, mod=opt.mod).to(device)

## print the model info
model_info_recording(logger, model)

criterion = myDiceLoss(opt).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=opt.lr,            # 学习率
    weight_decay=1e-2,  # 权重衰减
    betas=(0.9, 0.999), # 动量参数，可选，通常不变
    eps=1e-8            # 防止数值不稳定的小常数，可选，通常不变
)

bst_loss_val = 99999
bst_loss_train = 99999
loss_train_list = []
loss_val_list = []  
bst_acc = 0
acc_train_list = []
acc_val_list = []

for epoch in range(opt.max_epoch):
    
    # set random seed
    seed = 42  # 选择一个常数作为种子
    torch.manual_seed(seed+epoch)
    random.seed(seed+epoch)
    np.random.seed(seed+epoch)
    loss_epoch = 0
    model.train()

    iter_num = 0
    dice_tot = 0
    ratios = np.zeros(opt.classes)
    for idx, (image, mask_train, affine_mask_train, save_path_nii, crop_offset_mask, tp_name) in enumerate(train_loader):
        print(f"tp_name: {tp_name}")
        ## normalization
        # image = normalize_img(image)

        iter_num += 1
        image, mask_train = image.to(device), mask_train.to(device) ## shape 都是 (B, C=1 D, H, W)
        ## see ratio of each class
        ratios += [torch.sum(mask_train==i).item()/(mask_train.numel()+(1e-5)) for i in range(opt.classes)]

        ## forward
        output = model(image)
        save_pred_nii_simple(output, epoch, save_path_nii, affine_mask_train, crop_offset_mask, opt.mod)
        mask_train = mask_train.squeeze(1).long()
        loss_iter = criterion(output, mask_train)
        # loss_iter, mean, fore_label = criterion(output, mask_train)
        loss_epoch += loss_iter.item()

        ## backward
        optimizer.zero_grad()
        loss_iter.backward()
        optimizer.step()

        # 计算训练准确率
        # predicted = torch.sigmoid(output)  # 转换为概率计预测正确的标签数
        # diec_loc = cal_diceMetric(predicted, mask_train, opt.classes)
        diec_loc = cal_diceMetric_new(output, mask_train, opt.classes)
        dice_tot += diec_loc

    # print_classRatio(ratios, iter_num)

    if loss_epoch < bst_loss_train:
        bst_loss_train = loss_epoch
        bst_mode_path_train = os.path.join(path_log, 'model_train_bst.path')
        torch.save(model.state_dict(), bst_mode_path_train)

    ## loss & accuracy
    loss_train_list.append(loss_epoch/iter_num)
    train_accuracy = get_avgAcc(dice_tot, iter_num)
    logging.info(f"Epoch: {epoch+1}, training Loss: {loss_epoch}, training acc: {train_accuracy}")
    acc_train_list.append(train_accuracy)
    

    ## EVAL
    if epoch % opt.eval_freq == 0 and epoch != 0:
        ## eval
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_loss = 0
            dice_val_tot = 0
            iter_val_num = 0
            for idx, (image_val, mask_val, affine_mask_val, save_path_nii_val, crop_offset_val, tp_name_val) in enumerate(val_loader):
                iter_val_num += 1
                # print(f"tp_name_val: {tp_name_val}")
                image_val, mask_val = image_val.to(device), mask_val.to(device)
                # with torch.amp.autocast(device_type='cuda'):
                output_val = model(image_val)
                save_pred_nii_simple(output_val, epoch, save_path_nii_val, affine=affine_mask_val, crop_offsets=crop_offset_val, mod=opt.mod)
                mask_val = mask_val.squeeze(1).long()
                loss_val = criterion(output_val, mask_val)
                # loss_val, mean, fore_label = criterion(output_val, mask_val)

                ## loss & accuracy
                val_loss += loss_val.item()
                diec_val_loc = cal_diceMetric_new(output_val, mask_val, opt.classes)
                dice_val_tot += diec_val_loc


            if val_loss < bst_loss_val:
                bst_loss_val = val_loss
                bst_mode_val_path = os.path.join(path_log, 'model_val_bst.path')
                torch.save(model.state_dict(), bst_mode_val_path)
            
            loss_val_list.append(val_loss)
            val_accuracy = get_avgAcc(dice_val_tot, iter_val_num)
            logging.info(f"[Eval] Epoch: {epoch+1}, training Loss: {val_loss}, training acc: {val_accuracy}")
            acc_val_list.append(val_accuracy)

    save_acc_loss_curves(acc_train_list, acc_val_list, loss_train_list, loss_val_list, path_log)


