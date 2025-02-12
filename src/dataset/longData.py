import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from src.tools.utils import crop_to_multiple_of_16, save_nii_simple, normalize_img, get_nii, save_nii

class CustomDataset(Dataset):
    def __init__(self, dataFile_path, transform=None, path_log=None, isTrain=True, mod='long_seg', drop_rate=0.2):
        self.transform = transform
        self.dataFile_path = dataFile_path
        self.path_log = path_log
        self.isTrain = isTrain
        self.mod = mod
        self.drop_rate = drop_rate

        if self.mod == 'long_seg':
            self.subj_ids = sorted(os.listdir(self.dataFile_path))
        elif self.mod == 'single_seg':
            self.subj_ids = sorted(os.listdir(os.path.join(self.dataFile_path, 'images')))
        self.length = len(self.subj_ids)
        
        
    def __len__(self):

        return self.length
    
    def __getitem__(self, idx):

        # torch.manual_seed(idx)
        ## if idx >= subj nums
        # print(f"idx: {idx}")
        while idx >= len(self.subj_ids):
            idx = idx % self.length

        if self.mod == 'long_seg':
            '''the folder here should be like:
                |---subj1
                    |---tp1
                        |---CT1_image.nii.gz
                        |---CT1_mask.nii.gz
                    |---tp2
                        |---CT1_image.nii.gz
                        |---CT1_mask.nii.gz
                    |...
                |---subj2
                    |---tp1
                        |---CT1_image.nii.gz
                        |---CT1_mask.nii.gz
                    |---tp2
                        |---CT1_image.nii.gz
                        |---CT1_mask.nii.gz
                    |...
                |...
            '''

            subj = 'subject_' + str(idx).zfill(3)

            tp_path = os.path.join(self.dataFile_path, subj)
            tps = sorted(os.listdir(tp_path))
            # print(f"tps: {tps}")

            ## randomly choose 1 timepoint and its next
            tp_num = len(tps)
            tp_idx = torch.randint(0,tp_num-1, (1,)).item()
            # tp_next_idx = tp_idx + 1
            tp_next_idx = torch.randint(tp_idx+1,tp_num, (1,)).item()
            image1, _ = get_nii(os.path.join(tp_path, tps[tp_idx], 'CT1_image.nii.gz'))
            image2, _ = get_nii(os.path.join(tp_path, tps[tp_next_idx], 'CT1_image.nii.gz'))
            mask2, affine_mask2 = get_nii(os.path.join(tp_path, tps[tp_next_idx], 'CT1_mask.nii.gz'))
            tp_name = tps[tp_idx:(tp_idx+2)]
            image_choose = []
            image_choose.append(image1)
            image_choose.append(image2)
            image_choose = np.array(image_choose)
            mask_choose = np.expand_dims(mask2, axis=0)
        elif self.mod == 'single_seg':
            '''the folder here should be like:
                |---images
                    |---name1.nii.gz
                    |---name2.nii.gz
                    |...
                |---labels
                    |---name1.nii.gz
                    |---name2.nii.gz
                    |...
                |...
            '''
            image_dir_path = os.path.join(self.dataFile_path, 'images') # images dir path
            labels_dir_path = os.path.join(self.dataFile_path, 'labels') # labels dir path
            image_files = sorted(os.listdir(image_dir_path))
            label_files = sorted(os.listdir(labels_dir_path))
            # print(f"image_files: {image_files}")
            # print(f"label_files: {label_files}")

            ## randomly choose 1 pair of image and mask
            file_num = len(image_files)
            file_idx = torch.randint(0,file_num, (1,)).item()
            # print(image_files[file_idx].split('_')[1])
            # print(label_files[file_idx].split('_')[1].split('.')[0])
            assert image_files[file_idx].split('_')[1] == label_files[file_idx].split('_')[1].split('.')[0], "image and mask should have the same index, consistent"
            image_choose, _ = get_nii(os.path.join(image_dir_path, image_files[file_idx]))
            mask_choose, affine_mask2 = get_nii(os.path.join(labels_dir_path, label_files[file_idx]))
            image_choose = np.expand_dims(image_choose, axis=0)
            mask_choose = np.expand_dims(mask_choose, axis=0)
            tp_name = image_files[file_idx]

        else:
            raise ValueError("mod should be 'long_seg' or 'single_seg'")
        if self.transform:
            image_choose = self.transform(image_choose)

        
        
        image_choose, crop_offset_im = crop_to_multiple_of_16(image_choose, self.mod) ## [B, D, H, W]
        mask_choose, crop_offset_mask = crop_to_multiple_of_16(mask_choose, self.mod) ## [B=1, D, H, W]

        ## save image and mask for checking
        ## get path_log
        path_log = self.path_log
        sub_folder = 'train' if self.isTrain else 'val'
        save_path_nii = os.path.join(path_log, sub_folder, 'niiData')
        if self.mod == 'long_seg':
            save_path_nii = os.path.join(save_path_nii, tp_name[-1])
        else:
            save_path_nii = os.path.join(save_path_nii, tp_name.split('.')[0])  ## remove '.nii.gz' if exist
        # print(save_path_nii)
        os.makedirs(save_path_nii, exist_ok=True)
        # if not os.path.exists(save_path_nii):
        #     os.mkdir(save_path_nii)
        # os.makedirs(os.path.join('Data/Imaging_simple', sub_i, tp_name), exist_ok=True)

        ## save image_choose and mask_choose
        # print(f"image_choose shape: {image_choose.shape}")
        # print(f"mask_choose shape: {mask_choose.shape}")
        image_choose_save = torch.tensor(image_choose[-1], dtype=torch.float32).unsqueeze(0)
        mask_choose_save = torch.tensor(mask_choose[-1], dtype=torch.float32).unsqueeze(0)
        # print(f"image_choose_save shape: {image_choose_save.shape}")
        # print(f"mask_choose_save shape: {mask_choose_save.shape}")
        save_nii_simple(image_choose_save, save_path_nii, isMask=False, affine=affine_mask2, crop_offsets=crop_offset_im)
        save_nii_simple(mask_choose_save, save_path_nii, isMask=True, affine=affine_mask2, crop_offsets=crop_offset_mask)

        ## crop & norm
        # print(f"image_choose shape: {image_choose.shape}")
        # print(f"mask_choose shape: {mask_choose.shape}")
        image_choose = normalize_img(image_choose)

        # assert image_choose.shape == mask_choose.shape, "image and mask should have the same shape"

        image_choose = torch.tensor(image_choose, dtype=torch.float32)  # 转换为 torch.float32
        mask_choose = torch.tensor(mask_choose, dtype=torch.float32)  # 转换为 torch.float32

        # image_choose = torch.tensor(image_choose)  # 转换为 torch.float32
        # mask_choose = torch.tensor(mask_choose)  # 转换为 torch.float32
        if self.mod == 'long_seg':
            image_choose = image_choose.unsqueeze(1)
            mask_choose = mask_choose.unsqueeze(1) ## [B=1, C=1, D, H, W]
        # print(f"image_choose shape: {image_choose.shape}")
        # print(f"mask_choose shape: {mask_choose.shape}")

        ## Here we only choose the last timepoint as the mask
        mask_choose = mask_choose[-1].unsqueeze(0) ## [B=1, D, H, W]
        

        return image_choose, mask_choose, affine_mask2, save_path_nii, crop_offset_mask, tp_name



# 自定义 collate_fn 来跳过 None 值
def dropout_collate_fn(batch):
    # 跳过 None 值
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)