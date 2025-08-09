import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class CocoDataset(Dataset):
    def __init__(self,coco,image_dir,transforms=None):
        super().__init__()
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = self.coco.getImgIds()
        self.transform = transforms

    def __getitem__(self, index):
        '''
        ladImgs始终返回一个列表（因为可以接受多张图片的加载）
        '''
        img_info = self.coco.loadImgs(self.image_ids[index])[0]
        img_path = f'{self.image_dir}/{img_info["file_name"]}'

        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取标注id
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        # 获取标注信息，是一个数组，因为一张图不止一个目标
        anns = self.coco.loadAnns(ann_ids)

        # 生成mask（掩码），创造全零掩码
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        '''
        转换为torch格式
        .premute()改变维度顺序
        /255.0 将像素值归一化到0-1
        '''
        img = torch.from_numpy(img).float().permute(2, 0, 1)/255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return img,mask

    def __len__(self):
        return len(self.image_ids)
