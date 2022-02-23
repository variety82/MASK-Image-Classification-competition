import easydict
import albumentations as A
from albumentations.pytorch import ToTensorV2

args = {}

args["train_aug"] = A.Compose([A.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0, p=0.5), 
                               A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                               A.Blur(p=0.1), 
                               A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                               A.HorizontalFlip(p=0.5), 
                               A.ToGray(p=0.1), 
                               A.RandomBrightnessContrast(p=0.5),
                               A.Normalize(mean=(0.560, 0.5241, 0.501), std=(0.233, 0.243, 0.245)), 
                               ToTensorV2()]) 
                               
args["test_aug"] = A.Compose([A.Normalize(mean=(0.560, 0.5241, 0.501), std=(0.233, 0.243, 0.245)),
                             ToTensorV2()])
                             
args = easydict.EasyDict(args)

#111