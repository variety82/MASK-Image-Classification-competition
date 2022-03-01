모델 : efficientnet_b4

batch_size : 128, val_batch_size : 128, val_ratio : 0.05

optimizer : AdamW

lr_scheduler : cosine_annealing_warm_restart

1. age label 59변경
1. over sampling 진행 (30,40,60대를 한 번 씩 그대로 추가)
1. MaskSplitByprofileDataset 대신 MaskBaseDataset 사용 
1. efficientnet 내에서 drop_oup 0.6으로 수정
1. 테스트데이터셋에서 centercrop 추가



python train.py --epochs 30 --dataset MaskBaseDataset --augmentation CustomAug1 --batch_size 128 balid_batch_size 128 --optimizer AdamW --val_ratio 0.05 --criterion label_smoothing

python inference.py --batch_size 128 

(이미지 사이즈는 [224, 224] 입니당, Custom Aug1 내용은 아래와 같습니다)



class CustomAug1:
    def __init__(self, resize, mean, std):
        self.transform = A.Compose([
            A.CenterCrop(320, 256),
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.OneOf([A.MotionBlur(p=1),
                     A.GaussNoise(p=1),

            ], p=0.8),
            A.ColorJitter(p=0.3),
            A.Cutout(max_h_size=int(256 * 0.4), max_w_size=int(256 * 0.4), num_holes=1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def __call__(self, image):
        return self.transform(image=image)
