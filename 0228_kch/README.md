모델 : efficientnet_b4

batch_size : 128, val_batch_size : 128, val_ratio : 0.05

optimizer : AdamW

lr_scheduler : cosine_annealing_warm_restart

1. age label 59변경
1. over sampling 진행 (30,40,60대를 한 번 씩 그대로 추가)
1. MaskSplitByprofileDataset 대신 MaskBaseDataset 사용 
1. efficientnet 내에서 drop_oup 0.6으로 수정
1. 테스트데이터셋에서 centercrop 추가
