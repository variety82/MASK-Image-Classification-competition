----train.py 
python train_skf2.py --model_dir './model/27' --criterion 'label_smoothing' --optimizer 'AdamW' --valid_batch_size 3780 --model "Model" --name "EfficientNet" --epoch 10


----inference.py
python inference.py --model_dir './model/27/EfficientNet9' --model 'Model'


1. age label 변경 (59세를 60세 label로)
2. transform의 대대적인 변화 (get_transforms함수내에서 적용됨)
3. soft voting 추가
4. optimizer - AdamW, scheduler - cosineAnneling(?) 
5. model - tf_efficientnet_b0
6. batchsize 64 - 근데 사실 별 의미 없어보임 근데 train 에서는 256보다 64가 정확도가 더 높게나와서 64로 제출
7. valid/train set에 transform 따로 적용된거 완벽하게 확인 !! 
