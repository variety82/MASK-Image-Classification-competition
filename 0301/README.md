변경사항

1. wandb 추가
2. parser : n_splits 추가해서 몇개의 K fold를 할지 정해줄 수 있습니다.
3. getDataLoader에서 valid_batch_size를 인자로 받게 변경



명령어

python train_skf2.py --model_dir './model/1' --criterion 'label_smoothing' --optimizer 'AdamW' --valid_batch_size 1890 --batch_size 64 --model "Model" --name "sjh-EfficientNetb0-test" --epoch 10 --n_splits 5  



--name : "[자기이름]-[모델명]"

--n_splits : 5

--valid_batch_size는 입력해줄 필요가 없음 왜냐하면

valid_batch_size = 18900 / n_splits 로 계산이 돼서

args.valid_batch_size는 안쓰임