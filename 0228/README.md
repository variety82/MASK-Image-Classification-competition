python train_skf2.py --model_dir './model/27' --criterion 'label_smoothing' --optimizer 'AdamW' --valid_batch_size 3780 --model "Model" --name "EfficientNet" --epoch 10

python inference.py --model_dir './model/27/EfficientNet9' --model 'Model'
