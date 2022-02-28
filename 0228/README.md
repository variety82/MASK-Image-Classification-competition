--model "Model" --optimizer 'Adam' --name "EfficientNet-Album-2" --epochs 10 --criterion "label_smoothing" --valid_batch_size 3780 --val_ratio  0.2 --optimizer 'Adaw' --dataset 'MaskSplitByProfileDataset'

--model_dir './model/27' --criterion 'label_smoothing' --optimizer 'AdamW' --valid_batch_size 3780 --model "Model" --name "EfficientNet" --epoch 10
