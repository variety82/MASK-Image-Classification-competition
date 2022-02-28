import argparse
import os
from importlib import import_module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats
from dataset import TestDataset, MaskBaseDataset
from torch.nn import functional as F


def load_model(saved_model, num_classes, device, i):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)
    saved_model = saved_model + f'/{i}'
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    pred_total = []
    for i in range(5):
        model = load_model(model_dir, num_classes, device, i).to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'images')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        with torch.no_grad():
            fold_preds = []
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                pred = F.softmax(pred, dim = 1)
                fold_preds.append(pred.data.cpu().numpy())
                
            pred_total = np.vstack(fold_preds)
            
    


    info['ans'] = pred_total[0].squeeze(axis = 0).astype(np.int)
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(128, 96), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
