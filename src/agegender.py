class AgeGenderModel(nn.Module):
    """
    CNN model with 2 heads and SE-block
    with multitask model learns faster
    """
    def __init__(self, encoder, encoder_channels, 
                 age_classes, gender_classes, output_channels=512):
        super().__init__()
        
        # encoder features (resnet50 in my case)
        # output should be bs x c x h x w
        self.encoder = encoder
        
        # sqeeze-excite
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.downsample = nn.Conv2d(encoder_channels, output_channels, 1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.nonlin1 = nn.ReLU()
        
        self.excite = nn.Conv2d(output_channels, output_channels, 1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.nonlin2 = nn.ReLU()
        
        self.age_head = nn.Conv2d(output_channels, age_classes, 1)
        self.gender_head = nn.Conv2d(output_channels, gender_classes, 1)
    
    def forward(self, x):
        features = self.encoder(x)
        features = self.squeeze(features)
        features = self.downsample(features)
        features = self.nonlin1(self.bn1(features))
        
        weights_logits = self.excite(features)
        features = features * weights_logits.sigmoid()
        features = self.nonlin2(self.bn2(features))
        
        age_logits = self.age_head(features).view(features.size(0), -1)
        gender_logits = self.gender_head(features).view(features.size(0), -1)
        return age_logits, gender_logits

def accuracy(pred: torch.Tensor, gt: torch.Tensor):
    """
    accuracy metric
    
    expects pred shape bs x n_c, gt shape bs x 1
    """
    return (pred.max(1)[1] == gt).float().mean()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(model, loader, opt, 
          age_criterion, gender_criterion, metric_fn, 
          device="cuda", sched=None, epoch=1, print_every=50):
    model.train()
    t0 = time.time()
    for batch_idx, (image, (age_gt, gender_gt)) in enumerate(loader):
        data_time = time.time() - t0
        opt.zero_grad()
        t1 = time.time()
        image, age_gt, gender_gt = image.to(device), age_gt.to(device), gender_gt.to(device)
        age_logits, gender_logits = model(image)
        
        # BCE expects one-hot vector
        age_gt_onehot = torch.zeros(*age_logits.size(), device=age_logits.device)
        age_gt_onehot = age_gt_onehot.scatter_(1, age_gt.unsqueeze(-1).long(), 1)
        gender_gt = gender_gt.long()
        
        model_time = time.time() - t1
        loss_age = age_criterion(age_logits, age_gt_onehot)  # bce
        loss_gender = gender_criterion(gender_logits, gender_gt)  # softmax+ce
        loss = (loss_age + loss_gender) / 2
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        gender_acc = metric_fn(gender_logits, gender_gt)
        age_acc = metric_fn(age_logits, age_gt)
        
        if batch_idx % print_every == 0:
            print(f"train epoch {epoch}, {batch_idx} / {len(loader)}, loss age {loss_age.item():.3f} "
                  f"loss gender {loss_gender.item():.3f} loss: {loss.item():.3f}, "
                  f"gender acc: {gender_acc.item():.2%} age acc: {age_acc.item():.2%} "
                  f"data/model times {data_time*1000:.1f} ms, {model_time*1000:.1f} ms")
        t0 = time.time()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def validate(model, loader, metric_fn, device="cuda", epoch=1):
    model.eval()
    gender_acc_list = []
    age_acc_list = []
    for image, (age_gt, gender_gt) in loader:
        with torch.no_grad():
            image, age_gt, gender_gt = image.to(device), age_gt.to(device), gender_gt.to(device)
            age_logits, gender_logits = model(image)
            gender_acc_list.append(metric_fn(gender_logits, gender_gt).item())
            age_acc_list.append(metric_fn(age_logits, age_gt).item())
    gender_acc = np.mean(gender_acc_list)
    age_acc = np.mean(age_acc_list)
    print(f"val epoch {epoch}, gender acc {gender_acc:.2%}, age acc {age_acc:.2%}")
    return (gender_acc + age_acc) / 2


mobilenet_v3 = timm.create_model('tf_mobilenetv3_large_100', pretrained=True)
mobilenet_v3_encoder = nn.Sequential(*list(mobilenet_v3.children())[:-4]).cuda()
model = AgeGenderModel(mobilenet_v3_encoder, 960, age_classes=3, gender_classes=2).cuda()
num_epochs = 10
lr = 3e-4

optimizer = torch.optim.AdamW(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
age_criterion = nn.BCEWithLogitsLoss()
gender_criterion = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs+1):
    train(model, train_loader, optimizer, 
        age_criterion, gender_criterion, metric_fn=accuracy, 
        device="cuda", epoch=epoch, print_every=30)

val_acc = validate(model, valid_loader, metric_fn=accuracy, device="cuda", epoch=epoch)

scheduler.step(val_acc)
model = torch.nn.DataParallel(model).cuda()
torch.save(model.module.state_dict(), "mobilenetv3_age.pth")
