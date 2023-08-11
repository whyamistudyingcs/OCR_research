import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Data import MyDataset
from EfficientNetV2.model import efficientnetv2_s
from Utils import has_log_file, find_max_log


def train(args):
    print("===Train EffNetV2===")
    # self-defined data augmentation
    transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # mean and std must be fixed, if pretrained weight is applied
         transforms.ColorJitter()])

    train_set = MyDataset(args.data_root + 'train.txt', num_class=args.num_classes, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda:0')

    # load model
    model = efficientnetv2_s(num_classes=args.num_classes)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lr-scheduler: reduce learning rate by half, if the model does not improve in two consecutive epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    print("load model...")

    if has_log_file(args.log_root):
        max_log = find_max_log(args.log_root)
        print("continue training with " + max_log + "...")
        checkpoint = torch.load(max_log)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch'] + 1
    else:
        print("train for the first time...")
        loss = 0.0
        epoch = 0

    while epoch < args.epoch:
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('epoch %5d: batch: %5d, loss: %8f, lr: %f' % (
                    epoch + 1, i + 1, running_loss / 200, optimizer.state_dict()['param_groups'][0]['lr']))
                running_loss = 0.0

        scheduler.step(loss)
        print('Save checkpoint...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   args.log_root + 'log' + str(epoch) + '.pth')
        print('Saved')
        epoch += 1

    print('Finish training')