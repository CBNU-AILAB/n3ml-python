import time
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from n3ml.model import Cao2015_Tailored


def train(train_loader, model, criterion, optimizer):
    model.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0.0

    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images

    return train_acc, train_loss


def validate(val_loader, model, criterion):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        loss = criterion(preds, labels)

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    val_acc = num_corrects.float() / total_images
    val_loss = total_loss / total_images

    return val_acc, val_loss


def app(opt):
    print(opt)

    # TODO: weight normalization 적용하는 방법 생각해야함
    # TODO: 최소 픽셀값과 최대 픽셀값이 0과 1이기 때문에 별도로
    # TODO: 정규화를 할 필요가 없다.
    # Cao2015 procedure
    # 1. Train tailored CNN and then save its weights
    # 2. Load its weights and then apply them to SNN
    # 3. Infer using SNN

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(24),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(24),
                torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers)

    model = Cao2015_Tailored().cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    best_acc = 0

    for epoch in range(opt.num_epochs):
        start = time.time()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end - start, epoch, train_acc,
                                                                                 train_loss))

        val_acc, val_loss = validate(val_loader, model, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()}
            torch.save(state, opt.pretrained)
            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, best_acc, val_loss))

        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--pretrained', default='pretrained/cao2015_tailored.pt')

    app(parser.parse_args())
