import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from load_cifar10 import NoisyCifar10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.ResNet import ResNet18, ResNet34

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with Noisy Labels...')
parser.add_argument('--outf', default='./checkpoints', help='folder to output images and model checkpoints')
parser.add_argument('--train_mode', default='noisy', help='Training mode: clean or noisy')
parser.add_argument('--noise_type', default='uniform', help='noise type: flip_random, flip_to_one, uniform')
parser.add_argument('--noise_ratio', default=30, help='noise rate')
parser.add_argument('--model', default='ResNet18', help='model')
parser.add_argument('--optim', default='Sgd', help='optimizer')
parser.add_argument('--lr', default=0.01, help='learning rate')
parser.add_argument('--Epoch', default=200, help='Whole Training Epoch')
parser.add_argument('--batch', default=200, help='Batch Size')
args = parser.parse_args()

# =================================== 准备数据集并预处理 ================================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = NoisyCifar10('train', _transform=transform_train, _noise_type=args.noise_type, _noisy_ratio=args.noise_ratio)
test_data = NoisyCifar10('test', _transform=transform_test)

trainloader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=100, shuffle=False)
# ======================================================================================================

# ========================================= 模型定义 ====================================================
if args.model == 'ResNet18':
    net = ResNet18().to(device)
elif args.model == 'ResNet34':
    net = ResNet34().to(device)
# ======================================================================================================

# ======================================= 定义损失函数和优化方式 ============================================
criterion = nn.CrossEntropyLoss()
if args.optim == 'Sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
# =======================================================================================================


if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.mkdir(args.outf)

    min_loss = 0.1
    best_acc = 40
    print("Start Training %s under noise type=%s, noise ratio=%s, using %s!" %
          (args.model, args.noise_type, args.noise_ratio, args.optim)
          )
    with open("./logs/%s_%s_%s_%s_acc.txt" % (args.model, args.noise_type, args.noise_ratio, args.optim), "w") as f1:
        with open("./logs/%s_%s_%s_%s_log.txt" % (args.model, args.noise_type, args.noise_ratio, args.optim), "w")as f2:
            for epoch in range(0, args.Epoch):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                loss_mean = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    if args.train_mode == 'noisy':
                        inputs, _, labels = data
                    else:
                        inputs, labels, _ = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    loss_mean = sum_loss / (i + 1)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '%
                          (epoch + 1, (i + 1 + epoch * length), loss_mean, 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '%
                             (epoch + 1, (i + 1 + epoch * length), loss_mean, 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    print('Saving model......')
                    if epoch > 150:
                        torch.save(net.state_dict(), '%s/%s_%s_%s_%s_%03d.pth' %
                                   (args.outf, args.model, args.noise_type, args.noise_ratio, args.optim, epoch + 1))
                    f1.write("EPOCH=%03d, Accuracy= %.3f%%" % (epoch + 1, acc))
                    f1.write('\n')
                    f1.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if args.train_mode == 'clean':
                        if acc > best_acc:
                            f3 = open("./logs/%s_%s_%s_%s_best_acc.txt" %
                                      (args.model, args.noise_type, args.noise_rate, args.optim), "w")
                            f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                            f3.close()
                            best_acc = acc
                    elif args.train_mode == 'noisy':
                        if loss_mean < min_loss:
                            f3 = open("./logs/%s_%s_%s_%s_min_loss.txt" %
                                      (args.model, args.noise_type, args.noise_rate, args.optim), "w")
                            f3.write("EPOCH=%d,min_loss = %.3f%%" % (epoch + 1, loss_mean))
                            f3.close()
                            min_loss = loss_mean
                if args.optim == 'Sgd':
                    scheduler.step()
            print("Training Finished, TotalEPOCH=%d" % args.Epoch)

