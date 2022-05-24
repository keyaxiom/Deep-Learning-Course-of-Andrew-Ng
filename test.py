import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from load_cifar10 import NoisyCifar10
from models.ResNet import ResNet18
from torch.utils.data.sampler import SubsetRandomSampler
from perturbation_stage import s_test
from torch.autograd import grad

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with Noisy Labels...')
parser.add_argument('--outf', default='./checkpoints', help='folder to output denoise labels and model checkpoints')
parser.add_argument('--noise_type', default='uniform', help='noise type: flip_random, flip_to_one, uniform')
parser.add_argument('--noise_ratio', default=30, help='noise rate')
parser.add_argument('--model', default='ResNet18', help='model')
parser.add_argument('--lr', default=0.01, help='learning rate')
parser.add_argument('--batch', default=200, help='Batch Size')
args = parser.parse_args()

# =================================== 准备数据集并预处理 ================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = NoisyCifar10('train', _transform=transform, _noise_type=args.noise_type, _noisy_ratio=args.noise_ratio)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True)

test_data = NoisyCifar10('test', _transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=200, shuffle=False)

indices = list(range(len(test_data)))
np.random.seed(2)
np.random.shuffle(indices)
train_indices, val_indices = indices[2000:], indices[:2000]

valid_sampler = SubsetRandomSampler(val_indices)
val_loader = torch.utils.data.DataLoader(test_data, batch_size=200, sampler=valid_sampler)
# ======================================================================================================

# ========================================= 模型定义 ====================================================
net = ResNet18().to(device)
net.load_state_dict(torch.load("./checkpoints/ResNet18_uniform_30_Sgd_200.pth"))
criterion = nn.CrossEntropyLoss()
# ======================================================================================================

h_estimate = s_test(val_loader, net, train_loader, gpu=1, damp=0.01, scale=2500.0, recursion_depth=1000)
torch.save(h_estimate, './h_estimate.pkl')
print(h_estimate)

denoise_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
influence = []
denoise_label = []
count = 0
for i, data in enumerate(denoise_loader, 0):
    # 准备数据
    inputs, true_labels, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)

    params = [p for p in net.parameters() if p.requires_grad]
    diff = torch.unbind(-F.log_softmax(outputs, dim=1) - criterion(outputs, labels), dim=1)
    single = []
    for k in range(10):
        grad_diff = grad(diff[k], params, retain_graph=True)
        elem_wise_products = 0.
        for diff_elem, h_elem in zip(grad_diff, h_estimate):
            elem_wise_products += torch.sum(diff_elem * h_elem).item()
        single.append(elem_wise_products)

    single_np = np.array(single)
    denoise_label.append(single_np.argmax())
    influence.append(single)
    print("[id: %d] single: %s | noisy label: %d | true_labels: %d | denoise_label: %d|"
          % (i+1, str(single), labels.item(), true_labels.item(), single_np.argmax()))
    if single_np.argmax() == true_labels.item():
        count += 1
    if i > 5000:
        break
print("denoised labels acc:", count/5000)
np.save('./influence.npy', np.array(influence))
np.save('./denoised_labels.npy', np.array(denoise_label))

























