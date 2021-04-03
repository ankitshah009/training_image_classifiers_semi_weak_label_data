'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import get_full_supervised
from models import *
from utils import progress_bar, _compute_instance_metrics, get_count_from_instance_prediction, _compute_bag_metrics
CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--min-lr', default=1e-4, type=float, help="minimal learning rate")
parser.add_argument('--epoch', '-e', default=200, type=int)
parser.add_argument('--batch-size', '-bs', default=128, type=int)
parser.add_argument('--bag-size','-bags', default=8,type=int)
parser.add_argument('--meta-path', required=True, type=str)
parser.add_argument("--evaluate", "-eval", default=0, type=int, help="0: no evaluate\n1: fully-supervised evaluation\n2: bag-level evaluation")
parser.add_argument("--trial", default=0, type=int)
parser.add_argument("--meta_path", default="s", type=str)
parser.add_argument("--comment", "-m", required=True, type=str)
parser.add_argument("--base_classifier",default='Resnet18',type=str, choices=['Resnet18', 'Resnet34','Resnet50','Mobilenet_v2'])
parser.add_argument("--model-name",default='',type=str)

args = parser.parse_args()

if args.evaluate > 1:
    assert args.meta_path != "", "cannot be empty"

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
writer = SummaryWriter ("runs/fully-supervised-{}".format(args.comment))
# Data
print('==> Preparing data..')
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

# trainset = get_full_supervised(
#     root='./data', train=True, download=True, transform=transform_train, meta_path=args.meta_path)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=CUDA)

# testset = get_full_supervised(root='./data', train=False, download=True, transform=transform_test, meta_path=args.meta_path)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=CUDA)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
base_classifier = args.base_classifier
if base_classifier=='Resnet18':
        net = ResNet18()
elif base_classifier=='Resnet50':
	net = ResNet50() 
elif base_classifier=='Resnet34':
	net = ResNet34() 
elif base_classifier=='Mobilenet_v2':
	net = MobileNetV2() 

else:
    raise NotImplementedError


if CUDA:
    cnt = torch.cuda.device_count()
    print ("==> Use %d GPUs" % cnt)
    net = torch.nn.DataParallel (net)
    net.cuda ()
    cudnn.benchmark = True

checkpoint_folder = "./checkpoint"
if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)

checkpoint_name = 'fully-supervised-ckpt_' + args.base_classifier + '_'  + str(args.lr) + '_' + str(args.batch_size) + '_' + str(args.meta_path).split('/')[-1] +  '.pth'
print(checkpoint_name)
if args.resume or args.evaluate > 0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_folder), 'Error: no checkpoint directory found!'
    model_name = checkpoint_name
    if args.model_name != "":
        model_name = args.model_name
    checkpoint = torch.load(os.path.join(checkpoint_folder, model_name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar ("acc/train", correct/total, epoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if CUDA :
                inputs = inputs.cuda ()
                targets = targets.cuda ()
            outputs, attention = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar ("acc/test", correct/total, epoch)
    print('Acc: %.3f%% (%d/%d)'% (100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoint_folder, checkpoint_name))
        best_acc = acc

def evaluate_bag(root, transform, meta_path):
    net.eval()
    testset = CIFAR10_BAG (root=root, train=False,
                           download=True, transform=transform,
                           meta_path=meta_path, trial=args.trial)
    testloader = torch.utils.data.DataLoader (testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=CUDA)
    y_scores = []
    y_trues = []
    for batch_idx, (inputs, countings, instance_targets) in enumerate (testloader) :
        if CUDA :
            inputs = inputs.cuda () # (B, b, c, h, w)
            countings = countings.cuda ()
            instance_targets = instance_targets.cuda ()
        batch_size, bag_size, c, h, w = inputs.size()
        inputs = inputs.view(bag_size * batch_size, c, h, w)
        outputs, _ = net(inputs)
        normalized_activation_map = outputs.view(batch_size, bag_size, -1)
        y_trues.append(instance_targets.detach())
        y_scores.append (normalized_activation_map.detach ())

    y_scores = torch.cat(y_scores, dim=0) # (bs, bag, nclass)
    y_trues = torch.cat(y_trues, dim=0)   # (bs, bag)
    bag_scores_by_classification = get_count_from_instance_prediction (y_scores)
    ins_prec = _compute_instance_metrics(y_scores, y_trues)
    # bag_prec_by_classification, reg_loss_by_classification \
    #     = _compute_bag_metrics (bag_scores_by_classification, bag_trues)
    # print('[TEST] Bag Prec(cls): %.3f%% Regression Loss(cls) %.3f'% (100.* bag_prec_by_classification, reg_loss_by_classification))
    print('[TEST] Ins Prec: %.3f%%'% (100. * ins_prec))

if args.evaluate == 0:
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train(epoch)
        if optimizer.param_groups[0]['lr'] > args.min_lr:
            scheduler.step()
        test(epoch)
elif args.evaluate == 1:
    test (0)

elif args.evaluate == 2:
    from dataset import CIFAR10_BAG
    stat = evaluate_bag("./data", transform_test, args.meta_path)

else:
    raise NotImplementedError
