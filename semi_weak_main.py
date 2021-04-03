'''Train CIFAR10 with PyTorch.'''
import argparse
import os
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import _compute_bag_metrics, search_count_all, _compute_instance_metrics, _search_batch, get_count_from_instance_prediction, _get_entropy, evaluate_instance_prediction
from dataset import CIFAR10_BAG
from models import *
from tqdm import tqdm
import sys
CUDA = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--min-lr', default=1e-5, type=float, help="minimal learning rate")
parser.add_argument('--epoch', '-e', default=100, type=int)
parser.add_argument('--batch-size', '-bs', default=128, type=int)
parser.add_argument('--meta-path', required=True, type=str)
parser.add_argument('--evaluate', "-eval", action="store_true")
parser.add_argument('--trial', "-t", default=0, type=int)
parser.add_argument("--comment", "-m", required=True, type=str)
parser.add_argument("--entropy-loss-weight", "-elw", default=1.0, type=float)
parser.add_argument("--binary-loss-weight", "-blw", default=1.0, type=float)   # not very usefuls
parser.add_argument("--loss-type", default="poisson", type=str)                    # useful
parser.add_argument("--entropy-weighted", "-ew", default=0, type=int, choices=[0, 1])
parser.add_argument("--count-loss-weight", "-clw", default=1.0, type=float)
parser.add_argument("--regularize", default=0., type=float)  # not very useful
parser.add_argument("--bag-size", required=True, type=int)
parser.add_argument("--base_classifier","-bclass",default="Resnet18",type=str)
parser.add_argument("--num_workers",default=2,type=int)

class Logger(object):
    def __init__(self,run_number):
        self.run_number = run_number
        self.terminal = sys.stdout
        self.log = open("./logfiles/logfile_" + str(run_number) + "_gpu.log", "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    def flush(self):
        pass    

args = parser.parse_args()
print("==> args: {}".format(vars(args)))
bag_size = args.bag_size
best_ins_prec = 0
n_class = 10
regularize = args.regularize
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
config_name = os.path.basename(args.meta_path) + f"ew={args.entropy_weighted}.loss={args.loss_type}.blw={args.binary_loss_weight}.elw={args.entropy_loss_weight}.clw={args.count_loss_weight}.reg={args.regularize}"
model_name = "trial={}.ckpt.pth".format(args.trial)
if args.comment == "":
    writer = None
else:
    writer = SummaryWriter ("runs/{}".format(args.comment))

if not os.path.exists('./logfiles'):
        os.makedirs('./logfiles')

c1 = Logger(args.comment + '_and_TRIAL_' + str(args.trial))
sys.stdout = c1
print('==> Run experiement: [{}]'.format(args.comment))
checkpoint_folder = os.path.join("./checkpoint", config_name)

if not os.path.exists("./checkpoint"):
    os.mkdir("./checkpoint")
if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)

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

trainset = CIFAR10_BAG(root='./data', train=True,
                       download=True,
                       transform=transform_train, meta_path=args.meta_path, trial=args.trial)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=CUDA)

testset = CIFAR10_BAG(root='./data', train=False,
                      download=True, transform=transform_test,
                      meta_path=args.meta_path, trial=args.trial)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=CUDA)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.base_classifier=="Resnet18":
        net = MILResNet18(classifier_type=args.base_classifier)
elif args.base_classifier=="Resnet34":
        net = MILResNet18(classifier_type=args.base_classifier)
elif args.base_classifier=="Resnet50":
        net = MILResNet18(classifier_type=args.base_classifier)
elif args.base_classifier=="Mobilenet_v2":
        net = MILResNet18(classifier_type=args.base_classifier)

if CUDA:
    cnt = torch.cuda.device_count()
    print ("==> Use %d GPUs" % cnt)
    net.cuda ()
    if cnt > 1:
        net = torch.nn.DataParallel (net)

    cudnn.benchmark = True

if args.resume or args.evaluate:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_folder), 'Error: no checkpoint directory found!'

    checkpoint = torch.load(os.path.join(checkpoint_folder, model_name))
    net.load_state_dict(checkpoint['net'])
    best_ins_prec = checkpoint['ins_prec']
    start_epoch = checkpoint['epoch']

optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global bag_size
    net.train()
    train_loss = 0
    train_entropy_loss = 0
    train_binary_loss = 0
    total = 0
    y_scores = []
    y_trues = []
    bag_scores = []
    bag_trues = []
    start = time.time()
    for batch_idx, (inputs, countings, instance_targets) in tqdm(enumerate(trainloader)):
        # instance_targets is not 0/1 but class label
        if CUDA:
            inputs = inputs.cuda()
            countings = countings.cuda()  # (B, n_class)
            instance_targets = instance_targets.cuda() # (B, bag)
        with torch.autograd.set_detect_anomaly(False):
            optimizer.zero_grad()
            count_pred, activation_map, instance_embedding = net(inputs)  # (b, n_class), (bs, bag, n_class)
            normalized_activation_map = activation_map.softmax(dim=2)
            if args.entropy_weighted == 1:
                weight = _get_entropy(countings)  # [0, log2(10)] larger means more diversity. Lower means sparser.
                weight = math.log2(10) - weight + 1
            elif args.entropy_weighted == 0:
                weight=None
            else:
                raise NotImplementedError
            count_loss, predicted, reg_loss = MILResNet18.get_count_loss (count_pred, countings,
                                                                          loss_type=args.loss_type,
                                                                          weight=weight,
                                                                          bag_size=bag_size)  # predicted: (bs, n_class)
            entropy_loss = MILResNet18.get_entropy_loss(normalized_activation_map, weight=weight)
            get_binary_predict = MILResNet18.get_binary_predict(instance_embedding, activation_map)
            binary_loss = MILResNet18.get_bce_loss(get_binary_predict, countings, weight=weight)
            loss = args.count_loss_weight * count_loss \
                   + args.entropy_loss_weight * entropy_loss \
                   + args.binary_loss_weight * binary_loss \
                   + args.regularize * reg_loss
            
            loss.backward()
            optimizer.step()
            train_loss += count_loss.item()
            train_entropy_loss += entropy_loss.item()
            train_binary_loss += binary_loss.item()
            total += inputs.size(0)
            y_scores.append(normalized_activation_map.detach())
            y_trues.append(instance_targets.detach())
            bag_scores.append(predicted.detach())
            bag_trues.append(countings.detach())

    end = time.time()
    y_scores = torch.cat(y_scores, dim=0) # (bs, bag, nclass)
    y_trues = torch.cat(y_trues, dim=0)   # (bs, bag)
    bag_scores = torch.cat(bag_scores, dim=0)
    bag_trues = torch.cat(bag_trues, dim=0)
    ins_prec = _compute_instance_metrics(y_scores, y_trues)
    bag_prec, reg_loss = _compute_bag_metrics(bag_scores, bag_trues)

    print("loss: {}, binary loss: {} entropy: {}".format(train_loss, train_binary_loss, train_entropy_loss))
    print('[TRAIN] Bag Prec: %.3f%% -- %d secs / epoch'% (100.* bag_prec, end - start))
    print('[TRAIN] Ins Prec: %.3f%%'% (100. * ins_prec))
    print('[TRAIN] Regression Loss: %.3f'% (reg_loss))

    if writer is not None:
        writer.add_scalar("bag prec/train", bag_prec, epoch)
        writer.add_scalar("inc prec/train", ins_prec, epoch)
        writer.add_scalar("Regression Loss/train", reg_loss, epoch)

def test(epoch):
    global best_ins_prec, bag_size, n_class
    net.eval()
    total = 0
    y_scores = []
    y_trues = []
    bag_scores = []
    bag_expected_scores = []
    bag_trues = []
    with torch.no_grad():
        for batch_idx, (inputs, countings, instance_targets) in enumerate(testloader):
            if CUDA :
                inputs = inputs.cuda ()
                countings = countings.cuda ()
                instance_targets = instance_targets.cuda ()

            count_pred, activation_map, instance_embedding = net(inputs)  # (b, n_class), (bs, bag, n_class)
            normalized_activation_map = activation_map.softmax (dim=2)
            _, predicted, _ = MILResNet18.get_count_loss(count_pred, countings, loss_type=args.loss_type)  # predicted: (bs, n_class)
            get_binary_predict = MILResNet18.get_binary_predict(instance_embedding, activation_map)
            total += inputs.size(0)
            y_scores.append(normalized_activation_map.detach())
            bag_expected_scores.append(count_pred.detach())
            y_trues.append(instance_targets.detach())
            bag_scores.append(predicted.detach())
            bag_trues.append(countings.detach())

    y_scores = torch.cat(y_scores, dim=0) # (bs, bag, nclass)
    y_trues = torch.cat(y_trues, dim=0)   # (bs, bag)
    bag_scores = torch.cat(bag_scores, dim=0)
    bag_trues = torch.cat(bag_trues, dim=0)
    bag_expected_scores = torch.cat(bag_expected_scores, dim=0)
    bag_scores_by_classification = get_count_from_instance_prediction (y_scores)
    ins_prec = _compute_instance_metrics(y_scores, y_trues)
    bag_prec, reg_loss = _compute_bag_metrics(bag_scores, bag_trues)
    bag_prec_by_classification, reg_loss_by_classification \
        = _compute_bag_metrics(bag_scores_by_classification, bag_trues)
    if args.evaluate :
        predicted = search_count_all (bag_expected_scores, bag_size=bag_size,
                                      n_class=10)
        instace_prediction = _search_batch(y_scores, predicted)  # (batch, bag)
        new_ins_prec = _compute_instance_metrics(instace_prediction, y_trues)
        print('[DECODE] Ins Prec: %.3f%%'% (100. * new_ins_prec))
        evaluate_instance_prediction(instace_prediction, y_trues)

    print("true")
    print(bag_trues[:5])
    print ("pred by regression")
    print(bag_scores[:5])
    print("pred stats")
    for col in range(n_class):
        lis = []
        for count in range(0, bag_size+1):
            lis.append((bag_scores[:, col] == count).float().mean().item())
        lis = [round(i, 4) for i in lis]
        print(f"[{col}] {lis}")

    # print ("pred by classification")
    # print(bag_scores_by_classification[:5])
    # for col in range(10):
    #     lis = []
    #     for count in range(0, 10):
    #         lis.append((bag_scores_by_classification[:, col] == count).float().mean().item())
    #     lis = [round (i, 4) for i in lis]
    #     print(f"[{col}] {lis}")

    print('[TEST] Bag Prec: %.3f%% Regression Loss %.3f'% (100.* bag_prec, reg_loss))
    print('[TEST] Bag Prec(cls): %.3f%% Regression Loss(cls) %.3f'% (100.* bag_prec_by_classification, reg_loss_by_classification))
    print('[TEST] Ins Prec: %.3f%%'% (100. * ins_prec))
    if writer is not None:
        writer.add_scalar ("bag prec/test", bag_prec, epoch)
        writer.add_scalar ("inc prec/test", ins_prec, epoch)
        writer.add_scalar ("Regression Loss/test", reg_loss, epoch)

    # Save checkpoint.
    if ins_prec > best_ins_prec:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'ins_prec': ins_prec,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoint_folder, model_name))
        best_ins_prec = ins_prec

if not args.evaluate:
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train(epoch)
        scheduler.step()
        test(epoch)
else:
    test(0)
