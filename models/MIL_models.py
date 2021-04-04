from models import *
import torch.nn as nn
import torch.nn.functional as F
import torch
CUDA = torch.cuda.is_available()

def mae_loss(y_p, y_t):
    return (y_p - y_t).abs()

def mse_loss(y_p, y_t):
    return ((y_p - y_t) ** 2) / 2

def KL(y_p, y_t):
    y_p = y_p.softmax(dim=0)
    y_t = y_t.float().softmax(dim=0)
    return (y_p * (torch.log2(y_p / y_t)))

BETA = 1.0
def soft_mae(y_p, y_t, weight=None):
    return weight * nn.SmoothL1Loss(beta=BETA, reduction="none")(y_p, y_t)

def poisson_loss(y_p, y_t, weight=None):
    return nn.PoissonNLLLoss(log_input=False, full=False, reduction="none")(y_p, y_t)

    
    
    
class MILResNet18(nn.Module):
    "embedding-based: represents each instance as an embedding and then aggregate,"

    def __init__(self, classifier_type, n_class=10):
        super(MILResNet18, self).__init__()

        self.n_class = n_class
        if classifier_type == "Resnet18":
                self.L = 128 # hyperparam - no specific reason for this
                self.D = 50  # hyperparam - no specific reason for this
                self.cnn_extractor = ResNet18()
                self.linear_extractor = nn.Sequential(nn.Linear(512, self.L), nn.Tanh())
        elif classifier_type == "Resnet34":
                self.L = 128 # hyperparam - no specific reason for this
                self.D = 50  # hyperparam - no specific reason for this
                self.cnn_extractor = ResNet34()
                self.linear_extractor = nn.Sequential(nn.Linear(512, self.L), nn.Tanh())
        elif classifier_type == "Resnet50":
                self.L = 256 # hyperparam - no specific reason for this
                self.D = 100  # hyperparam - no specific reason for this
                self.cnn_extractor = ResNet50()
                self.linear_extractor = nn.Sequential(nn.Linear(512*4, self.L), nn.Tanh())
        elif classifier_type == "Mobilenet_v2":
                self.L = 256 # hyperparam - no specific reason for this
                self.D = 100  # hyperparam - no specific reason for this
                self.cnn_extractor = MobileNetV2()
                self.linear_extractor = nn.Sequential(nn.Linear(1280, self.L), nn.Tanh())
        # self.linear_extractor = nn.Sequential(nn.Identity())
        # self.linear_extractor = nn.Sequential(nn.Linear(512, self.L), nn.ReLU())
        self.attention = nn.Sequential (
            nn.Linear (self.L, self.D),
            nn.Tanh (),
            nn.Linear (self.D, self.n_class)
        )
        self.attention = nn.Sequential(nn.Linear(self.L, 50), nn.ReLU(), nn.Linear(50, n_class))
        self.fc = nn.Sequential (nn.Linear(self.L, n_class))

    @staticmethod
    def get_binary_predict(instance_embedding, activation_map):
        # (B, bag, self.L), (B, bag, n_class)
        # A = self.attention(self.instance_embedding)         # (B, bag, n_class)
        # binary_predict = (A * activation_map).sum(dim=1)    # (B, n_class)
        binary_predict, _ = activation_map.max(dim=1)    # (B, n_class)
        binary_predict = binary_predict.sigmoid()
        return binary_predict

    def forward(self, x):
        # x: (batch, bag_size, channel, 32, 32)
        batch, bag, c, h, w = x.size()
        x = x.view(batch * bag, c, h, w)
        _, H = self.cnn_extractor(x) # (B x Bag, 512)
        H = self.linear_extractor(H)  # (B x bag, self.L)
        instance_embedding = H.view(batch, bag, self.L) # (B, bag, self.L)
        activation_map = self.fc(instance_embedding) # (B, bag, n_class)
        normalized_activation_map = activation_map.softmax(dim=2) # (B, bag, n_class)
        count_pred = normalized_activation_map.sum(dim=1) # (B, bag, n_class)
        self.instance_embedding = instance_embedding
        return count_pred, activation_map, instance_embedding

    @staticmethod
    def instance_predict(normalized_activation_map):
        # (B, bag, n_class)
        bs, bag, n_class = normalized_activation_map.size()
        _, pred = normalized_activation_map.max(dim=2)
        count_pred = torch.zeros()
        return count_pred

    @staticmethod
    def get_count_loss(y_pred, y_true, loss_type="l2", weight=None, bag_size=8):
        """
        y_pred: (batch_size, n_class)
        y_true: (batch_size, n_class)
        """
        reg_loss = y_pred.abs().sum(dim=1).mean()
        # reg_loss = -(reg * (reg+1e-12).log2()).sum(dim=1).mean()
        pos_loss, neg_loss, num = 0, 0, 0
        # inv_gt_count = (y_true > 0).float () / (y_true + 1e-10)
        if loss_type == "l1":
            loss = mae_loss
        elif loss_type == "l2":
            loss = mse_loss
        elif loss_type == "smoothl1":
            loss = soft_mae
        elif loss_type == "poisson":
            loss = poisson_loss
        elif loss_type == "KL":
            loss = KL
        else:
            raise NotImplementedError
        for i in range (y_pred.size (0)) :
            # categories present in video
            pred_count = y_pred[i, :]
            val = loss(pred_count, y_true[i])
            if weight is not None:
                pos_loss += val.sum () * weight[i]# relative L1
            else:
                pos_loss += val.sum ()
            num += 1

        loss = pos_loss / num
        return loss, y_pred.detach().round(), reg_loss

    @staticmethod
    def get_bce_loss(y_pred, y_true, weight=None):
        """
        binary cross entropy loss
        normalized_activation_map: (batch_size, bag, n_class):
        y_true: (batch_size, n_class): count
        """
        b_y_true = (y_true > 0).float()
        if weight is not None:
            return F.binary_cross_entropy(y_pred, b_y_true, weight=weight.unsqueeze(1))
        else:
            return F.binary_cross_entropy(y_pred, b_y_true)


    @staticmethod
    def get_entropy_loss(normalized_activation_map, weight=None):
        # (bs, bag, n_class)
        # -y*log(y)
        entropy = (normalized_activation_map * (normalized_activation_map + 1e-12).log2()).sum(dim=2)
        if weight is not None:
            entropy = entropy * weight.unsqueeze(1)
        return -entropy.mean()

            # bag size = 8, >= 12.5% -> positive
    
        # 1) bag size = 8, count = k
        # bag size = 8, instance -> embedding -> sigmoid(fc(embedding)) -> score
        # sum(score over bags) -> MAE(sum, ground truth)
        
        # 2) topK.
        # in training, we have counting -> A: attention value for 8. pick topCounting(attent)
        
        # testing 
        # module -> predict the counting directly -> counting -> tokCounting()
        # ->loss (three losses: classification loss + regression loss + top)  
        
        
        # in a summary 
        # the following three method are for instance-level prediction.
        # -> a. fully supervised upperbound (ankit, simplify the experimenting,
        # can we use full CIFAR for this one.) (Resnet18 -> 94%)
        # multi-class classification -> image -> class 0,1,2.. -> 0/1 for a
        # specific class
        
        # -> b. weakly label learning baseline (no-counting) -> 94% for class 0
        # the positive 
        # supervised)(attention-threshold) -> binary classification
        # model could only prediction pos/neg
        
        # -> c. weakly label learning baseline (with counting supervised) ->
        # scalable analysis (bag-size && num of bags && # of positve instances
        # in a pos bag)
        
        # bag-level prediction
        # -> SVM method 
    
    
    
