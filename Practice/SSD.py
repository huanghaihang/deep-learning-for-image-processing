from d2l import torch as d2l
import torch
from torch import nn
#
# img = d2l.plt.imread('../Test1_official_demo/1.jpg')
# h, w = img.shape[:2]
# print((h,w))


#锚框类别预测器 #num_anchors是指一个像素自动生成多少个anchor，不是总的anchor数
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors*(num_classes+1), kernel_size=3, padding=1)

#边框回归(到真实bbox的偏移)
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors*4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2,8,20,20)),cls_predictor(8,5,10))
Y2= forward(torch.zeros((2,16,10,10)),cls_predictor(16,3,10))
# print(Y1.shape)
# print(Y2.shape)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0,2,3,1),start_dim=1)#prtmute把通道数放到最后
#拉成一个矩阵后就可以连接了
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

concat_preds([Y1,Y2]).shape
# print(Y3)

def down_sample_blk(in_channels, out_channels):
    blk=[]
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels= out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

#print(forward(torch.zeros((2,3,20,20)), down_sample_blk(3,20)).shape)

#特征提取器
def base_net():
    blk=[]
    num_filters=[3,16,32,64]
    for i in range(len(num_filters)-1):
        blk.append(down_sample_blk(num_filters[i],num_filters[i+1]))
    return nn.Sequential(*blk)

#print(forward(torch.zeros((2,3,256,256)),base_net()).shape) ------>[2, 64, 32, 32]


def get_blk(i):
    if i==0:
        blk = base_net()
    elif i==1:
        blk= down_sample_blk(64,128)
    elif i==4:
        blk = nn.AdaptiveAvgPool2d((1,1))
    else:
        blk = down_sample_blk(128,128)
    return blk




#为每个块定义前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):#size,ratio: 指定生成锚框的size和ratio
    Y = blk(X)#这个stage的feature map
    anchors = d2l.multibox_prior(Y,size= size, ratios=ratio)#生成锚框
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y,anchors,cls_preds,bbox_preds)
#对每个stage设置anchor的超参
sizes=[[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],[0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1



#完整的模型
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channel = [64,128,128,128,128]
        for i in range(5):#对每一个stage我们要定义他的blk(就是网络), cls_predictor和bbox_predictor
            setattr(self,f'blk_{i}',get_blk(i))
            setattr(self,f'cls_{i}',
                    cls_predictor(idx_to_in_channel[i],num_anchors,num_classes))
            setattr(self,f'bbox_{i}',
                    bbox_predictor(idx_to_in_channel[i],num_anchors))
    def forward(self,X):
        anchors, cls_preds, bbox_preds = [None]*5, [None]*5, [None]*5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X,getattr(self, f'blk_{i}'),sizes[i], ratios[i],getattr(self,f'cls_{i}'),getattr(self,f'bbox_{i}'))
        anchors = torch.cat(anchors,dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0],-1,self.num_classes+1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


#
# X = torch.zeros((32,3,256,256))
# anchors, cls_preds, bbox_preds= net(X)
#

batch_size= 32
train_iter = d2l.load_data_bananas(batch_size)

device, net = d2l.try_gpu(0), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(),lr=0.2, weight_decay=5e-4)

loss_function = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

#def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_mask):