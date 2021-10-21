import numpy as np
import pandas as pd
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch import nn
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns


labels_dataframe = pd.read_csv('../data_set/leaves_data/train.csv')
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v : k for k, v in class_to_num.items()}



class LeavesData(Dataset):
    def __init__(self, csv_file, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        self.data_info = pd.read_csv(csv_file,header=None)
        self.data_len = len(self.data_info.index)-1
        self.train_len = int(self.data_len*(1-valid_ratio))


        if mode == 'train':
            self.train_img = np.asarray(self.data_info.iloc[1:self.train_len,0])
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len,1])
            self.image_arr = self.train_img
            self.label_arr = self.train_label
        elif mode=='valid':
            self.valid_img = np.asarray(self.data_info.iloc[self.train_len:,0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:,1])
            self.image_arr = self.valid_img
            self.label_arr = self.valid_label
        elif mode=='test':
            self.test_img = np.asarray(self.data_info.iloc[1:,0])
            self.image_arr = self.test_img


        self.real_len = len(self.image_arr)

        print('Finish read the mode {} set of Leaves Dataset({} samples found)'.format(mode,self.real_len))



    def __getitem__(self, index):
        single_image_name = self.image_arr[index]

        img_as_img = Image.open(self.file_path + single_image_name)

        if self.mode =='train':
            transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomHorizontalFlip(p=0),
                                            transforms.ToTensor()
                                            ])
        else:
            transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor()])
        img_as_img = transform(img_as_img)


        if self.mode=='test':
            return img_as_img
        else:
            label = self.label_arr[index]
            number_label = class_to_num[label]
            return img_as_img, number_label

    def __len__(self):
        return self.real_len


train_path = '../data_set/leaves_data/train.csv'
test_path = '../data_set/leaves_data/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = '../data_set/leaves_data/'


train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')

#定义data loader

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=False,num_workers=5)
valid_loader = torch.utils.data.DataLoader(val_dataset,batch_size=8,shuffle=False,num_workers=5)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=8,shuffle=False,num_workers=5)


# 给大家展示一下数据长啥样
# def im_convert(tensor):
#     """ 展示数据"""
#
#     image = tensor.to("cpu").clone().detach()
#     image = image.numpy().squeeze()
#     image = image.transpose(1, 2, 0)
#     image = image.clip(0, 1)
#
#     return image
#
#
# fig = plt.figure(figsize=(20, 12))
# columns = 4
# rows = 2
#
# dataiter = iter(valid_loader)
# inputs, classes = dataiter.next()
#
# for idx in range(columns * rows):
#     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
#     ax.set_title(num_to_class[int(classes[idx])])
#     plt.imshow(im_convert(inputs[idx]))
# plt.show()


#看是在cpu还是gpu
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(device)

def set_parameter_require_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.require_grad = False

def res_model(num_classes, feature_extracting=False, use_pretrained = True):
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_require_grad(model_ft,feature_extracting)#冻结特征层参数
    num_ftrs = model_ft.fc.in_features#num_ftrs = net.fc.in_features # 提取fc层的输入参数
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))# 修改输出维度为num_classes

    return model_ft


#定义超参数
learning_rate = 0.1
weight_decay = 1e-3
num_epoch = 20
model_path ='./pre_res_model.ckpt'

model = res_model(176)
model = model.to(device)
model.device = device
#定义损失函数
loss_function = nn.CrossEntropyLoss()
#定义优化器
optimizer = torch.optim.Adam(model.parameters(),learning_rate, weight_decay=weight_decay)

n_epoch = num_epoch
best_acc = 0.0
for epoch in range(n_epoch):
    model.train()
    train_loss = []
    train_acc= []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)


        outputs = model(imgs)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()
        #计算acc
        acc = (outputs.argmax(dim=-1)==labels).float().mean()

        #record the loss and accuracy.
        train_loss.append(loss.item())
        train_acc.append(acc)

        #算损失和acc
    train_loss = train_loss.sum()/len(train_loss)
    train_acc = train_acc.sum/ len(train_acc)

    print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d}] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

# ---------- Validation ----------
model.eval()

valid_loss=[]
valid_accs=[]
for batch in tqdm(valid_loader):
    imgs, labels = batch
    #不参与参数更新
    with torch.no_grad():
        logits = model(imgs.to(device))

    outputs = model(imgs)
    loss_function = nn.CrossEntropyLoss()

    loss = loss_function(imgs, labels)


    acc = (outputs.argmax(dim=-1)==labels.to(device)).float().mean()

    valid_loss.append(loss.item())
    valid_accs.append(acc)

valid_loss = sum(valid_loss) / len(valid_loss)
valid_acc = sum(valid_accs) / len(valid_accs)

print(f"[Valid | {epoch+1:03d}/{n_epoch:03d}] loss ={valid_loss:.5f}, acc = {valid_acc:.5f}")

if valid_acc>best_acc:
    best_acc = valid_acc
    torch.save(model.state_dict(), model_path)
    print('saving model with acc {:.3f}'.format(best_acc))



#predict
saveFileName = './submission.csv'

model = res_model(176)

model = model.to(device)
model.load_state_dict(torch.load(model_path))

model.eval()
predictions=[]
for batch in tqdm(test_loader):
    imgs = batch
    with torch.no_grad():
        outputs = model(imgs.to(device))
    predictions.extend(outputs.argmax(dim=-1).cpu().numpy().tolist())

preds=[]
for i in predictions:
    preds.append(num_to_class(i))
test_data = pd.read_csv(test_path)
test_data['label'] = pd.Series(preds)
submission = pd.concat([test_data['image'], test_data['label']], axis=1)
submission.to_csv(saveFileName, index=False)
print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")

