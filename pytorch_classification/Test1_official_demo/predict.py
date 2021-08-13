import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet

#调用训练好的模型权重(Lenet.pth)进行预测
def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('1.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
        # predict = torch.softmax(outputs, dim=1)用softmax的方法
        # predict = torch.argmax(predict).data.numpy()
        # print(predict)
    print(classes[int(predict)])#索引


if __name__ == '__main__':
    main()
