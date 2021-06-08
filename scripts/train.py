from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
from tqdm import tqdm
# import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint
import argparse

use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
data_dir = '/root/workplace/img_classification/data/miniimagenet'
batch_size = 16		#批次大小（通过loaddata函数打包）
lr = 0.01  		  #学习率
momentum = 0.9     #动量
num_epochs = 100     #训练轮次
input_size = 224    #数据集图像处理大小
class_num = 100     #分多少个类别
#net_name = 'efficientnet-b2'	#需要用到的EfficientNet预训练模型名称
i=0
Species_id = []
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')

def loaddata(data_dir, batch_size, set_name, shuffle):
    #对数据进行数据增强
    data_transforms = {
         "train": transforms.Compose([transforms.RandomResizedCrop(input_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(388),
                                   transforms.CenterCrop(388),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers=0 if CPU else =1
    #创建了一个 batch，生成真正网络的输入
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=0) for x in [set_name]}
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes
    
def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    train_loss = []
    loss_all = []
    acc_all = []
    since = time.time()
    best_model_wts = model_ft.state_dict() #存放训练过程中需要学习的权重和偏执系数
    best_acc = 0.0
    #作用是启用batch normalization和drop out。
    model_ft.train(True)
    for epoch in range(1,num_epochs):
        data_loader, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train', shuffle=True)
        # print(dset_loaders)
        print('Data Size', dset_sizes)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        optimizer = lr_scheduler(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0
        count = 0
        data_loader = data_loader['train']
        data_loader = tqdm(data_loader)

        for step, data in enumerate(data_loader):
            #inputs和labels的个数为64个，等于一个batch,这个循环执行21次,1323/64=20.67
            inputs, labels = data
            #torch.LongTensor是64位整型
            labels = torch.squeeze(labels.type(torch.LongTensor))#主要对数据的维度进行压缩，去掉维数为1的的维度
            if use_gpu:
                #Variable可以看成是tensor的一种包装，其不仅包含了tensor的内容，还包含了梯度等信息，因此在神经网络中常常用Variable数据结构。
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
             
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            #取概率最大的值
            _, preds = torch.max(outputs.data, 1)

            #print(_, preds)
            #optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            optimizer.zero_grad()
            #计算得到loss后就要回传损失
            loss.backward()
            #回传损失过程中会计算梯度，然后需要根据这些梯度更新参数
            optimizer.step()

            count += 1

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes
        loss_all.append(int(epoch_loss*100))
        acc_all.append(int(epoch_acc*100))
        # print(epoch_loss)

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_ft.state_dict()
        if epoch_acc == 1.0:
            break
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    
        # 保存训练模型
        save_dir = './models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_ft.load_state_dict(best_model_wts)
        model_out_path = save_dir + '/{}.pth'.format('model_name')
        torch.save(best_model_wts, model_out_path)

    # return train_loss, best_model_wts,model_ft

def evaluate(model):
    model.eval()

    # 用于存储预测正确的样本个数
    running_corrects = 0
    data_loader,data_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='val', shuffle=True)
    print('Data Size', data_sizes)
    data_loader = data_loader['val']
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
    #for data in data_loader['val']:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))#主要对数据的维度进行压缩，去掉维数为1的的维度
        if use_gpu:
        #Variable可以看成是tensor的一种包装，其不仅包含了tensor的内容，还包含了梯度等信息，因此在神经网络中常常用Variable数据结构。
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
             
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        #取概率最大的值
        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects / data_sizes
    
    print(' Acc: {:.4f}'.format(acc))

def get_key(dct, value):
    return [k for (k, v) in dct.items() if v == value]

def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer



def main(needTrain=True,needTest=True):
    
    args = parser.parse_args()
        # train
   
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    if  needTrain:
        # 离线加载预训练
      
        net_weight = '' 
        '''if os.path.exists(net_weight):
            
            # 修改全连接层
            num_ftrs = model._fc.in_features
            model._fc = nn.Linear(num_ftrs, class_num)

            state_dict = torch.load(net_weight)
            model.load_state_dict(state_dict)
            print('load weight from {}'.format(net_weight))
        else:
            num_ftrs = model._fc.in_features
            model._fc = nn.Linear(num_ftrs, class_num)'''
        # 修改全连接层
        '''num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, class_num)'''

        criterion = nn.CrossEntropyLoss()   #获得交叉熵损失
        if use_gpu:
            model = model.cuda()
            criterion = criterion.cuda()
            
        optimizer = optim.SGD((model.parameters()), lr=lr,
                            momentum=momentum, weight_decay=0.0002)

        # train_loss, best_model_wts,model_ft= train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
        train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

    # test
    if needTest:
        weight_path = '/root/workplace/pytorch-image-models-master/models/model_name.pth'
        assert os.path.exists(weight_path)
        # 修改全连接层
        '''num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, class_num)'''

        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict)
        print('load weight from {}'.format(weight_path))
        if use_gpu:
            model = model.cuda()
        evaluate(model)


    # test
    '''if not needTrain:
        best_model_wts =  data_dir +"/model/" +net_name+".pth"
        best_model_wts = torch.load(best_model_wts)

                # 修改全连接层
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, class_num)

        criterion = nn.CrossEntropyLoss()   #获得交叉熵损失
        if use_gpu:
            model_ft = model_ft.cuda()
            criterion = criterion.cuda()
        model_ft.load_state_dict(best_model_wts)
        

       
    
    data_transforms = transforms.Compose([
        transforms.Resize(350),
        transforms.CenterCrop(350),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        '''




        
    '''Species_id = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # find the mapping of folder-to-label

    data = datasets.ImageFolder('/home/incar/data/mini-imagenet/splitimages/train')
    mapping = data.class_to_idx
    print(mapping)
        
    # start testing
    if needTest:
        data_file = pd.read_csv('/home/incar/data/mini-imagenet/splitimages/mytest.csv')
        File_id = data_file["FileID"].values.tolist()

        for i in range(len(File_id)):
            test_dir = File_id[i] + '.jpg'
            img_dir = '/home/incar/data/mini-imagenet/splitimages/mytest/'+test_dir
            # load image
            if  not  os.path.isfile(img_dir):
                print("no  file ".format(img_dir))
                Species_id.append("none")
                continue
            if os.stat(img_dir).st_size == 0:
                print("ignore  file ".format(img_dir))
                Species_id.append("none")
                continue
                            
            img = Image.open(img_dir)
            inputs = data_transforms(img)
            inputs.unsqueeze_(0)

            if use_gpu:
                model = model_ft.cuda() # use GPU
            else:
                model = model_ft
            model.eval()
            if use_gpu:
                inputs = Variable(inputs.cuda()) # use GPU
            else:
                inputs = Variable(inputs)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            class_name = get_key(mapping, preds.item())
            class_name = '%s' % (class_name) 
            class_name = class_name[2:-2]
            
            print(img_dir)
            print('prediction_label:', class_name)
            print(30*'--')
            Species_id.append(class_name)

        test = pd.DataFrame({'FileId':File_id,'SpeciesID':Species_id}) #将结果存储在.csv文件中
        test.to_csv('result.csv',index = None,encoding = 'utf8')
'''

if __name__ == '__main__':
    main(needTest=True,needTrain=False)

