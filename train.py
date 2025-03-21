import os
import numpy as np
import torch
import torchvision
import argparse
from modules import ae, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from dataloader import *
import copy
import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
import collections
from utils import transforms as T
from utils.preprocessor import Preprocessor
from utils.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from utils import IterLoader
from modules.cm import ClusterMemory
import torch.nn.functional as F
from sklearn import manifold


def inference(loader, model, device):    #测试
    model.eval()
    cluster_vector = []
    feature_vector = []
    for step, x in enumerate(loader):
        x = x.float().to(device)
        with torch.no_grad():
            c,h = model.forward_cluster(x)
        c = c.detach()
        h = h.detach()
        cluster_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())
    cluster_vector = np.array(cluster_vector)
    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return cluster_vector,feature_vector


#################################T-sne###############################
color = [(0.1, 0.1, 0.1, 1.0),
(0.5, 0.5, 0.5, 1.0),
(1.0, 0.6, 0.1, 1.0),
(1.0, 0.0, 0.0, 1.0),
(1.0, 0.1, 0.7, 1.0)]

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(15, 12), dpi=100)
    cx, cy = [], []
    r = []
    #print(X.shape[0])
    for i in range(X.shape[0]):
        cx.append(X[i, 0])
        cy.append(X[i, 1])
        plt.scatter(X[i, 0], X[i, 1], s=100, color=color[y[i]], edgecolor='black', marker='+')




################################ LIU ###############################
def extract_features(loader, model, device):  #提取特征 （数据加载，模型加载，设备加载）
    model.eval()
    feature_vector = []
    for step, x in enumerate(loader):  # 在加载的数据集里 进行以下操作
        x = x.float().to(device)
        with torch.no_grad():
            h = model.forward_(x)  #用network.py中的模型的前向传播函数
        h = h.detach()
        feature_vector.extend(h.cpu().detach().numpy())
    feature_vector = np.array(feature_vector)
    return feature_vector   #特征提取后 见network.py

def  parse_data(inputs):  #就是后面把特征和标签分开
        f,l = inputs
        return f.cuda(), l.cuda()
################################ LIU ###############################


def train(dataloader,model,m): #call @ line 190
    loss_epoch = 0
    for step, x in enumerate(DL):
        optimizer.zero_grad()
        x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)  
        x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)  
        z_i, z_j, c_i, c_j = model(x_i, x_j)    #生成数据增强的四组特征
        batch = x_i.shape[0]
        criterion_instance = contrastive_loss.DCL(temperature=0.5, weight_fn=None)
        criterion_cluster = contrastive_loss.ClusterLoss(cluster_number, args.cluster_temperature, loss_device).to(loss_device)
        loss_instance = criterion_instance(z_i, z_j)+criterion_instance(z_j, z_i)
        loss_cluster = criterion_cluster(c_i, c_j)
        #################################### LIU  ****************************
        inputs=dataloader.next()  # 这里的 loader 是有伪标签的
        inputs, labels = parse_data(inputs)
        inputs = inputs.float()
        f_out = model.forward_(inputs)
        loss_new = m(f_out, labels)
        #################################### LIU  ****************************
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch

def draw_fig(list,name,epoch):
    x1 = range(0, epoch+1)
    print(x1)
    y1 = list
    save_file = '/home/amax/4t/amax/CGLIU/second/MGCL/results/' + name + 'Train_loss.png'
    plt.cla()
    plt.title('Train loss vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig(save_file)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cancer_type", '-c', type=str, default="BRCA")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cluster_number', type=int,default=5)
    args = parser.parse_args()# 参数实例化
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,'LUAD': 3, 'PAAD': 2, 'SKCM': 4,'STAD': 3, 'UCEC': 4, 'UVM': 4, 'GBM': 2}
    
    cluster_number = cancer_dict[args.cancer_type]  # 按照癌症种类选择 
    print(cluster_number)

    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")


    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    model_path = './save/' + args.cancer_type
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = SummaryWriter(log_dir="./log")
    
    #load data  # dataloader确认，原始特征 (特征/样本数量 * (特征维度)) 
    DL,org_fea = get_feature(args.cancer_type, args.batch_size, True)   #这里的loader 是没有标签的，注意后面参数是True 的这里是用来提取特征的

    # initialize model
    ae = ae.AE()
    model = network.Network(ae, args.feature_dim, cluster_number) #模型就是network.py里的
    model = model.to(device)


    ############################ MGCL  ##################################
    features = extract_features(DL, model, device)  #模型提取的特征 (特征/样本数量 * 128(特征维度))
    features = torch.tensor(features)
    sim = features.mm(features.t())  #根据特征求出相似度矩阵
    cluster = KMeans(n_clusters=cluster_number, random_state=0)  #定义聚类方式
    pseudo_labels = cluster.fit_predict(sim) #根据聚类方式打出伪标签
    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0) #类个数

    @torch.no_grad()
    def generate_cluster_features(labels, features):  #输入为 伪标签 & 特征
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
           if label == -1:
              continue
           centers[labels[i]].append(features[i])

        centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
        centers = torch.stack(centers, dim=0)
        return centers  #输出为 质心特征（同类特征取均值）

    cluster_features = generate_cluster_features(pseudo_labels, features)  #得到质心矩阵

    memory = ClusterMemory(args.feature_dim, num_cluster, temp=args.temp,
                           momentum=args.momentum).cuda()
    memory.features = F.normalize(cluster_features, dim=1).cuda()


    pseudo_labeled_dataset = []
    for i, (feature, label) in enumerate(zip(org_fea, pseudo_labels)):
        pseudo_labeled_dataset.append((feature, label))  # 所有的原始特征 + 所有的伪标签


    train_loader = IterLoader(DataLoader(pseudo_labeled_dataset, args.batch_size))
    train_loader.new_epoch()


    ############################ MGCL  ##################################
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_device = device
    
    # train
    loss=[]
    for epoch in range(args.start_epoch, args.epochs+1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(train_loader,model,memory)
        loss.append(loss_epoch)
        logger.add_scalar("train loss", loss_epoch)
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
    save_model(model_path, model, optimizer, args.epochs)
    draw_fig(loss,args.cancer_type,epoch)
    
    #inference
    dataloader,org_feature = get_feature(args.cancer_type,args.batch_size,False) #这里的 loader也没有标签，是用来测试提取特征，最后参数为False
    
    # load model
    model = network.Network(ae, args.feature_dim, cluster_number)
    model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.epochs))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    X,h = inference(dataloader, model, device)
    output = pd.DataFrame(columns=['sample_name', 'dcc'])  # 建立新的DataFrame
    fea_tmp_file = '/home/amax/4t/amax/CGLIU/second/MGCL/subtype_file/fea/' + args.cancer_type + '/rna.fea'
    sample_name = list(pd.read_csv(fea_tmp_file).columns)[1:]
    output['sample_name'] = sample_name
    output['dcc'] = X+1
    out_file = './results/' + args.cancer_type +'.dcc'
    #output.to_csv(out_file, index=False, sep='\t')

    fea_out_file = './results/' + args.cancer_type +'.fea'
    fea = pd.DataFrame(data=h, index=sample_name,columns=map(lambda x: 'v' + str(x), range(h.shape[1])))
    #fea.to_csv(fea_out_file, header=True, index=True, sep='\t')

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(features)
    plot_embedding(X_tsne, pseudo_labels)
    plt.savefig('tsne.jpg')
