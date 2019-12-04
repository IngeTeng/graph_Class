import numpy as np
import os
TrainFcmMat = np.load(file='Graph_datasets/TrainGCN.npy')
TestFcmMat = np.load(file='Graph_datasets/TestGCN.npy')
#合并数据
allFcmMat = np.vstack((TrainFcmMat,TestFcmMat))
#print(allFcmMat.shape)
n_node = (TrainFcmMat.shape[0]+TestFcmMat.shape[0])*TestFcmMat.shape[2]
A_maxtrix = []
edge_attributes = []
graph_indicator = []
graph_labels = []
# tmp_j = 0
# tmp_k = 0
for i in range(TrainFcmMat.shape[0]):
    #print(TrainFcmMat[i,0,0])
    if TrainFcmMat[i,0,0] == 1:
        graph_labels.append(-1)
    else:
        graph_labels.append(1)

    for j in range(1,TrainFcmMat.shape[1]):
        graph_indicator.append(i)

        for k in range(TrainFcmMat.shape[2]):
            tmp_j = i * TrainFcmMat.shape[2]
            tmp_k = i * TrainFcmMat.shape[2]
            #表示有连接
            if TrainFcmMat[i,j,k] >= 0.05:
                #每一个节点都得有单独的id
                x = tmp_j+j
                y = tmp_k+k+1
                A_maxtrix.append([x,y])

                edge_attributes.append(TrainFcmMat[i,j,k])

#节点个数
# n_node = (TrainFcmMat.shape[0]+TestFcmMat.shape[0])*TestFcmMat.shape[2]
# #需要构造A.txt   edge_attributes.txt   graph_indicator.txt      graph_labels.txt
# A_maxtrix = []
# edge_attributes = []
# graph_indicator = []
# graph_labels = []
# #训练集
# for i in range(TrainFcmMat.shape[0]):
#     #print(TrainFcmMat[i,0,0])
#     if TrainFcmMat[i,0,0] == 1:
#         graph_labels.append(-1)
#     else:
#         graph_labels.append(1)
#     for j in range(1,TrainFcmMat.shape[1]):
#         graph_indicator.append(i)
#         for k in range(TrainFcmMat.shape[2]):
#             #表示有连接
#             if TrainFcmMat[i,j,k] >= 0.05:
#                 A_maxtrix.append([j-1,k])
#                 edge_attributes.append(TrainFcmMat[i,j,k])
# #测试集
# for i in range(TestFcmMat.shape[0]):
#     #print(TrainFcmMat[i,0,0])
#     if TestFcmMat[i,0,0] == 1:
#         graph_labels.append(-1)
#     else:
#         graph_labels.append(1)
#     for j in range(1,TestFcmMat.shape[1]):
#         graph_indicator.append(i)
#         for k in range(TestFcmMat.shape[2]):
#             #表示有连接
#             if TestFcmMat[i,j,k] >= 0.05:
#                 A_maxtrix.append([j-1,k])
#                 edge_attributes.append(TestFcmMat[i,j,k])
#
A_maxtrix = np.array(A_maxtrix)
edge_attributes = np.array(edge_attributes)
graph_indicator = np.array(graph_indicator)
graph_labels = np.array(graph_labels)
dataset_name = 'BeetleFly'
path = 'Graph_datasets/'+dataset_name
folder = os.path.exists(path)

# 判断结果
if not folder:
    # 如果不存在，则创建新目录
    os.makedirs(path)
    print('-----创建成功-----')

else:
    # 如果目录已存在，则不创建，提示目录已存在
    print(path + '目录已存在')

np.savetxt(path+'/'+dataset_name+'_A.txt',A_maxtrix,fmt='%d',delimiter=', ')
np.savetxt(path+'/'+dataset_name+'_edge_attributes.txt',edge_attributes,fmt='%0.8f',delimiter='')
np.savetxt(path+'/'+dataset_name+'_graph_indicator.txt',graph_indicator,fmt='%d',delimiter='')
np.savetxt(path+'/'+dataset_name+'_graph_labels.txt',graph_labels,fmt='%d',delimiter=' ')