import pandas as pds
import numpy as np
import os
import glob
# import seaborn as sns
import matplotlib.pyplot as plt
from shutil import copyfile

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

data_file = pds.read_excel('../256.192.model/objec_para.xlsx')
X = np.asarray(data_file)
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2], c=kmeans.labels_, cmap='rainbow')
# plt.show()
path = '../clusters/'
import pdb; pdb.set_trace()
if not os.path.isdir(path):
    os.mkdir(path)

for i in range(kmeans.labels_.shape[0]):

    cluster_id = kmeans.labels_[i]
    cluster_id_path = path + str(cluster_id)
    if not os.path.isdir(cluster_id_path):
        os.mkdir(cluster_id_path)
    else:
        files = glob.glob('../256.192.model/individual/' + str(i)+'_'+'*.png')
        for file in files:
            copyfile(file, cluster_id_path + '/'+os.path.basename(file))

    # import pdb; pdb.set_trace()
import pdb; pdb.set_trace()
print(data_file)
