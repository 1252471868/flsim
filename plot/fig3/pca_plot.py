import pickle
import matplotlib.pyplot as plt
import numpy as np


with open("output/clients_weights_pca.pkl", 'rb') as a:
    data_weight1 = pickle.load(a)

with open("output/clients_prefs.pkl", 'rb') as b:
    data_label1 = pickle.load(b)

x1 = data_weight1[:,0]
y1 = data_weight1[:,1]
l1 = data_label1
total1 = np.zeros((3,2))
total1[0] = l1
total1[1] = x1
total1[2] = y1
color = np.array(["red","green","blue","yellow","pink","black","orange","purple","cyan","brown"])
total1 = np.transpose(total1)

fig, ax = plt.subplots()
for i in range(10):
    label = "label: "+str(i)
    index = np.where(total1[:,0]==i)
    for j in index:
        ax.scatter(total1[j,1], total1[j,2], c=color[i], label = label)

ax.legend()
plt.title('PCA on the MNIST dataset(1 client)')
plt.xlabel('C0')
plt.ylabel('C1')

plt.savefig("output/clients_pca.png")


plt.show()

# with open("../output/pca_20clients/clients_weights_pca.pkl", 'rb') as c:
#     data_weight2 = pickle.load(c)

# with open("../output/pca_20clients/clients_prefs.pkl", 'rb') as d:
#     data_label2 = pickle.load(d)

# x2 = data_weight2[:,0]
# y2 = data_weight2[:,1]
# l2 = data_label2
# total2 = np.zeros((3,20))
# total2[0] = l2
# total2[1] = x2
# total2[2] = y2
# color = np.array(["red","green","blue","yellow","pink","black","orange","purple","cyan","brown"])
# total2 = np.transpose(total2)

# fig, ax = plt.subplots()
# for i in range(10):
#     label = "label: "+str(i)
#     index = np.where(total2[:,0]==i)
#     for j in index:
#         ax.scatter(total2[j,1], total2[j,2], c=color[i], label = label)

# ax.legend()
# plt.title('PCA on the MNIST dataset(20 clients)')
# plt.xlabel('C0')
# plt.ylabel('C1')

# plt.savefig("../plots/20_clients_pca.png")


# plt.show()