
# coding: utf-8

# In[1]:

import os  
import cv2           
from time import time
 
import pandas as pd
import numpy as np 

from random import shuffle  

from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,silhouette_score, homogeneity_score,adjusted_mutual_info_score,completeness_score,v_measure_score,adjusted_rand_score

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as pdf
get_ipython().magic('matplotlib inline')


# In[40]:

data_dir='I:/101_ObjectCategories/101_ObjectCategories'
IMG_Size = 50


# In[ ]:

def create_categories():    
    train=[]
    b=[]
    i=0
    for categ in os.listdir(data_dir):
        path = os.path.join(data_dir,categ)
        i=i+1
        for img in (os.listdir(path))[:40]:
            path2 = os.path.join(path,img)
            if (os.path.exists(path2)):
                gray = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(path2)


                equ = cv2.equalizeHist(gray)
                res = np.hstack((gray,equ))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img = clahe.apply(gray)
                img = cv2.blur(img,(5,5))
#                 # noise removal
#                 ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#                 kernel = np.ones((3,3),np.uint8)
#                 opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

#                 # sure background area
#                 sure_bg = cv2.dilate(opening,kernel,iterations=3)

#                 # Finding sure foreground area
#                 img= cv2.distanceTransform(opening,cv2.DIST_L2,5)
                img = cv2.resize(img, (IMG_Size,IMG_Size))
                img = img.reshape(IMG_Size*IMG_Size)
                train.append(list(img))
                b.append(i)
    np.save('train_data.npy', train)
    train=pd.DataFrame(train)
    train['category']=b
    return train
data=shuffle(create_categories())
train=data
print(data.shape)


#                                     ##########################
#                                     ####### Clustering #######
#                                     ##########################

# In[ ]:

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(train.iloc[:,:2500])
data = pd.DataFrame(x_scaled)


# In[ ]:

data['category']=train['category']


# In[ ]:

data.head()


# In[175]:

n_samples, n_features = data.shape
n_digits = len(np.unique(data.category))
labels = data.category
sample_size = len(data)


# In[176]:

np.random.seed(42)

print(79 * '_')
print('% 9s' %    'init'
      '        time   inertia   homo    compl   v-meas   ARI    AMI    silhouette')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             homogeneity_score(labels, estimator.labels_),
             completeness_score(labels, estimator.labels_),
             v_measure_score(labels, estimator.labels_),
             adjusted_rand_score(labels, estimator.labels_),
             adjusted_mutual_info_score(labels,  estimator.labels_),
             silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

#kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(79 * '_')


# In[177]:

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=80)
kmeans.fit(reduced_data)


# In[178]:

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1


# In[179]:

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))


# In[180]:

type(reduced_data)
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])


# In[181]:

type(Z)
Z = Z.reshape(xx.shape)


# In[182]:

Z.shape


# In[183]:

plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
pdf.savefig("depth")
plt.show()


#                                    ########################################
#                                    ######Applying K Nearest Neighbors######
#                                    ########################################

# In[184]:

data.head()


# In[185]:

train_input=train.iloc[:int(len(train)*0.7),:2500].as_matrix()
train_output=train['category'][:int(len(train)*0.7)].as_matrix()
test_input=train.iloc[int(len(train)*0.7):,:2500].as_matrix()
test_output=train['category'][int(len(train)*0.7):].as_matrix()


# In[186]:

print(train_input.shape)
print(train_output.shape)
print(test_input.shape)
print(test_output.shape)


# In[187]:

acc=[]
conf=[]
for n in range(1,15):
    clf= KNeighborsClassifier(n_neighbors=n,weights='distance')
    clf.fit(train_input, train_output)
    predicted_output=clf.predict(test_input)
    acc.append(accuracy_score(test_output, predicted_output)*100)
    conf.append(confusion_matrix(test_output, predicted_output))


# In[188]:

plt.plot(acc,color='red')
plt.title("different neighbors")
plt.show('estimators')  
plt.close()


#                                   #####################################
#                                   #### Plotting Confusion matrix ######
#                                   #####################################

# In[161]:

conf_mat_ind=10


# In[162]:

norm_conf = []
for i in conf[conf_mat_ind]:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

width, height = conf[conf_mat_ind].shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf[conf_mat_ind][x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('confusion_matrix.png', format='png')


#                                         ########################
#                                         #############Neural Network###########
#                                         ########################

# In[168]:

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(train_input, train_output)                         
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)


# In[169]:

pred=clf.predict(test_input)
accuracy_score(test_output, pred)*100


# In[170]:

confusion_matrix(test_output, predicted_output)


# In[36]:

image=cv2.imread('I:/101_ObjectCategories/101_ObjectCategories/airplanes/image_0002.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')


# In[37]:

blur = cv2.blur(image,(5,5))

plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.show()


# In[38]:

plt.hist(gray.ravel(),256,[0,256]); plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# In[39]:

equ = cv2.equalizeHist(gray)
res = np.hstack((gray,equ))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
plt.subplot(221),plt.imshow(gray,cmap='gray')
plt.subplot(222),plt.imshow(cl1,cmap='gray')
np.array(cl1)


# In[ ]:

ret, thresh = cv2.threshold(cl1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# In[ ]:




# In[ ]:




# In[ ]:

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


# In[ ]:

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(image,markers)
image[markers == -1] = [255,0,0]


# In[ ]:

plt.imshow(unknown, cmap = 'gray', interpolation = 'bicubic')
plt.show()


# In[ ]:

def averages(arr,img_size):
    img = np.reshape(arr,(img_size[0],img_size[1],3))
    uAvg = np.average(img,axis=0).tolist()
    vAvg = np.average(img,axis=1).tolist()
    wAvg = np.average(img,axis=2).tolist()
    print(len(uAvg),len(vAvg),len(wAvg))
    out = []
    for x in uAvg+vAvg+wAvg:
        out += x
    return np.array(out)


# In[ ]:

def other_features(arr,img_size):
    img = np.reshape(arr,(img_size[0],img_size[1],3))
    u = np.sum(img,axis=0).tolist()
    v = np.sum(img,axis=1).tolist()
    w = np.sum(img,axis=2).tolist()
    out = []
    for x in u+v+w:
        out += x
    return np.array(out)


# In[ ]:

ot=other_features(im,(50,50))
np.append(avg,ot)

