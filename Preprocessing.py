
# coding: utf-8
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
%matplotlib inline

TRAIN_DIR = 'I:/101_ObjectCategories/101_ObjectCategories/tick'
data_dir='I:/101_ObjectCategories/101_ObjectCategories'
IMG_SIZE = 50


def create_categories():    
    train=[]
    b=[]
    i=0
    for categ in os.listdir(data_dir)[10:20]:
        path = os.path.join(data_dir,categ)
        i=i+1
        for img in (os.listdir(path)):
            path2 = os.path.join(path,img)
            if (os.path.exists(path2)):
                gray = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
                img = cv2.equalizeHist(gray)
                img = cv2.GaussianBlur(img,(5,5),0)
#                img = cv2.Laplacian(img,cv2.CV_64F)
                img= cv2.Canny(img,100,200)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                img,cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                idx=0
                for c in cnts:
                    x,y,w,h = cv2.boundingRect(c)
                    if w>50 and h>50:
                        idx+=1
                        n_img=gray[y:y+h,x:x+w]
                img = cv2.equalizeHist(n_img)
                img = cv2.GaussianBlur(img,(5,5),0)
#                img= cv2.Canny(img,100,200)
#                img = cv2.Laplacian(img,cv2.CV_64F)
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


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(train.iloc[:,:2500])
data = pd.DataFrame(x_scaled)
data['category']=train['category']


ax = train.groupby('category').size().plot(kind='bar', figsize=(10,2))

n_samples, n_features = data.shape
n_digits = len(np.unique(data.category))
labels = data.category
sample_size = len(data)

# In[54]:


from sklearn import neighbors, datasets
n=1
clf = neighbors.KNeighborsClassifier(n, weights='distance')
clf.fit(x, y)


# In[55]:

from sklearn.metrics import accuracy_score
print(accuracy_score(y,clf.predict(x)))


# In[2]:

image=cv2.imread('I:/101_ObjectCategories/101_ObjectCategories/airplanes/image_0002.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')


# In[117]:

blur = cv2.blur(image,(5,5))

plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.show()


# In[107]:

plt.hist(gray.ravel(),256,[0,256]); plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# In[90]:

equ = cv2.equalizeHist(gray)
res = np.hstack((gray,equ))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
plt.subplot(221),plt.imshow(gray,cmap='gray')
plt.subplot(222),plt.imshow(cl1,cmap='gray')
np.array(cl1)


# In[91]:

ret, thresh = cv2.threshold(cl1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# In[92]:

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


# In[93]:

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(image,markers)
image[markers == -1] = [255,0,0]


# In[95]:

plt.imshow(unknown, cmap = 'gray', interpolation = 'bicubic')
plt.show()


# In[1]:

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


# In[29]:

def other_features(arr,img_size):
    img = np.reshape(arr,(img_size[0],img_size[1],3))
    u = np.sum(img,axis=0).tolist()
    v = np.sum(img,axis=1).tolist()
    w = np.sum(img,axis=2).tolist()
    out = []
    for x in u+v+w:
        out += x
    return np.array(out)


# In[37]:

ot=other_features(im,(50,50))
np.append(avg,ot)

