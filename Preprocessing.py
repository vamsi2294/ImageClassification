
# coding: utf-8

# In[1]:

import cv2                 
import numpy as np        
import os                 
from random import shuffle
from tqdm import tqdm     
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

TRAIN_DIR = 'I:/101_ObjectCategories/101_ObjectCategories/tick'
cat_dir='I:/101_ObjectCategories/101_ObjectCategories'
IMG_SIZE = 50
LR = 1e-3


# In[3]:

def create_categories():    
    train=[]
    b=[]
    i=0
    for categ in tqdm(os.listdir(cat_dir)[:10]):
        path = os.path.join(cat_dir,categ)
        i=i+1        
        for img in (os.listdir(path))[:20]:
            path2 = os.path.join(path,img)
            if (os.path.exists(path2)):     
                img = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                img = img.reshape(50*50)
                train.append(list(img))
                b.append(i)
    np.save('train_data.npy', train)
    train_df=pd.DataFrame(train)
    train_df['category']=b
    return train_df


# In[4]:

train=create_categories()


# In[5]:

from sklearn.utils import shuffle
train=shuffle(train)
train


# In[39]:

train.groupby('category').count()


# In[7]:

x=train[:,:2499]
y=train['category'].as_matrix()


# In[52]:

x=x.as_matrix()


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

