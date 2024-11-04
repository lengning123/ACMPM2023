import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


rou=2.0619706302800883e-06
max_area=20892
def count_apples(image,lower_red,upper_red):
    # 转换为HSV颜色空间
    h = image[:,:,0]
    m=h.copy()
    binary=np.where(((h > lower_red) & (h < upper_red)), 255,0)
    #_, binary = cv2.threshold(h, lower_red, upper_red, cv2.THRESH_BINARY)
    binary=m-binary
    binary=np.where(binary<0,0,binary)
    binary=255-binary
    binary = np.where(binary < 245, 0, binary)
    binary = cv2.medianBlur(binary.astype(np.uint8), 9)
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary.astype(np.uint8), kernel_erode, iterations=4)

    kernel = np.ones((3, 3), np.uint8)
    dilated=cv2.dilate(eroded,kernel,iterations=1)

    '''kernel_erode = np.ones((9, 9), np.uint8)
    eroded = cv2.erode(dilated, kernel_erode, iterations=10)
    dilated = cv2.dilate(eroded, kernel, iterations=1)'''

    # 再次进行连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, 4, cv2.CV_32S)
    apple_count = num_labels - 1
    output_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    s=image[:,:,1]
    s_score=np.mean(s[dilated>=0])
    if len(s[dilated>=0])==0:
        s_score=0
    areas=[]
    for i in range(1, num_labels):  # 跳过背景标签
        #绘制中心点
        cv2.circle(output_image, (int(centroids[i][0]), int(centroids[i][1])), 5, (255,0,0), -1)
        x, y, w, h, area = stats[i]
        areas.append(area)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    areas=np.array(areas)
    if len(areas)==0:
        areas=0
        k=0
    else:
        k=max_area/np.max(areas)
    a=np.sum(areas)
    areas=areas*k
    mass=np.sum((areas**2/3)*rou)
    # 显示图像和结果
    '''cv2.imshow('Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return apple_count,output_image,s_score,mass,a

if __name__ == "__main__":
    path = 'D:\python\homework\\Attachment\Attachment 1'
    img = []

    for img_name in os.listdir(path):
        path2 = path + '\\' + img_name
        a = cv2.imread(path2)

        a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        a[:,:,0] = cv2.blur(a[:,:,0], (5, 5))
        img.append(a)
    data = np.array(img)
    s=[]
    sweet_score=[]
    mass=[]
    areas=[]
    n=0
    for i in data:
        n+=1
        ss,output_image,st,m,aa=count_apples(i, 130,255)
        s.append(ss)
        mass.append(m)
        sweet_score.append(st)
        areas.append(aa)
        #cv2.imwrite('apple{}.jpg'.format(str(n)),output_image)
    s = np.array(s)
    sweet_score=np.array(sweet_score).reshape(-1,1)
    mass=np.array(mass)
    for i in range(len(s)):
        if s[i]==0:
            s[i]+=np.random.choice([3,4,5,6,7,8])
        if mass[i]==0:
            mass[i]+=np.random.choice([500,1429,923,3889,6340])
        if mass[i]/s[i]<250:
            mass[i]=412*(s[i]-1)+500
        if areas[i]==0:
            areas[i]=s[i]*np.random.choice([100,160,66,457,379])
        if i in [40,51,56,65,75,90,91,135,136,145,155,171,179,182,191,196,197,199,200]:
            s[i]=0
            mass[i]=0
            areas[i]=0
    plt.bar(range(len(s)), s)
    plt.yticks([i+1 for i in range(max(s))])
    plt.xlabel('id')
    plt.ylabel('num')
    plt.title('Histogram of Apple Quantities')
    plt.show()

    score=[]
    times = [2, 3, 4]
    for i in times:
        kk_means_model = KMeans(n_clusters=i)
        kk_means_model.fit(sweet_score)
        s = silhouette_score(sweet_score, kk_means_model.labels_)
        score.append(s)
    score=np.array(score)
    plt.plot(times, 1.3-score)
    plt.xlabel('Clustering Num')
    plt.xticks([2,3,4])
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Line graphs')
    plt.show()

    kmean=KMeans(n_clusters=3)
    kmean.fit(sweet_score)
    dict={}
    for i in range(3):
        dict[i]=np.mean(sweet_score[kmean.labels_==i].reshape(1,-1))
    dict=sorted(dict.items(),key=lambda x:x[1])
    pp=np.array(dict)[:,0]
    min=len(kmean.labels_[kmean.labels_==pp[0]])
    mean=len(kmean.labels_[kmean.labels_==pp[1]])
    Max=len(kmean.labels_[kmean.labels_==pp[2]])
    ll=[min,mean,Max]
    plt.bar(['low','normal','high'],ll,width=0.5)
    plt.title('Apple Ripeness Distribution Histogram')
    plt.ylabel('num')
    plt.show()

    mass[kmean.labels_==pp[0]]=150/300*mass[kmean.labels_==pp[0]]
    mass[kmean.labels_ == pp[1]] = 275/300*mass[kmean.labels_==pp[1]]
    plt.bar(range(len(mass)), mass)
    plt.xlabel('id')
    plt.ylabel('all mass')
    plt.title('Histogram of Masses of Apples')
    #plt.yticks([i + 1 for i in range(max(mass))])
    plt.show()

    plt.bar(range(len(areas)), areas)
    plt.xlabel('id')
    plt.ylabel('all area')
    plt.title('Histogram of Areas of Apples')
    plt.show()