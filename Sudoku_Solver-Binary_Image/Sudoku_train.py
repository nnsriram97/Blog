from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread('D:\Sriram\pyprograms\sudoku\sudoku.png',1)
frame=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
samples=np.empty((0,100))
responses=[]
keys = [i for i in range(48,58)]
_,cnt,hr = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,cnt,-1,(255,0,0),3)
plt.imshow(img),plt.title("frame")
plt.xticks(),plt.yticks()
plt.show()
cv2.waitKey(0)
count=0
numb=[]
for i in range(len(cnt)):
	area=cv2.contourArea(cnt[i])
	if area>=40000:
		x,y,w,z=hr[0][i]
		if w!=-1:
			a,b,c,d=cv2.boundingRect(cnt[w])
			print c,d
			errx=100-c
			erry=180-d
			b=b-(erry/2)
			a=a-(errx/2)
			c=100
			d=180
			roi=frame[b:b+d,a:a+c]
			cv2.imshow("1",roi)
			cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,0),4)
			q=cv2.resize(img,(400,400))
			cv2.imshow("frame",q)
			count=count+1
			key=cv2.waitKey(0)
			if key in keys:
				responses.append(int(chr(key)))
				roismall=cv2.resize(roi,(10,10))
				roismall=roismall.reshape(1,100)
				samples=np.append(samples,roismall,0)
				if key not in numb:
					numb.append(key)
					picname= chr(key)+".png"
					cv2.imwrite(picname,img[b:b+d,a:a+c])
plt.imshow(img),plt.title("frame")
plt.xticks(),plt.yticks()
plt.show()
cv2.waitKey(0)
print count
samples=np.array(samples,np.float32)
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
np.savez('knn.npz',train=samples,traindata=responses)
print "training complete"


