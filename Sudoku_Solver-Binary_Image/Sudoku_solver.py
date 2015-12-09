from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread('D:\Sriram\pyprograms\sudoku\sudoku.png',1)
deflt="D:\Sriram\pyprograms\sudoku\\"
frame=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print frame.shape
samples=np.empty((0,100))
responses=[]
keys = [i for i in range(48,58)]
_,cnt,hr = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,cnt,-1,(255,0,0),2)
count=0
model= cv2.ml.KNearest_create()
with np.load('D:\Sriram\pyprograms\sudoku\knn.npz') as data:
	print data.files
	train1=data['train']
	train_labels= data['traindata']
	print train1
	print train_labels
model.train(train1,cv2.ml.ROW_SAMPLE,train_labels)
matrix=np.zeros((9,9),np.uint8)
print matrix
for i in range(len(cnt)):
	area=cv2.contourArea(cnt[i])
	if area>=40000:
		x,y,w,z=hr[0][i]
		if w!=-1:
			a,b,c,d=cv2.boundingRect(cnt[w])
			#print c,d
			errx=100-c
			erry=180-d
			b=b-int(erry/2)
			a=a-int(errx/2)
			c=100
			d=180
			roi=frame[b:b+d,a:a+c]
			cv2.rectangle(img,(a,b),(a+c,b+d),(255,0,0),4)
			q=cv2.resize(img,(400,400))
			cv2.imshow("frame",q)
			roismall=cv2.resize(roi,(10,10))
			roismall=roismall.reshape(1,100)
			roismall1=np.float32(roismall)
			r=model.findNearest(roismall1,1)
			print int(r[1][0])
			count=count+1
			count1=81-count
			row=int(count1/9)
			col=int(count1%9)
			matrix[row][col]=int(r[1][0])
		else:
			count=count+1
			count1=81-count
			row=int(count1/9)
			col=int(count1%9)
			matrix[row][col]=0
print count
#print matrix
def nextcell(matrix,i,j):
	for x in range(i,9):
		for y in range(j,9):
			if matrix[x][y]==0:
				return x,y
	for x in range(0,9):
		for y in range(0,9):
			if matrix[x][y]==0:
				return x,y
	return -1,-1
def checkcell(matrix,i,j,val):
	rowcheck=colcheck=1
	for x in range(9):
		if val==matrix[i][x]:
			rowcheck=0
	if rowcheck==1:
		for x in range(9):
			if val==matrix[x][j]:
				colcheck=0
	if colcheck==1 and rowcheck==1:
		topx=3*(i/3)
		topy=3*(j/3)
		for x in range(topx,topx+3):
			for y in range(topy,topy+3):
				if matrix[x][y]==val:
					return False
		return True
	return False
def sudokusolve(matrix,i=0,j=0):
	i,j=nextcell(matrix,i,j)
	if i==-1:
		return True
	for val in range(1,10):
		if checkcell(matrix,i,j,val):
			matrix[i][j]=val
			if sudokusolve(matrix,i,j):
				return True
			matrix[i][j]=0
	return False

if sudokusolve(matrix,0,0):
	print matrix
	print "Eureka"
else:
	print "Could not solve"
count=0
val=0
for i in range(len(cnt)):
	area=cv2.contourArea(cnt[i])
	if area>=40000:
		x,y,w,z=hr[0][i]
		if w!=-1:
			count=count+1
			continue
		else:
			count=count+1
			count1=81-count
			row=int(count1/9)
			col=int(count1%9)
			val=matrix[row][col]
			s=str(val)
			a,b,c,d=cv2.boundingRect(cnt[i])
			im=cv2.imread(deflt+s+".png",1)
			img[b+10:b+190,a+50:a+150]=im
plt.imshow(img),plt.title("frame")
plt.xticks([]),plt.yticks([])
plt.show()
cv2.destroyAllWindows()





