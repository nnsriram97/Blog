from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
deflt="D:\Sriram\pyprograms\sudoku\\"
with np.load('D:\Sriram\pyprograms\sudoku\knnvid1.npz') as data:
	print data.files
	train=data['train']
	train_labels= data['traindata']
	print train
	print train_labels
model= cv2.ml.KNearest_create()
model.train(train,cv2.ml.ROW_SAMPLE,train_labels)
cap=cv2.VideoCapture(0)
ret,img=cap.read()
while(1):
	while(1):
		ret,img=cap.read()
		key=cv2.waitKey(10) & 0xff
		if key==51:
			break
		img=cv2.GaussianBlur(img,(5,5),0)
		frame=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#frame=cv2.Laplacian(frame,cv2.CV_32F)
		#frame=cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
		kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
		close=cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel)
		#cv2.imshow("close",close)
		div=np.float32(frame)/(close)
		res=np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
		#print div
		#print frame
		res2=cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
		thresh=cv2.adaptiveThreshold(res,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
		#cv2.imshow("thresh",thresh)
		_,cnt,hr = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		samples=np.empty((0,100))
		responses=[]
		keys = [i for i in range(48,58)]
		area=0
		temparea=0
		p=0
		bestapprox=None
		for i in range(len(cnt)):
			temparea=cv2.contourArea(cnt[i])
			if temparea>1000:
				peri=cv2.arcLength(cnt[i],True)
				epsilon=0.05*peri
				approx=cv2.approxPolyDP(cnt[i],epsilon,True)
				if len(approx)==4 and temparea>area:
					area=temparea
					bestapprox=approx
					p=i
		box=cnt[p]
		img=cv2.polylines(img,[bestapprox],True,(0,255,0),3)
		cv2.drawContours(img,cnt,p,(255,0,0),2)
		cv2.imshow("frame",img)
		(x,y)=thresh.shape
		mask=np.zeros((x,y),np.uint8)
		mask=cv2.drawContours(mask,cnt,p,255,-1)
		mask=cv2.drawContours(mask,cnt,p,0,2)
		masked=cv2.bitwise_and(mask,res)
		#cv2.imshow("masked",masked)
		#cv2.imshow("img",newimg)
		#cv2.waitKey(0)
		kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))
		dx = cv2.Sobel(masked,cv2.CV_16S,1,0)
		dx = cv2.convertScaleAbs(dx)
		cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
		ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#cv2.imshow("dx",close)
		close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
		_,contour, hr = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contour:
			x,y,w,h = cv2.boundingRect(cnt)
			if h/w > 5:
				cv2.drawContours(close,[cnt],0,255,-1)
			else:
				cv2.drawContours(close,[cnt],0,0,-1)
		close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
		closex = close.copy()
		#cv2.imshow("closex",closex)
		kernely=cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
		dy=cv2.Sobel(masked,cv2.CV_16S,0,2)
		dy=cv2.convertScaleAbs(dy)
		cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
		ret,close=cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		close=cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely,iterations=1)
		_,cnt, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		for i in range(len(cnt)):
			x,y,w,h = cv2.boundingRect(cnt[i])
			if w/h > 5:
				cv2.drawContours(close,cnt,i,255,-1)
			else:
				cv2.drawContours(close,cnt,i,0,-1)
		close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
		#cv2.imshow("close",close)
		closey=close
		#cv2.imshow("closey",closey)
		grid=cv2.bitwise_and(closex,closey)
		cv2.imshow("grid",grid)
		_,contour, hier = cv2.findContours(grid,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		cent=[]
		try:
			for cnt in contour:
				mom=cv2.moments(cnt)
				(x,y)=int(mom['m10']/mom['m00']),int(mom['m01']/mom['m00'])
				cent.append((x,y))
				cv2.circle(img,(x,y),4,(0,255,0),-1)
		except :
			continue
		print len(cent)
		if len(cent)>=50 and len(cent)<100 and len(bestapprox)==4:
			break
		elif len(cent)==100:
			break
		elif len(cent)!=100:
			continue
	if len(cent)==100:
		ce=np.array(cent,np.uint32)
		ce=ce.reshape((100,2))
		ce2=ce[np.argsort(ce[:,1])]
		#print ce2	
		b = np.vstack([ce2[i*10:(i+1)*10][np.argsort(ce2[i*10:(i+1)*10,0])] for i in xrange(10)])
		#print b
		points = b.reshape((10,10,2))
		print points
		output=np.zeros((450,450,3),np.uint8)
		for i in xrange(3):
			for j in range(3):
				partimg=np.array([points[i*3,j*3,:],points[(i)*3,(j+1)*3,:],points[(i+1)*3,(j+1)*3,:],points[(i+1)*3,j*3,:]],np.float32)
				print partimg
				dest=np.array([[j*150,i*150],[(j+1)*150,(i)*150],[(j+1)*150,(i+1)*150],[(j)*150,(i+1)*150]],np.float32)
				gpres=cv2.getPerspectiveTransform(partimg,dest)
				warp=cv2.warpPerspective(res2,gpres,(450,450))
				output[i*150:(i+1)*150,j*150:(j+1)*150]=warp[i*150:(i+1)*150,j*150:(j+1)*150]
	else:
		dest=np.array(bestapprox,np.float32)
		nw=np.array([[0,0],[0,450],[450,450],[450,0]],np.float32)
		M=cv2.getPerspectiveTransform(dest,nw)
		output=cv2.warpPerspective(res2,M,(450,450))
	cv2.rectangle(output,(0,0),(450,450),0,1)
	cv2.imshow("oUtput",output)		
	f1=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
	f1g=cv2.adaptiveThreshold(f1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	f1g=cv2.morphologyEx(f1g,cv2.MORPH_OPEN,(5,5))
	_,cnt,hr = cv2.findContours(f1g,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.imshow("img",f1g)
	key=cv2.waitKey(0) & 0xff
	if key==51:
		break
	elif key==50:
		cv2.destroyAllWindows()
		sys.exit()
	else:
		continue
#sys.exit()
count=0
matrix=np.zeros((9,9),np.uint8)
#cv2.drawContours(output,cnt,-1,(255,0,0),2)
#cv2.imshow("output",output)
print matrix
count=0
for i in xrange(9):
	for j in range(9):
		tst=f1g[(i*50)+5:((i+1)*50)-5,(j*50)+5:((j+1)*50)-5]
		roismall=cv2.resize(tst,(10,10))
		roismall=roismall.reshape(1,100)
		roismall=np.float32(roismall)
		r=model.findNearest(roismall,2)
		print r[1][0]
		count=count+1
		matrix[i][j]=int(r[1][0])
print matrix
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
for i in xrange(9):
	for j in range(9):
		val=matrix[i][j]
		s=str(val)
		im=cv2.imread(deflt+s+".png",1)
		im=cv2.resize(im,(40,40))
		output[(i*50)+5:((i+1)*50)-5,(j*50)+5:((j+1)*50)-5]=im[:,:]
plt.imshow(output),plt.title("frame")
plt.xticks([]),plt.yticks([])
plt.show()
cv2.destroyAllWindows()





