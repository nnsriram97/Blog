import cv2
import numpy as np
import bluetooth
target_name = "HC-05"
target_address = None
nearby_devices = bluetooth.discover_devices()
print nearby_devices
print("found %d devices" % len(nearby_devices))
for bdaddr in nearby_devices:
    if target_name == bluetooth.lookup_name( bdaddr ):
        target_address = bdaddr
        break

if target_address is not None:
    print "found target bluetooth device with address ", target_address
else:
    print "could not find target bluetooth device nearby"
sersoc=bluetooth.BluetoothSocket(bluetooth.RFCOMM)
port = 1
sersoc.connect((target_address,port))
print "connected"
lor=np.array([160,75,75])
upr=np.array([179,255,255])	
log=np.array([50,75,75])
upg=np.array([70,255,255])
lob=np.array([110,75,75])
upb=np.array([128,255,255])
loy=np.array([24,75,75])
upy=np.array([40,255,255])
lop=np.array([150,75,75])
upp=np.array([162,255,255])
lobl=np.array([0,0,0])
upbl=np.array([0,60,100])
kernel=np.ones((5,5),np.uint8)

## lines calc
frame=cap.read()
frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
## Centre Blue ###

mask=cv2.inRange(img,loy,upy,None)    
mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)	
_,cnt,hie=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
temp=0
p=0
for i in range(len(cnt)):
	if (cv2.contourArea(cnt[i])>temp):
		temp=cv2.contourArea(cnt[i])
		p=i
(xb,yb),rb=cv2.minEnclosingCircle(cnt[p])
xb,yb=int(xb),int(yb)
### Centre Red ###
mask=cv2.inRange(img,lor,upr,None)    
mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)	
_,cnt,hie=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
temp=0
p=0
for i in range(len(cnt)):
	if (cv2.contourArea(cnt[i])>temp):
		temp=cv2.contourArea(cnt[i])
		p=i
(xr,yr),rr=cv2.minEnclosingCircle(cnt[p])
xr,yr=int(xr),int(yr)
### Centre Green ###
mask=cv2.inRange(img,log,upg,None)    
mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)	
_,cnt,hie=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
temp=0
p=0
for i in range(len(cnt)):
	if (cv2.contourArea(cnt[i])>temp):
		temp=cv2.contourArea(cnt[i])
		p=i
(xg,yg),rg=cv2.minEnclosingCircle(cnt[p])
xg,yg=int(xg),int(yg)

### Centre for Black ###
mask=cv2.inRange(img,lobl,upbl,None)    
mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)	
_,cnt,hie=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
temp=1000
p=0
for i in range(len(cnt)):
	if (cv2.contourArea(cnt[i])>temp):
		temp=cv2.contourArea(cnt[i])
		p=i
(xbl,ybl),rbl=cv2.minEnclosingCircle(cnt[p])
xbl,ybl=int(xbl),int(ybl)
### line calculation ###
dest=[]
dest.append((xg,yg),(xr,yr),(xb,yb),(xbl,ybl))

routes=[]
m=(yg-ybl)/(xg-xbl)
c=yg-(m*xg)
routes.append((m,c))
m=(yb-ybl)/(xb-xbl)
c=yb-(m*xb)
routes.append((m,c))
m=(yr-ybl)/(xr-xbl)
c=yr-(m*xr)
routes.append((m,c))
"""
for m,c in routes:
	a=-m
	b=1
	c=-c
	x2=int((b*xbl-(b*ybl*m)-c)/(a*m+b))
	y2=int(((x2-xbl)/m)+ybl)
	dest.append((x2,y2))
	mtemp=(ybl-y2)/(xbl-x2)
	ctemp=y2-(m*x2)
	routes.append((mtemp,ctemp))"""
####
loot=2000
kernel=np.ones((5,5),np.uint8)
Wradius = 0.037;
Blength = 0.106;
Bvelocity = 0.065;
pwm_lower = 45.0;
pwm_upper = 240.0;
omega_lower = 0.0;
omega_upper = 8.35;
kp=2
def detectf(lx,ly):
	ret,frame=cap.read()
	img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	mask=cv2.inRange(img,loy,upy,None)    
	mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)	
	_,cnt,hie=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	temp=1000
	p=0
	x3=y3=x4=y4=0
	for i in range(len(cnt)):
		if (cv2.contourArea(cnt[i])>temp):
			temp=cv2.contourArea(cnt[i])
			p=i		
	if len(cnt)!=0:
		print p
		try:
			moments=cv2.moments(cnt[p])
			x3,y3=int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])	
		except:
			print "error"
	mask=cv2.inRange(img,lop,upp,None)    
	mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)	
	_,cnt,hie=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	temp=0
	p=0
	for i in range(len(cnt)):
		if (cv2.contourArea(cnt[i])>temp):
			temp=cv2.contourArea(cnt[i])
			p=i
	if len(cnt)!=0:
		try:
			moments=cv2.moments(cnt[p])
			x4,y4=int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])
		except:
			print "error"
	dist1=math.sqrt((x3-lx)**2+(y3-ly)**2)
	dist2=math.sqrt((x4-lx)**2+(y4-ly)**2)
	if dist1>=dist2:
		return 0
	else :
		return 1
def movf(lx,ly):
	x3=y3=x4=y4=0
	global Bvelocity
	check=0
	while (check>=0):
		ret,frame=cap.read()
		img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		mask=cv2.inRange(img,loy,upy,None)    
		mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)	
		_,cnt,hie=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		temp=0
		p=0
		x=y=0
		for i in range(len(cnt)):
			if (cv2.contourArea(cnt[i])>temp):
				temp=cv2.contourArea(cnt[i])
				p=i
		if len(cnt)!=0:
			try:
				moments=cv2.moments(cnt[p])
				x3,y3=int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])
			except:
				print "error"
		mask=cv2.inRange(img,lop,upp,None)    
		mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)	
		_,cnt,hie=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		temp=0
		p=0
		for i in range(len(cnt)):
			if (cv2.contourArea(cnt[i])>temp):
				temp=cv2.contourArea(cnt[i])
				p=i
		if len(cnt)!=0:
			try:
				moments=cv2.moments(cnt[p])
				x4,y4=int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])
			except:
				print "error"
		r=gogoal([x4,y4],[x3,y3],[lx,ly])
		if (abs(r*180/3.14)<15):
			Bvelocity=0.405
		else:
			Bvelocity=0.025
		pr=pid(r)
		lp,rp=kin2d(pr)
		lp=int(0.8*lp)
		rp=int(0.8*rp)
		if lp<100:
			lp="0"+str(lp)
		if rp <100:
			rp="0"+str(rp)
		sig=str(lp)+' '+str("000")+' '+str("000")+' '+str(rp)+'\n'
		sersoc.send(sig)
		print sig
		cv2.line(frame,(x3,y3),(lx,ly),(255,0,0),3)
		cv2.circle(frame,(x3,y3),3,(255,0,0),2)
		cv2.imshow("Frame",frame)
		k=cv2.waitKey(10) & 0xff
		if k==27:
			break
		check=(x3-lx)**2+(y3-ly)**2-400
	print x3,y3
	return

def gogoal(p,y,g):
	ordir= [p[0]-y[0],p[1]-y[1]]
	destdir=[y[0]-g[0],y[1]-g[1]]
	dot=ordir[0]*destdir[0]+ordir[1]*destdir[1]
	det=ordir[0]*destdir[1]-ordir[1]*destdir[0]
	return math.atan2(det,dot)

def pid(rad):
	error=rad
	pwm=error*kp
	return pwm

def kin2d(pwm):
	s=2*Bvelocity/Wradius
	d=Blength*pwm/Wradius
	lw=(s+d)/2
	rw=(s-d)/2
	print "lw",lw,"rw",rw
	if lw>omega_upper or rw>omega_upper:
		if lw>omega_upper:
			temp=lw-omega_upper
			lw=omega_upper
			rw=rw-temp
			if rw<omega_lower:
				rw=omega_lower
		else:
			temp=rw-omega_upper
			rw=omega_upper
			lw=lw-temp
			if lw<omega_lower:
				lw=omega_lower
	elif lw<omega_lower or rw<omega_lower:
		if lw<omega_lower:
			temp=omega_lower-lw
			lw=omega_lower
			rw=rw+temp
			if rw>omega_upper:
				rw=omega_upper
		else:
			temp=omega_lower-rw
			rw=omega_lower
			lw=lw+temp
			if lw>omega_upper:
				lw=omega_upper
	#print lw,rw
	lw=int(pwm_lower+(((pwm_upper-pwm_lower)/(omega_upper-omega_lower))*lw))
	rw=int(pwm_lower+(((pwm_upper-pwm_lower)/(omega_upper-omega_lower))*rw))
	return lw,rw

### alogorithm for next loot##
A = 0 #1000
B = 1 #

C = 2 #250
home = 4;
true = 1
false = 0
amountremaining = loot-colloot
# order ABC


def getnewgoal( pl, lootsrem):

    if amountremaining <= 0:
        return home

    if lootsrem == 0:
        return home

    if amount <= 250:
        if pl[C] == 1:
            return C
        elif pl[B] == 1:
            return B
        elif pl[A] == 1:
            return A

    elif amount <= 500:
        if pl[B] == 1:
            return B
        elif pl[C] == 1:
            return C
        elif pl[A] == 1:
            return A
        else:
            return home

    elif amount <= 1000:
        if pl[A] == 1:
            return A
        elif pl[B] == 1:
            return B
        elif pl[C] == 1:
            return C
        else:
            return home

    else:
        if (pl[A] == true) and (pl[B] == true) and (pl[C] == true): #111
            return A

        elif (pl[A] == false) and (pl[B] == true) and (pl[C] == true): #011
            return B

        elif (pl[A] == false) and (pl[B] == false) and (pl[C] == true): #001
            return C

        elif (pl[A] == false) and (pl[B] == false) and (pl[C] == false): #000
            return home

        elif (pl[A] == false) and (pl[B] == true) and (pl[C] == false): #010
            return B

        elif (pl[A] == true) and (pl[B] == false) and (pl[C] == false): # 100
            return A

        elif (pl[A] == true) and (pl[B] == false) and (pl[C] == true): #101
            return A

        elif (pl[A] == true) and (pl[B] == true) and (pl[C] == false): #110
            return A

####

### moving func###
curdest=1
curline=5
colloot=0
## Choose the route ###
while colloot<=loot:
	curdest=0
	if curdest==0:
		path=[1,(2,1),(3,1)]
		localdest={1:(xg,yg),2:(xbl,ybl),3:(xbl,ybl)}
	elif curdest==1:
		path=[(1,2),2,(3,2)]
		localdest={1:(xbl,ybl),2:(xb,yb),3:(xbl,ybl)}
	elif curdest==2:
		path=[(1,3),(2,3),3]
		localdest={1:(xbl,ybl),2:(xbl,ybl),3:(xr,yr)}
	elif curdest=3:
		path=[1,2,3]
		localdest={1:(xbl,ybl),2:(xbl,ybl),3:(xbl,ybl)}
	for i in path[curline-1]:
			for j in i:
				lx,ly=localdest[j]
				## call func which takes care of moving
				#f=detectf(lx,ly)
				movf(lx,ly)
				
				if localdest==curdest:
					sersoc.send("000 000 000 000\n")

cv2.destroyAllWindows()
