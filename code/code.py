from skimage import io, color
import numpy as np
import tensorflow as tf
import os
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def build_graph(X): #X=[h,w,3]
	h,w=X.shape[:2]
	F=np.zeros((h,w,29))
	for i in range(h):
		for j in range(w):
			i1,j1=min(max(i,1),h-2),min(max(j,1),w-2)
			F[i][j]=np.append(X[i1-1:i1+2,j1-1:j1+2].flatten(),([i,j]))
	#N[i]=list of 25 indices of its neigbours' coordinates
	#W[i]=list of 25 weights
	N,W=np.zeros((h*w,25)),np.zeros((h*w,25))
	r=max(1,int(h/60))
	for i in range(h):
		for j in range(w):
			f,L=F[i][j],[]
			for x in range(0,h-1,r):
				for y in range(0,w-1,r):
					L.append([x*w+y,np.linalg.norm(f-F[x][y])])
			L=sorted(L,key=lambda x: x[-1])[:25]
			for l in range(20,25):
				L[l][-1]=L[20][-1] #clip smallest weight to 80th percentile
			L2=[(x,np.exp(-1*wt/L[18][-1])) for x,wt in L]
			s=sum([x[-1] for x in L2])
			L2=[(x,wt/s) for x,wt in L2]
			N[i*w+j]=np.array([x[0] for x in L2])
			W[i*w+j]=np.array([x[1] for x in L2])
	return [N,W]

def read_image(S,fpath):
	rgb = io.imread(fpath)
	lab = color.rgb2lab(rgb)
	lab=lab[:256,:256,:]
	r=int(256/S)
	return lab[0:256:r,0:256:r,:]

def save_image(S):
	for i in range(3):
		print (np.amax(S[:,:,i]))
		print (np.amin(S[:,:,i]))
	S=color.lab2rgb(S)
	io.imsave("out.jpg",S)

def nl_image(M,L):
	#M is [h*w,3], N is [h*w,d], W is [h*w,d]
	M1=[]
	N, W = tf.constant(L[0],dtype=tf.int32), tf.constant(L[1],dtype=tf.float32)
	for i in range(M.shape[0]):
		indices=tf.gather(N,i)
		patch=tf.gather(M,indices)
		weights=tf.tile(tf.expand_dims(tf.gather(W,i),1),[1,3])
		M1.append(tf.multiply(patch, weights) ) 
	return tf.stack(M1)

def f1(X,reuse):
	f1X = tf.layers.conv2d(
		tf.expand_dims(X,0),50,(5,5),padding='same',activation=tf.nn.relu,reuse=reuse,
		name='conv1',trainable='false',bias_initializer=tf.truncated_normal_initializer())
	f1X=tf.reshape(f1X,(f1X.shape[1]*f1X.shape[2],50))	
	return f1X

def f2(Xp,G,reuse):
	X=tf.reshape(Xp,(Xp.shape[0]*Xp.shape[1],Xp.shape[2]))
	Xnl=nl_image(X,G)
	Xnl=tf.reshape(Xnl,(Xnl.shape[0]*Xnl.shape[1],Xnl.shape[2]))
	Xnl=tf.expand_dims(Xnl,1)
	f2X= tf.layers.conv2d(
		tf.expand_dims(Xnl,0),50,(25,1),padding='valid',activation=tf.nn.relu,reuse=reuse,
		name='conv2',strides=(25,1),trainable='false',bias_initializer=tf.truncated_normal_initializer())
	f2X=tf.reshape(f2X,(f2X.shape[0]*f2X.shape[1]*f2X.shape[2],f2X.shape[3]))
	return f2X

###################################################

for size in [32,64,128,256]:
	start = time.clock()

	Xt=read_image(size,"../images/1-content.jpg")
	Gt=build_graph(Xt)
	Xs=read_image(size,"../images/1-style.jpg")
	Gs=build_graph(Xs)
	print ("size="+str(size))


	if size==32:
		X=np.zeros((32,32,3))
		X[:,:,0]=np.random.uniform(1,99,size=(size,size))
		X[:,:,1]=np.random.uniform(-35,57,size=(size,size))
		X[:,:,2]=np.random.uniform(-16,33,size=(size,size))
		save_image(X)
		X=tf.Variable(tf.constant(X,dtype=tf.float32), name="X"+str(size))
		reuse=None
	else:
		X=tf.Variable(tf.image.resize_images(X,tf.constant(np.array([size,size]),dtype=tf.int32)),name="X"+str(size))
		reuse=True

	f1X = f1(X,reuse)
	T1=tf.matmul(tf.transpose(f1X),f1X)

	f1Xs = f1(tf.constant(Xs,dtype=tf.float32),True)
	G1=tf.matmul(tf.transpose(f1Xs),f1Xs)
	
	loss1=tf.norm(tf.reshape(tf.subtract(T1,G1),[T1.shape[0]*T1.shape[1]]),ord=2)
	gamma1=tf.divide(1,tf.reduce_max(tf.abs(tf.gradients(loss1,[X]))))

	if size==32:
		sess = tf.Session()
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	gamma1=sess.run(gamma1)
	print ("gamma1 is: "+str(gamma1))
	loss1=tf.multiply(gamma1,loss1)

	f2X=f2(X,Gt,reuse)
	T1=tf.matmul(tf.transpose(f2X),f2X)

	f2Xs=f2(tf.constant(Xs,dtype=tf.float32),Gs,True)
	G1=tf.matmul(tf.transpose(f2Xs),f2Xs)

	loss2=tf.norm(tf.reshape(tf.subtract(T1,G1),[T1.shape[0]*T1.shape[1]]),ord=2)
	gamma2=tf.divide(1,tf.reduce_max(tf.abs(tf.gradients(loss2,[X]))))
	
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	gamma2=sess.run(gamma2)
	print ("gamma2 is: "+str(gamma2))
	loss2=tf.multiply(gamma2,loss2)

	loss3=tf.multiply(0.01*size*size,tf.image.total_variation(X))

	loss=tf.add(loss1,tf.add(loss2,loss3))

	lb=[ [ [0,-86,-107] for _ in range(size)] for _ in range(size)]
	ub=[ [ [100,98,94] for _ in range(size)] for _ in range(size)]
	optimizer = tf.contrib.opt.ScipyOptimizerInterface(
		loss, var_to_bounds={X: (lb, ub)})

	print ("----")
	print (sess.run(loss))
	optimizer.minimize(sess)
	print (sess.run(loss))
	print ("----")
	Xf=sess.run(X).astype(np.float64)
	save_image(Xf)

	print ("Time taken for "+str(size)+"is: "+str(time.clock()-start))
