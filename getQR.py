import numpy as np
'''
def getQR(inQmat,intaus):
	r=np.zeros(inQmat.shape)
	for rowi in range(inQmat.shape[1]):
		for coli in range(inQmat.shape[1]):
			if coli>=rowi:
				r[rowi,coli]=inQmat[rowi,coli]
	q=np.eye(inQmat.shape[0])
	for coli in range(inQmat.shape[1]):
		v=np.zeros(1,inQmat.shape[0])
		for id in range(inQmat.shape[0]):
			if id<coli:
				v[1,coli]=0
			elif id==coli:
				v[1,coli]=1
			else:
				v[1,coli]=inQmat[coli,id]
		
		h=np.eye(inQmat.shape[0])-intaus[coli]*v*np.transpose(v)
		q=q*h
	
	return q,r
'''
	
def getNumpyQR(inx,order):
	rows,=inx.shape
	cols=order
	inA=np.zeros((rows,cols))
	for rowi in range(len(inx)):
		for coli in range(cols):
			if coli==0:
				inA[rowi,coli]=1.0
			else:
				inA[rowi,coli]=inA[rowi,coli-1]*inx[rowi]
	q,r=np.linalg.qr(inA)
	return q,r