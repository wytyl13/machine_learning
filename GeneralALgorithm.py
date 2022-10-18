'''================================================================
*   Copyright (C) 2022 IEucd Inc. All rights reserved.
*   
*   FileName:GeneralALgorithm.py
*   Author:weiyutao
*   CreateTime:2022-03-29
*   Describe:本文件使用python构建大论文中使用的机器学习算法，比如线性回归、非线性回归、逻辑回归、PCA、无监督聚类分析、推荐算法和异常点检测算法
*
================================================================'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio


#the set for pandas.
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide',True)




'''
 This is function PCA,aim to reduce the data dimension and retain the largest amount of information.

 Parameters:
	X:the original data set,the dimansion is m * n
	k:the eigenvalues after reduce the dimension
 return:
	X_restore:the data set after reducing the dimension,the dimension is m * k
'''
def PCA(X,k):

	#to the mean，simplify the math problem
	X_demean = X - np.mean(X,axis = 0)

	#Covaraince matrix calculation:C = X.T @ X / (m - 1)
	C = X_demean.T @ X_demean / len(X)

	#Calculate the eigencalues of the covariance matrix and the corresponding eigenvectors
	#U:eigenvectors,the dimansion is n * n
	#S:eigenvalues,the dimainsion is n * 1
	#notation:the difference between in mathmatical derivation V and the return value U,the dimension of V is k * n,the dimension of U is n * n and it is after the conversion of the V,so we don't need to conversion again at the time of data dimension reduction later.
	#the difference between V and U : the U[:,0:K - 1].T is equal to the mathmatical derivation V what is made of k times (1,n) vector.
	U,S,V = np.linalg.svd(C)
	
	#the dimension of U_k is n * k,the U_k is the orthogonal basis what is the V of the mathmatical derivation about the vector of the plane! 
	U_k = U[:,0:k]

	#at the time of the mathmatical derivation,the expression is X @ V.T,the dimension of X_reduction is m * k,the X_reduction is the data set of reducing dimension.the X_demean is original data set,multiplied the U_k is equals to the projection on the orthogonal of original data set 
	X_reduction = X_demean @ U_k

	#data reduction
	#X_restore = X_reduction.reshape(len(X),k) @ U_k.reshape(k,n) + np.mean(X,axis = 0)

	return X_reduction





#***************************************************************************************************






#the next algorithm is clustering algorithm what is unsupervised learning

#step1 : cluster distribution -> through each sample points to every clustering center distance in order to assign the clustering center for each sample that the smallest two-norm.the cost of this step is minimize the two-norm what is Σ||x - μ||^2，the optimal object of this step is cluster assign.
'''
 the function is cluster distribution

 parameters:
	X:ordinary data set,the dimension is m * n
	centros:the initial clustering center,the type is array 
 return:
	idx:the cluster id of every sample points and the type of idx is array 
'''
def find_centroidx(X,centros):
	idx = []
	for i in range(len(X)):
		#calculate the two-norm,the dimension of the return value dist is 1 * len(centros)
		dist = np.linalg.norm((X[i] - centros),axis = 1)
		
		#get the smallest arg of dist what is the current sample corresponding clustering center arg 
		id_i = np.argmin(dist)
		
		#store the results you find in the list
		idx.append(id_i)
	
	#returns an array rather than the list
	return np.array(idx)


#step2:based on the results of step 1 to adjust the location of each cluster center.the cost of this step is minimize the distance from every clustering center to its corresponding sample points.we need to use the mean of the corresponding sample as the moving position of its clustering center,the optimal object is the location of teh every clustering center.

'''
 the function is to move every clustering center to the bosom of its sample points.

 parameters:
	X:ordinary data set,the dimension of X is m * n
	idx:the results of step 1,the dimension of idx is 1 * m
	K:the len of clustering center

 return:
	centros:coordinates of the cluster center after moving.the type of centors is array,the dimension of centros is m * n
'''
def move_centros(X,idx,K):
	centros = []
	for i in range(K):
		centros_i = np.mean(X[idx == i],axis = 0)
		centros.append(centros_i)
	
	return np.array(centros)

#step3:integrate the first two step to establish the Kmeans algorithm

'''
 this is unsupervised learning about the clustering algorithm that used the kmeans algorithm what is the combination of step1 and step2.

 paramters:
	X:the original data set

 return:
	idx:the latest location of the clustering center
	centros_all:all of the centros that involved the initial clustering center
'''
def runKmeans(X,centros,iters):
	K = len(centros)
	centros_all = []
	centros_all.append(centros)
	centros_i = centros
	for i in range(iters):
		idx = find_centroidx(X,centros_i)
		centros_i = move_centros(X,idx,K)
		centros_all.append(centros_i)
	
	return idx,np.array(centros_all)

#a method to draw the moving trajectory of data set and clustering centers
def plot_data(X,centros_all,idx):
	fig,ax = plt.subplots()

	#c:assign colors according to the clustering center they belong to
	ax.scatter(X[:,0],X[:,1],c = idx,cmap = 'rainbow',label = "data Source")

	#k:color;x:dot display pattern;--:imaginary line
	ax.plot(centros_all[:,:,0],centros_all[:,:,1],'kx--',label = "centrosMobileTrajectory")
	ax.set_xlabel('x1',fontsize = 15)
	ax.set_ylabel('x2',fontsize = 15)
	ax.set_title('dataSource VS centrosMobileTrajectory',fontsize = 20)
	ax.legend()
	plt.show()
	

#random initialization of the cluster center
def init_centros(X,K):
	
	#the dimension of index is 1 * k what is a list about index
	index = np.random.choice(len(X),K)
	
	#return the indexed original set as the initial clustering center,the dimension of return value is k * n
	return X[index]



#***********************************************************************************************
#the next algorithm is recommendation algorithm base on the content,the mathmatical deduction is following:theta1-theta_n_u said each user perference for the movie,the dimension of each theta is n_f * 1 what is a column vector of n_f elements,and the meaning of n_f is the score characteristic number of each movie.the x1-x_n_m is the score characteristic of each number,the number of movies is n_m and the score characteristic number of each movie is equals to the n_f what is the elements number of the theta.matrix theta and matrix x are consist of the theta1-theta_n_u.T and the x1-x_n_m.T.the y is the score about each user of each movie,and the dimension of matrix theta is n_u * n_f,the demension of matrix x is n_m * n_f.so the conclusion of the latest is Y = matrix theta @ matrix x.T.the Y is matrix y what the dimension is ((n_u,n_f) @ (n_f,n_m)) = n_u * n_m,the meaning of Y is the score of each user for each movie.and the core problem is the users do not have any movie score.If it exist,then the user will not have any information we can provide in our conclusion.so the core problem about recommendation algorithm is we should concantrate on those have not any movie score rather than than that have not any user score about the movie.That is we should concentrate on the matrix y or  the Y that its row vector whether all is empty.we should find the solution about it.




#step1:costfunction
'''
 the function is to calculate the cost

 paramrers:
	x:original data set,the movie socre standard.the dimension is n_m * n_f what is the matrix x
	theta:original data set,the user perference for movie,the demension is n_u * n_f,matrix theta
	y:the real score that each user for each movie,the dimension is n_u * n_m.
	r:corresponding y,reaction is whether a user to a particular movie score.the elments is 0 or 1
	n_u:the user number
	n_m:the movie number
	lamda:regularization coefficient

 return:
	cost:loss value
'''
def costFunction(x,theta,y,r,n_u,n_m,n_f,lamda):
	
	#dot product to filter out the missing value what made no grading project set to 0
	cost = 0.5 * np.square((x @ theta.T - y) * r).sum()

	#the collabrative algorith what is combine the optimization of x and theta.The principle of recommendation algorithm based on the content is x and theta as to optimize parameters,respectively!then using the gradient descent algorithm to constantly iteration to find the optimal value.here is the simplift the way what let the two cost functions in one place,the thing we need to do is to minimize the cost function.
	reg1 = 0.5 * lamda * np.square(x).sum()
	reg2 = 0.5 * lamda * np.square(theta).sum()

	return cost + reg1 + reg2



#step2:gradient descent algorithm
def gradientDescent(x,theta,y,r,iters,alpha,lamda):
	costs = []
	for i in range(iters):

		#the dimension of x is n_m * n_f,theta is n_u * n_f,y is n_m * n_u,if we understand the dimension,the following expression we are easy to understand!a partial derivatives for the two optimized parameters!
		x = x - (((x @ theta.T - y) * r) @ theta + lamda * x) * alpha / len(x)
		theta = theta - (((x @ theta.T - y)).T @ x + lamda * theta) * alpha / len(x)

	return x,theta


#step3:the mean normalization,the purpose is to solve the core problem what user did not give any movie score above mentioned.If using mean normalization for ordinary data set,the missing value of rating will be filled of negative average rather than zero.Note we only need the average score for normalization!the dimension of y is n_m * n_u
def normalization(y,r):
	y_mean = (y.sum(axis = 1) / r.sum(axis = 1)).reshape(-1,1)
	y_norm = (y - y_mean) * r

	return y_mean,y_norm


#step4:initialize the parameters about theta and x.
def initialParameters(n_u,n_m,n_f):
	
	#the theta and x is matrix theta and matrix x
	x = np.random.random((n_m,n_f))
	theta = np.random.random((n_u,n_f))
	
	return x,theta



#************************************************************************************************

if __name__ == "__main__":

	'''
	test recommendation algorithm
	path = "d:/网盘下载/作业代码资料/全部作业代码-无答案/ml_totalexerise/exerise8/data/ex8_movies.mat"
	data = sio.loadmat(path)
	y,r = data['Y'],data['R']
	n_m,n_u = y.shape
	my_ratings = np.zeros((n_m,1))
	my_ratings[9] = 5
	my_ratings[66] = 5
	my_ratings[96] = 5
	my_ratings[121] = 5
	my_ratings[148] = 5
	my_ratings[285] = 5
	my_ratings[490] = 5
	my_ratings[599] = 5
	my_ratings[643] = 5
	my_ratings[958] = 5
	my_ratings[1117] = 5
	
	y = np.c_[y,my_ratings]
	r = np.c_[r,my_ratings != 0]
	n_m,n_u = y.shape

	y_mean,y_norm = normalization(y,r)
	n_f = 3
	x,theta = initialParameters(n_u,n_m,3)
	lamda = 10
	alpha = 0.05
	x_fit,theta_fit = gradientDescent(x,theta,y,r,100,alpha,lamda)
	y_pred = x_fit @ theta_fit.T
	y_pred = y_pred[:,-1] + y_mean.flatten()
	
	index = np.argsort(-y_pred)
	index10 = index[:10]
	path_movieids = "d:/网盘下载/作业代码资料/全部作业代码-无答案/ml_totalexerise/exerise8/data/movie_ids.txt"

	movies = []
	with open(path_movieids,'r',encoding = 'latin 1') as f:
		for line in f:
			tokens = line.strip().split(' ')
			movies.append(''.join(tokens[1:]))
	
	for i in range(10):
		print(index10[i],movies[index10[i]],y_pred[index[i]])
	'''




	

	'''test kmeans algorithm
	path = "D:/网盘下载/作业代码资料/全部作业代码-无答案/ML_totalexerise/exerise7/data/ex7data2.mat"
	data = sio.loadmat(path)
	
	X = data["X"]
	for i in range(8):
		initialCentros = init_centros(X,3)
		idx,centros_all = runKmeans(X,initialCentros,100)
		print(idx)	
	'''
	


	'''
	test code
	
	path = "C:/Users/80521/desktop/originalData1.xls"
	
	data = pd.read_excel(path)
	
	X = data.iloc[:,1:-1]
	
	X_reduction = PCA(X,2)
	
	fig,ax = plt.subplots()
	ax.scatter(X_reduction[0],X_reduction[1])
	plt.show()
	
	
	print(X_reduction)
	
	'''
	






















