import math
import numpy as np


####################### BRANIN #################################
def branin(x):
	assert len(x)==2, 'defined just on 2-dimensions'
	return (x[1] - (5.1 / (4 * math.pi * math.pi)) *x[0]*x[0] + (5 / (math.pi)) *x[0] -6) ** 2 + 10*(1- (1 / (8 * math.pi))) * np.cos(x[0]) + 10

def meno_branin(x):
	f = branin(x)
	return -f
#################################################################


######################### ROSENBROCK ############################
def rosenbrock(x):
	assert len(x)>1, 'not defined on 1 dimension'
	if len(x) == 2:
	  return ((1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)
	else: 
	  f = 0
	  for i in range(len(x) - 1):
	    f += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
	  return f

def meno_rosenbrock(x):
	f = rosenbrock(x)
	return -f
################################################################




########################## HARTMANN6 ###########################
def hartmann6(x):
	assert len(x)==6, 'wrong input dimensions'
	alpha =  np.array([1.0, 1.2, 3.0, 3.2])
	A = np.matrix('10 3 17 3.5 1.7 8;\
					.05 10 17 .1 8 14;\
					3 3.5 1.7 10 17 8;\
				   17 8 .05 10 .1 14')
	P = np.matrix('1321 1696 5569 124 8283 5886;\
						2329 4135 8307 3736 1004 9991;\
						2348 1451 3522 2883 3047 6650;\
						4047 8828 8732 5743 1091 381')

	P = 1e-04*P
	f = 0
	arg = 0
	for i in range(4):
		arg = 0
		for j in range(6):
			arg += -A[i,j]*(x[j] - P[i,j])**2
		f += alpha[i]*np.exp(arg)
	return -f

def meno_hartmann6(x):
	f = hartmann6(x)
	return -f
#a = np.array([0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573])
#########################################################################




########################## HARTMANN3 ###########################
def hartmann3(x):
	assert len(x)==3, 'wrong input dimensions'
	alpha =  np.array([1.0, 1.2, 3.0, 3.2])
	A = np.matrix('3. 10 30;\
					.1 10 35;\
					3. 10 30;\
				   0.1 10 35')
	P = np.matrix('3689 1170 2673;\
						4699 4387 7470;\
						1091 8732 5547;\
						381 5743 8828')

	P = 1e-04*P
	f = 0
	arg = 0
	for i in range(4):
		arg = 0
		for j in range(3):
			arg += -A[i,j]*(x[j] - P[i,j])**2
		f += alpha[i]*np.exp(arg)
	return -f

def meno_hartmann3(x):
	f = hartmann3(x)
	return -f
#########################################################################



########################### ACKLEY ######################################
def ackley(x):
	assert len(x)>1, 'not defined on 1 dimension'
	n = len(x)
	a = 20
	b = .2
	c = 2*math.pi
	sum1 = 0 
	sum2 = 0
	f = 0
	for i in range(n):
		sum1 += (x[i]**2)/n
		sum2 += (np.cos(c*x[i]))/n
	f = -a*np.exp(-b*sum1) - np.exp(sum2) + a + np.exp(1)
	return f

def meno_ackley(x):
	f = ackley(x)
	return -f
##########################################################################