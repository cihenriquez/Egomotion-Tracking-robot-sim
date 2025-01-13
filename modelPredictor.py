###########################
# Clase para tratar con identificacion de modelos
# Ultima actualizacion: 12-03-2020
# Martin Calvo
###########################

import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import eig
from numpy.linalg import lstsq

from scipy.integrate import solve_ivp
from functools import partial  # https://docs.python.org/3.6/library/functools.html
                               # required to pass additional parameters to solve_ivp

class Stimator:
        ## Inicializacion de la clase
        def __init__(self, n_inputs, n_outputs, n_time, n_saveAns = 20, minLambda = 0.1, buff = False, buffSize = 4,bigVal = 1e15):
                self.y_prev = np.zeros((n_time,n_outputs))
                self.u_prev = np.zeros((n_time,n_inputs))
                self.minLambda = minLambda
                self.nSamples = n_time
                self.count = 0
                self.full = False
                self.bigVal = bigVal
                self.prev_ans = np.zeros((n_inputs,n_outputs))[0]
                self.save_Ans = np.zeros((n_saveAns, n_outputs))
                self.buffU = np.zeros((buffSize,n_inputs))
                self.buffY = np.zeros((buffSize,n_outputs))
                self.useBuff = buff
                self.W = np.eye(n_time)
                self.Z = np.zeros((n_time,n_outputs))

                self.bestError = -1
                self.bestFit = []
                self.bestVar = 9e20
                self.meanAns = []
                self.ansRMSE = -1

                self.mseErr = -1
                self.bestMSE = []

                self.minVar = -1
                self.myAns = []
                self.myMSE = -1
                self.trueVal = 0

                self.auxIndex = -1

                self.xdFun = None

                self.paramP = []
                self.stateP = []

        ## Se agregan los datos
        def appendData(self,d_in,d_out):
                if(not self.useBuff):
                        newY = d_out
                        newU = d_in
                else:
                        self.buffY = np.roll(self.buffY,1,axis = 0)
                        self.buffY[0,:] = d_out
                        newY = np.mean(self.buffY, axis = 0)
                        self.buffU = np.roll(self.buffU,1, axis = 0)
                        self.buffU[0,:] = d_in
                        newU = np.mean(self.buffU, axis = 0)
                self.y_prev = np.roll(self.y_prev,1,axis = 0)
                self.y_prev[0,:] = newY
                self.u_prev = np.roll(self.u_prev,1, axis = 0)
                self.u_prev[0,:] = newU
                self.count += 1
                self.full = self.count > (self.nSamples + self.useBuff*self.buffY.shape[0])

        ## Estimacion deterministica del tipo Y*A = U
        def stimatePseudoInv(self):
                Y = self.y_prev
                U = self.u_prev
                Yt = Y.transpose()
                YtY = np.matmul(Yt,Y)
                try:
                        InYtY = self.myInverse(YtY)
                        pseudoInv = np.matmul(InYtY,Yt)
                        self.prev_ans = np.matmul(pseudoInv,U)
                except:
                        #print("No pude invertir")
                        pass
                finally:
                        self.updateMeasure()
                        return self.prev_ans

        ## Calculo de varianza de los parametros
        def varTheta(self):
                Y = self.y_prev
                U = self.u_prev
                Yt = Y.transpose()
                YtY = np.matmul(Yt,Y)
                InYtY = self.myInverse(YtY)
                Ut = U.transpose()
                if(self.full):
                        return 0.5*(Ut.dot(U) - Ut.dot(Y).dot(inv(YtY)).dot(Yt).dot(U))
                else:
                        return [-1]
                #return inv(YtY)

        ## Misma inversion que la de arriba, pero usando lo de numpy
        def numpyLstSqr(self):
                try:
                        aux, res, rank, vals = lstsq(self.y_prev, self.u_prev, rcond=None)
                except:
                        pass
                finally:
                        self.prev_ans = aux.ravel()
                        self.updateMeasure()
                        return self.prev_ans, vals

        ## Chequeo de valores propios
        def eigenvalues(self):
                try:
                        Y = self.y_prev
                        aux = np.matmul(np.transpose(Y),Y)
                        vals, j = eig(aux)
                except:
                        pass
                finally:
                        return vals

        ## Inversa con matriz no invertible
        def myInverse(self,matrix):
                # https://stackoverflow.com/questions/7164397/find-the-min-max-excluding-zeros-in-a-numpy-array-or-a-tuple-in-python
                E,V = eig(matrix)
                if(0 in E): # No es invertible!
                        minval = np.min(matrix[np.nonzero(matrix)])
                        matrix += minval*self.minLambda*np.eye(matrix.shape[0])
                return inv(matrix)

        ## Media movil del estado esperado
        def updateMeasure(self,who=0):
                self.save_Ans = np.roll(self.save_Ans,1,axis = 0)
                if(type(self.prev_ans) == np.ndarray):
                        self.save_Ans[0,:] = self.prev_ans.ravel()
                else:
                        self.save_Ans[0,:] = self.prev_ans
                if(self.full):
                        if (self.minVar < 0 or self.minVar > self.getVariance()[0]):
                                self.minVar = self.getVariance()[0]
                                self.myAns = self.getAverageMeasure()
                                self.myMSE = self.getMSE(self.trueVal)

        def getVariance(self):
                if(self.full):
                        return np.var(self.save_Ans,axis = 0)
                else:
                        return np.ones(self.prev_ans.size)*-1

        def getAverageMeasure(self):
                return np.mean(self.save_Ans, axis=0)

        def fullInput(self):
                return not (0 in self.u_prev)

        def checkConstantTorque(self,tol = 0.005,which = 0):
                auxVal = self.u_prev[which,0]
                auxMat = abs(self.u_prev[which,:] - auxVal) < tol
                return np.all(auxMat)

        def getVariability(self):
                return np.var(self.save_Ans, axis = 0)

        ## Dado ciertos parametros, obtiene el error en la estimacion
        def checkParam(self,params,who=1,useMean=False):
                if(len(params) > 0):
                        if useMean:
                                model = self.getAverageMeasure()
                        else:
                                model = np.array(params)
                        if (self.full):
                                error =  np.sum((self.y_prev.dot(model) - self.u_prev.transpose())**2)
                        else:
                                error = -1
                        if((error < self.bestError and error > 0 and self.bestError > 0) or self.bestError < 0):
                                self.bestError = error
                                self.bestFit = params
                                self.errors = self.u_prev.transpose() - self.y_prev.dot(model)
                                V = 0.5*(self.errors**2).sum()
                                self.s2 = 2*V/(self.nSamples - 2)
                                self.cov = self.s2*np.linalg.pinv(self.y_prev.T.dot(self.y_prev))
                                self.phitphi = np.linalg.pinv(self.y_prev.T.dot(self.y_prev))
                                self.eigLS = np.linalg.eig(self.y_prev.T.dot(self.y_prev))
                                #self.bestVar = np.std(self.u_prev.transpose() - self.y_prev.dot(model))
                                #self.bestVar = self.varTheta()
                                self.meanAns = self.getAverageMeasure()
                                self.ansRMSE = np.sqrt(self.getMSE(self.trueVal))
                else:
                        #print(self.bestVar,self.kP[0,0])
                        if(self.bestVar > self.kP[who,who]):
                                self.bestVar = self.kP[who,who]
                                self.bestFit = np.array([self.kX[who],self.kX[who+1]])
                        error = -1
                return error

        def getBestFit(self):
                return self.bestFit

        def getBestError(self):
                return self.bestError/(self.nSamples)

        def getBestVar(self):
                return self.cov

        def getErrors(self):
                return self.errors

        def getMSE(self,trueVal):
                return np.mean((self.save_Ans - trueVal)**2, axis=0)

        def checkMSE(self,trueVal,who=0):
                newMSE = self.getMSE(trueVal)
                if(self.mseErr < 0 or newMSE[who] < self.mseErr):
                        self.mseErr = newMSE[who]
                        self.bestMSE = newMSE
                return self.bestMSE

        def getBestRMSE(self):
                return np.sqrt(self.bestMSE)

        ## Funciones pensadas para RLS
        def setInitialParams(self, S0, thet0):
                self.S0 = np.copy(np.array(S0))
                self.prevP = np.array(S0)
                self.prevTheta = np.array(thet0)

        def setForget(self,mu):
                self.mu = mu

        def setTimes(self, outTime, inTime):
                self.prevIn = inTime
                self.prevOut = outTime

        def tuneRLS(self,tuneVal):
                self.tuneTh = tuneVal

        def RLSStimate(self):
                phi = self.y_prev[0,:]
                error = self.u_prev[0,:] - phi.T.dot(self.prevTheta)
                gain = self.prevP.dot(phi)/(self.mu + np.array(phi.T.dot(self.prevP)).dot(phi))
                self.prevP = (self.prevP - np.array(gain.dot(phi.T)).dot(self.prevP))/self.mu
                newTheta = self.prevTheta + gain*error
                self.prevTheta = self.tuneTh*self.prevTheta + (1-self.tuneTh)*newTheta
                self.prev_ans = self.prevTheta
                self.updateMeasure()
                return self.prevTheta

        def getS(self):
                return self.prevP

        def resetR(self):
                self.prevP += self.S0

        ## Funciones pensadas para WLS
        def setWeight(self,W):
                self.W = W

        def appendWeight(self,weight):
                self.W = np.roll(self.W,1,axis=(0,1))
                self.W[0,0] = weight

        def stimateWeightedPseudoInv(self):
                Y = self.y_prev
                U = self.W.dot(self.u_prev)
                Yt = Y.transpose()
                YtY = np.matmul(Yt,self.W.dot(Y))
                try:
                        InYtY = self.myInverse(YtY)
                        pseudoInv = np.matmul(InYtY,Yt)
                        self.prev_ans = np.matmul(pseudoInv,U).ravel().tolist()
                except:
                        #print("No pude invertir")
                        pass
                finally:
                        self.updateMeasure()
                        return self.prev_ans

        ## Funciones pensadas para IV
        def setZsize(self,zSize):
                self.Z = np.zeros((self.nSamples,zSize))

        def appendInstrument(self,z):
                self.Z = np.roll(self.Z,1,axis=0)
                self.Z[0,:] = z

        def stimateSimlpeIV(self):
                Y = self.Z.T.dot(self.y_prev)
                U = self.Z.T.dot(self.u_prev)
                try:
                        InY = self.myInverse(Y)
                        self.prev_ans = np.matmul(InY,U).ravel()
                except:
                        #print("No pude invertir")
                        pass
                finally:
                        self.updateMeasure()
                        return self.prev_ans

        ## WIV
        def stimateWeightedIV(self):
                Y = self.Z.dot(self.y_prev)
                U = self.W.dot(self.Z.dot(self.u_prev))
                try:
                        InY = self.myInverse(Y)
                        self.prev_ans = np.matmul(InY,U)
                except:
                        #print("No pude invertir")
                        pass
                finally:
                        self.updateMeasure()
                        return self.prev_ans
        ## EIV
        def stimateFullIV(self):
                Y = self.Z.T.dot(self.y_prev)
                U = self.Z.T.dot(self.u_prev)
                Yt = Y.transpose()
                YtY = np.matmul(Yt,Y)
                try:
                        InYtY = self.myInverse(YtY)
                        pseudoInv = np.matmul(InYtY,Yt)
                        self.prev_ans = np.matmul(pseudoInv,U).ravel()
                except:
                        #print("No pude invertir")
                        pass
                finally:
                        self.updateMeasure()
                        return self.prev_ans
        ## EWIV
        def stimateFullWeightedIV(self):
                Y = self.Z.dot(self.y_prev)
                U = self.Z.dot(self.u_prev)
                Yt = Y.transpose()
                YtY = np.matmul(Yt,Y)
                try:
                        InYtY = self.myInverse(YtY)
                        pseudoInv = np.matmul(InYtY,Yt)
                        self.prev_ans = np.matmul(pseudoInv,U)
                except:
                        #print("No pude invertir")
                        pass
                finally:
                        self.updateMeasure()
                        return self.prev_ans

        ## TLS tomado de https://towardsdatascience.com/total-least-squares-in-comparison-with-ols-and-odr-f050ffc1a86a
        def tls(self):
                X = self.y_prev
                y = self.u_prev.T

                if X.ndim == 1: 
                    n = 1 # the number of variable of X
                    X = X.reshape(len(X),1)
                else:
                    n = np.array(X).shape[1] 
                
                Z = np.vstack((X.T,y)).T
                U, s, Vt = la.svd(Z, full_matrices=True)
                
                V = Vt.T
                Vxy = V[:n,n:]
                Vyy = V[n:,n:]
                a_tls = - Vxy  / Vyy # total least squares soln
                
                Xtyt = - Z.dot(V[:,n:]).dot(V[:,n:].T)
                Xt = Xtyt[:,:n] # X error
                y_tls = (X+Xt).dot(a_tls)
                fro_norm = la.norm(Xtyt, 'fro')
                
                self.prev_ans = a_tls.ravel()
                self.updateMeasure()

                return y_tls, X+Xt, a_tls.ravel(), fro_norm

        def wls(self):
                # Ordinary Least Squares solution (LS) x = (A^T*A)^{-1}*A^Ty
                y = self.u_prev
                A = self.y_prev
                xx2, residuals_L2_norm2, rank_A, eig_A  = np.linalg.lstsq(A,y)
                residuals = A.dot(xx2)-y
                
                # 'Naive' FGLS # Feasible Generalized Least Squares --- (more general version of weighted least squares WLS)
                # https://en.wikipedia.org/wiki/Generalized_least_squares#Feasible_generalized_least_squares
                # https://en.wikipedia.org/wiki/Weighted_least_squares
                # The residuals covariance matrix is here built naively assuming that sigma_i, the standard deviation of residual
                # r_i is the same residual r_i.
                residuals = 1/residuals
                sigma_residuals = np.abs(residuals)[:,0]
                Omega = np.diag(sigma_residuals**2.0)
                self.setWeight(Omega)
                return self.stimateWeightedPseudoInv()
                #Omega_inv = np.linalg.pinv(Omega)
                #AT_Omega_inv_A = (A.T).dot(Omega_inv.dot(A))
                #xx3_fgls = np.linalg.pinv(AT_Omega_inv_A).dot((A.T).dot(Omega_inv.dot(y)))
                #return xx3_fgls.ravel()

        def wls2(self):
                # Ordinary Least Squares solution (LS) x = (A^T*A)^{-1}*A^Ty
                y = self.u_prev
                A = self.y_prev
                xx2, residuals_L2_norm2, rank_A, eig_A  = np.linalg.lstsq(A,y)
                residuals = A.dot(xx2)-y
                
                # 'Naive' FGLS # Feasible Generalized Least Squares --- (more general version of weighted least squares WLS)
                # https://en.wikipedia.org/wiki/Generalized_least_squares#Feasible_generalized_least_squares
                # https://en.wikipedia.org/wiki/Weighted_least_squares
                # The residuals covariance matrix is here built naively assuming that sigma_i, the standard deviation of residual
                # r_i is the same residual r_i.
                residuals = 1/residuals
                sigma_residuals = np.abs(residuals)[:,0]
                Omega = np.diag(sigma_residuals**2.0)
                #self.setWeight(Omega)
                #return self.stimateWeightedPseudoInv()
                #Omega_inv = self.myInverse(Omega)
                AT_Omega_A = (A.T).dot(Omega.dot(A))
                xx3_fgls = np.linalg.pinv(AT_Omega_A).dot((A.T).dot(Omega.dot(y)))

                self.prev_ans = xx3_fgls.ravel()
                if not (np.isnan(self.prev_ans[0])):
                        self.updateMeasure()

                return xx3_fgls.ravel()

        def getCorr(self,u_who = 0, y_who = 0):
                x = self.u_prev[:,0]
                y = self.y_prev[:,0]
                rho = np.corrcoef(x,y)
                l = (1/2)*np.log(1-rho[1,0]**2)
                return l

        def getCorr2(self,u_who = 0, y_who = 0):
                x = self.u_prev[:,u_who]
                y = self.y_prev[:,y_who]
                rho = np.corrcoef(x,y)
                l = -(1/2)*np.log(1-rho[1,0])
                return l

        def setQ(self,Q):
                self.Q = Q

        def setR(self,R):
                self.R = R

        def setInit(self,x0,P0):
                self.kP = P0
                self.kX = x0
                self.dT = 0.01

        def setModel(self,A,B,H):
                self.kA = A
                self.kB = B
                self.kH = H

        def kalmanPredict(self):
                predX = self.kA@self.kX + np.reshape(self.kB@self.u_prev[0,:],self.kB.shape)
                predP = (self.kA@self.kP)@self.kA.T + self.Q

                kV = self.y_prev[0,:] - self.kH@predX
                kS = (self.kH@predP)@self.kH.T + self.R
                kG = (predP@self.kH.T)@(np.linalg.pinv(kS))
                self.kX = predX + kG@kV
                self.kP = predP - (kG@self.kH)@predP

                return self.kX

        def setLin(self,xOp, uOp, yOp):
                self.opX = xOp
                self.opU = uOp
                self.opY = yOp

        def eKalmanPred(self):
                dU = self.u_prev[0,:]# - self.opU
                dX = self.kX# - self.opX
                dY = self.y_prev[0,:]# - self.opY

                predX = self.calculateXd()
                #predX = self.kA@dX + np.reshape(self.kB@dU,self.kB.shape)
                predP = (self.kA@self.kP)@self.kA.T + self.Q

                #kV = dY - self.kH@predX
                kV = dY - self.calculateY()
                kS = (self.kH@predP)@self.kH.T + self.R
                kG = (predP@self.kH.T)@(np.linalg.pinv(kS))
                
                self.kX = predX + kG@kV# + self.opX
                self.kP = predP - (kG@self.kH)@predP
                self.prev_ans = self.kX[1:]
                self.updateMeasure()
                return self.kX

        def setXdFun(self,fun):
                self.xdFun = fun

        def calculateXd(self):
                return self.xdFun(self.kX,self.u_prev[0,:],self.dT)

        def setYFun(self,fun):
                self.yFun = fun

        def calculateY(self):
                return self.yFun(self.kX,self.u_prev[0,:])

        def setDt(self,dT):
                self.dT = dT

        def myModel(self):
                x1 = self.kX[0] + self.kX[1]*self.kX[2]
                x2 = 0.85*self.kX[1] + self.u_prev[0,0]
                x3 = self.kX[2]
                return np.array([x1,x2,x3])

        def modelo_robot_movil(self, t, x, u, p):
            FL = u[0]
            FW = u[1]

            rad = 0.15
            W = 0.4

            m = 1.0/(rad*p[0])
            c = p[1]/(rad*p[0])
            J = (1.0/p[2])*W/rad/2.0
            b = p[3]*((1.0/p[2])*W/rad/2.0)

            xdot = np.array([
                    x[3]*np.cos(x[2]),  # x_dot
                    x[3]*np.sin(x[2]),  # y_dot
                    x[4],               # theta_dot
                    #1./(robot._m*robot._r)*(TR+TL)-(robot._c/robot._m)*x[3],    # v_dot 
                    #robot._W/(2.*robot._J*robot._r)*(TR-TL)-(robot._b/robot._J)*x[4]]) # omega_dot
                    1./m*(FL)-(c/m)*x[3],    # v_dot 
                    1./(J)*(FW)-(b/J)*x[4]]) # omega_dot            
            return xdot

        def modelo_dfdp(self, x, u, p):
            FL = u[0]
            FW = u[1]

            dfdp = np.array([[0,0,0,0],
                             [0,0,0,0],
                             [0,0,0,0],
                             [FL, -x[3], 0, 0],
                             [0, 0, FW, -x[4]]])

            return dfdp

        def setParamP(self, p):
                self.paramP = p

        def setStateP(self, x):
                self.stateP = x

        def updateP(self, xm, tau, dT = 0.01, nSamples = 11):
                tX = np.linspace(0, dT, nSamples)
                xp = solve_ivp(partial(self.modelo_robot_movil, u=tau, p=self.stateP), (0,dT), self.stateP, method='BDF', teval=tX)
                xp = ((xp.y).T)[-1,:]
                ex = xm - xp
                Ap = modelo_dfdp(self.stateP, tau, self.paramP)
                Apinv = np.linalg.pinv(Ap)
                self.paramP = self.paramP - 0.1*Apinv.dot(ex)
                print(self.paramP)
