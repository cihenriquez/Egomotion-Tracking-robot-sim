# Importar simulador y predictor
from robotsim import *
from modelPredictor import Stimator

# Importar librerias utiles
import numpy as np
import matplotlib.pyplot as plt
import sys

# Funcion para generar torques (aumentando linealmente o instantaneamente) y marcas de tiempo
def genTau(vals,times,sT,step='inf'):
    ans = np.array([vals[0]])
    for i in range(len(vals)):
        if i==0 or step=='inf' or vals[i-1]==vals[i]:
            ans = np.append(ans,np.ones((int(times[i]/sT),1))*vals[i])
        else:
            ans = np.append(ans,np.linspace(vals[i-1]+int((vals[i] - vals[i-1])*sT/times[i]),vals[i],int(times[i]/sT)))
    return ans

# Funcion auxiliar que retorna arreglos para graficar lineas verticales
def drawLine(t_arr, pos, ref):
    return [t_arr[pos],t_arr[pos]],[max(ref),min(ref)]

# Funcion de modelo discreto f(x,u) para filtro de Kalman.
# x1: velocidad lineal (o rotacional)
# x2: masa (o inercia)
# x3: coeficiente roce viscoso
def stateUp(x,u,dT=0.01):
    x1 = (1-x[2]*dT/x[1])*x[0] + dT/x[1]*u[0]
    x2 = x[1]
    x3 = x[2]
    return np.array([x1,x2,x3])

# Funcion de medicion y = z(x,u) para filtro de Kalman.
# y1: aceleracion
# y2: velocidad
def measure(x,u,dT=0.01):
    y1 = u[0]/x[1] - x[2]/x[1]*x[0]
    y2 = x[0]
    return np.array([y1,y2])

# A = df(x,u)/dx
def getA(x,u,dT=0.01):
    A = np.eye(3)
    A[0,0] = 1-x[2]*dT/x[1]
    A[0,1] = x[2]*dT*x[0]/(x[1]**2)
    A[0,2] = -dT*x[0]/x[1]
    return A

# H = dz(x,u)/dx
def getH(x,u):
    H = np.array([[-x[2]/x[1],(x[2]*x[0]-u[0])/(x[1]**2),-x[0]/x[1]],[1,0,0]])
    return H

# Funciones del algoritmo del profesor
def modelo_robot_movil(t, x, u, p, robot):
    TR = u[0]
    TL = u[1]

    m = 1.0/(robot._r*p[0])
    c = p[1]/(robot._r*p[0])
    J = (1.0/p[2])*robot._W/robot._r/2.0
    b = p[3]*((1.0/p[2])*robot._W/robot._r/2.0)
        
    xdot = np.array([
            x[3]*np.cos(x[2]),  # x_dot
            x[3]*np.sin(x[2]),  # y_dot
            x[4],               # theta_dot
            #1./(robot._m*robot._r)*(TR+TL)-(robot._c/robot._m)*x[3],    # v_dot 
            #robot._W/(2.*robot._J*robot._r)*(TR-TL)-(robot._b/robot._J)*x[4]]) # omega_dot
            1./(m*robot._r)*(TR+TL)-(c/m)*x[3],    # v_dot 
            robot._W/(2.*J*robot._r)*(TR-TL)-(b/J)*x[4]]) # omega_dot            
    return xdot 

def modelo_dfdp(x, u, p):
    TR = u[0]
    TL = u[1]
        
    # p = [alpha; beta; gamma; delta] = [(1/(m*r); c/m; W/(2*J*r); b/J]
    dfdp = np.array([[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0],
                     [TR+TL, -x[3], 0, 0],
                     [0, 0, TR-TL, -x[4]]])

    return dfdp

### Parametros simulacion

# times es un arreglo que genera marcas de tiempo para cambios de torque
iters = 4
times = [1] + [1,8,5]*iters
simTime = sum(times)

# samplingTime
sT = 0.01
t = np.linspace(0,simTime,int(simTime/sT)+1)

# Muestra progreso de la simulacion
lastProg = 0

# Ruido del actuador
actNoise = 50

# windowTime (s) y slideSave (n) son el largo de la ventana movil y cuantas estimaciones se guardan
windowTime = 10
slideSave = 20

# Pueden ser seteadas por linea de comandos
if(len(sys.argv)==2):
        windowTime = int(sys.argv[1])

if(len(sys.argv)==3):
        windowTime = int(sys.argv[1])
        slideSave = int(sys.argv[2])

size = int(windowTime/sT)
scale = 1

# Torques a aplicar [Tr, Tl]
tR = genTau([0] + [0,500,0]*iters,times,sT,step='lin')
tL = genTau([0] + [0,500,0]*iters,times,sT,step='lin')
torque = np.array([tR,tL]).transpose()

# Almacenamiento de datos (IMU aceleraciones ODO velocidades)
imuMeas = np.zeros((len(t),2))
odoMeas = np.zeros((len(t),2))

# Masa de robot tiempo a tiempo
paramRob = np.zeros((len(t),2))

# Instanciar robot
robot = Robot('Olivaw')
robot.SetSimulationTime(sT)

# Redefinir variables a estudiar
robot._m = 3614 + 50*0
robot._c = 1000
robot._J = 3400
robot._b = 600

# Cosas algoritmo de estimacion profesor
pe = np.array([1/(robot._m*0.15), robot._c/robot._m, 0.4/(2*robot._J*0.15), robot._b/robot._J])  
pe = pe*np.array([1.0, 1.0, 1.0, 1.0])
predDyn = []

# Simular
for i in range(len(t)):
    # Aplicar nuevo  torque
    robot.SetActuator(torque[i,:])

    # Simular
    robot.UpdateState()

    x0 = robot._x
    tau0 = robot.tau

    # Guardar mediciones
    imuMeas[i,:] = robot.GetSensorIMU()
    odoMeas[i,:] = robot.GetSensorOdometry()
    paramRob[i,0] = robot._m
    paramRob[i,1] = robot._c

    # Estas lineas eran para simular un cambio en la masa en el intervalo 33% a 66% del total del tiempo
    if (i*100)/len(t) >= 33 and robot._m == 3614 and (i*100)/len(t) < 66:
        robot._m = 3614+50*5

    elif (i*100)/len(t) >= 66 and robot._m == 3614+50*5:
        robot._m = 3614

    # Estimacion metodo del profesor
    predDyn.append([1.0/(0.15*pe[0]),pe[1]/(0.15*pe[0])])
    xp = solve_ivp(partial(modelo_robot_movil, u=tau0, p=pe, robot=robot), (robot._t0,robot._tf), x0, method='BDF', teval=robot._tX)
    xp = ((xp.y).T)[-1,:]
    ex = robot._x - xp
    Ap = modelo_dfdp(x0, tau0, pe)
    pe = pe - 0.15*np.linalg.pinv(Ap).dot(ex)    # Apinv = (Ap*Ap^T)*Ap^T
    #print('xx4-dyn_est: ', ex, 'm_e:', 1.0/(0.15*pe[0]), 'c_e:', pe[1]/(0.15*pe[0]), 'J_e:', (1.0/pe[2])*0.4/0.15/2.0, 'b_e:', pe[3]*((1.0/pe[2])*0.4/0.15/2.0), '\n')

    # Mostrar cuanto falta
    if (i*100)/len(t) >= lastProg + 10:
        lastProg += 10
        #print(str(lastProg)+"%")

# Convertir a matriz de numpy
predDyn = np.array(predDyn)

# Graficos para ver el simulador
#plt.figure()
#plt.subplot(211)
#plt.plot(t,imuMeas[:,1])
#plt.subplot(212)
#plt.plot(t,odoMeas[:,1])
#plt.show()

# Instanciar estimacion LS
stimL = Stimator(1,2,size,n_saveAns=slideSave,buff=True,buffSize=10)
stimL.trueVal = robot._m
stateL = np.zeros((len(t),2))
avStateL = np.zeros((len(t),2))
valsL =  np.zeros((len(t),2))
varThetaL = np.zeros((len(t),2,2))
residL = np.zeros((len(t),1))

# Instanciar estimacion TLS
stimT = Stimator(1,2,size,n_saveAns=slideSave,buff=True,buffSize=10)
stimT.trueVal = robot._m
stateTLS = np.zeros((len(t),2))

# Instanciar estimacion RLS
stimRL = Stimator(1,2,size,n_saveAns=slideSave,buff=True,buffSize=10)
stimRL.trueVal = robot._m
stateRL = np.zeros((len(t),2))
stimRL.setForget(0.99)
stimRL.tuneRLS(0.1)

# Estados iniciales RLS (caso lineal y caso rotacional)
S0 = np.array([[50**2,0],[0,5**2]])
th0 = [3500,900]
#S0 = np.array([[50**2,0],[0,8**2]])
#th0 = [3200,500]
stimRL.setInitialParams(S0,th0)

# Instanciar estimacion IV (malos resultados)
stimIV = Stimator(1,2,size,n_saveAns=slideSave,buff=True,buffSize=10)
stimIV.setZsize(2)
stateIV = np.zeros((len(t),2))

# Instanciar estimacion WLS
stimW = Stimator(1,2,size,n_saveAns=slideSave,buff=True,buffSize=10)
stimW.trueVal = robot._m
stateW = np.zeros((len(t),2))

# Instanciar estimacion EKF
stimK = Stimator(1,2,size,n_saveAns=slideSave,buff=True,buffSize=10)
stimK.trueVal = robot._m
stateK = np.zeros((len(t),3))
varStateK = np.zeros((len(t),2))

# Estado inicial EKF
x0 = np.array([0,3500,900])
#x0 = np.array([0,3200,500])
u0 = np.array([0])

# Matrices EKF
A0 = getA(x0,u0)
H0 = getH(x0,u0)
Q = np.diag([1e-4,1**2,0.1**2])
R = np.diag([0.02**2,0.02**2])
P0 = np.diag([1e-18,100**2,50**2])

# Guardar modelo EKF
stimK.setModel(A0,0,H0)
stimK.setXdFun(stateUp)
stimK.setYFun(measure)

stimK.setQ(Q)
stimK.setR(R)
stimK.setInit(x0,P0)

# Variables para marcar cosas, no tienen impacto en los algoritmos
block = [False,False,False]
fullPos = [0,0,0]

# Agregar ruido de medicion al actuador
torque = torque + np.random.randn(torque.shape[0], torque.shape[1])*actNoise

# Reiniciar lastProg para mostrar progreso
lastProg = 0

trueVal = np.array([robot._m, robot._c])

for i in range(len(t)):

    # Comienza a guardar desde los 2 segundos
    if(t[i] >= 2):
            
        # Entradas para caso lineal (entra torque, sale velocidad y aceleracion)
        d_out = [imuMeas[i,0], odoMeas[i,0]]
        d_in = [(torque[i,0] + torque[i,1])/robot._r]

        # Entradas para caso rotacional (entra torque, sale velocidad y aceleracion)
        #d_out = [imuMeas[i,1], odoMeas[i,1]]
        #d_in = [(torque[i,0] - torque[i,1])*robot._W/(2*robot._r)]

        # Estimacion LS
        stimL.appendData(d_in,d_out)
        stateL[i], valsL[i] = stimL.numpyLstSqr()
        stimL.checkParam(stateL[i])
        stimL.checkMSE(trueVal)
        avStateL[i] = stimL.getAverageMeasure().ravel()

        # Estimacion TLS
        stimT.appendData(d_in,d_out)
        t1,t2,T,t3 = stimT.tls()
        if stimT.fullInput():
            stateTLS[i] = T
        stimT.checkParam(stateTLS[i])
        stimT.checkMSE(trueVal)

        # Esto era solo para marcar cuando el estimador lineal lleno su ventana de tiempo
        if stimL.fullInput() and not block[0]:
            fullPos[0] = i
            block[0] = True

        # Estimacion RLS
        stimRL.appendData(d_in,d_out)
        stateRL[i] = stimRL.RLSStimate()
        stimRL.checkParam(stateRL[i])
        stimRL.checkMSE(trueVal)

        # Estimacion IV
        #stimIV.appendData(d_in,d_out)
        #instrument = np.array(d_out)*np.array([1,1])
        #stimIV.appendInstrument(instrument)
        #stateIV[i] = stimIV.stimateFullIV()

        # Estimacion WLS
        stimW.appendData(d_in,d_out)
        stateW[i] = stimW.wls2()
        stimW.checkParam(stateW[i])
        stimW.checkMSE(trueVal)

        # Estimacion EKF
        stimK.appendData(d_in,d_out)
        stateK[i] = stimK.eKalmanPred().ravel()
        varStateK[i,0] = stimK.kP[1,1]
        varStateK[i,1] = stimK.kP[2,2]
        jA = getA(stateK[i],d_in)
        jH = getH(stateK[i],d_in)
        stimK.setModel(jA,0,jH)
        stimK.checkParam(stateK[i,1:])
        stimK.checkMSE(trueVal)

    # Reiniciar la matriz de RLS
    if (i%(1500/sT) == 0 and i > 0):
        print("Reset matrix at time " + str(t[i]))
        stimRL.resetR()

    # Mostrar cuanto falta
    if (i*100)/len(t) >= lastProg + 10:
        lastProg += 10
        print(str(lastProg)+"%")

# Bandas de confianza de EKF, +- 3 sigma deberia ser ~99% si no me equivoco
upM = stateK[:,1] + np.sqrt(varStateK[:,0])*3
downM = stateK[:,1] - np.sqrt(varStateK[:,0])*3

upC = stateK[:,2] + np.sqrt(varStateK[:,1])*3
downC = stateK[:,2] - np.sqrt(varStateK[:,1])*3

# Grafico particular de EKF con intervalos de confianza
plt.figure()
plt.subplot(311)
plt.plot(t,stateK[:,1],t,paramRob[:,0])
plt.fill_between(t,upM,downM,alpha=0.5)
plt.ylim(3500,3900)
plt.grid(True)
plt.subplot(312)
plt.plot(t,stateK[:,2],t,paramRob[:,1])
plt.fill_between(t,upC,downC,alpha=0.5)
plt.ylim(800,1100)
plt.grid(True)
plt.subplot(313)
plt.plot(t,odoMeas[:,0],t,stateK[:,0])

# Grafico de resultados rotacionales
plt.figure()
plt.subplot(211)
plt.plot(t,np.ones((len(t),1))*robot._J,t,stateL[:,0],'m:',t,stateRL[:,0],'r--',t,stateW[:,0],'g-.',t,stateTLS[:,0],'y-.',t,stateK[:,1],'b:')
#plt.ylabel("Mass [kg]")
plt.ylabel("Inertia [kg m²]")
plt.legend(["True value","LS","RLS","FGLS","TLS","EKF"])
plt.grid(True)
#plt.ylim(0,robot._m*1.2)
plt.ylim(0,robot._J*1.2)
plt.xlim(0,20)

plt.subplot(212)
plt.plot(t,np.ones((len(t),1))*robot._b,t,stateL[:,1],'m:',t,stateRL[:,1],'r--',t,stateW[:,1],'g-.',t,stateTLS[:,1],'y-.',t,stateK[:,2],'b:')
#plt.ylabel("Viscous friction coefficient [N/(m/s)]")
plt.ylabel("Viscous rotational friction\n coefficient [Nm/(rad/s)]")
plt.xlabel("Time [s]")
plt.grid(True)
#plt.ylim(0,robot._c*1.5)
plt.ylim(0,robot._b*1.5)
plt.xlim(0,20)
#plt.show()
#plt.savefig("simRes.pdf")

# Grafico de resultados lineales
plt.figure()
plt.subplot(211)
plt.plot(t,paramRob[:,0],t,stateL[:,0],'m:',t,stateRL[:,0],'r--',t,stateW[:,0],'g-.',t,stateTLS[:,0],'y-.',t,stateK[:,1],'b:')
plt.ylabel("Mass [Kg]")
plt.legend(["True value","LS","RLS","FGLS","TLS","EKF"])
plt.grid(True)
plt.ylim(0,robot._m*1.2)
#plt.xlim(0,20)

plt.subplot(212)
plt.plot(t,paramRob[:,1],t,stateL[:,1],'m:',t,stateRL[:,1],'r--',t,stateW[:,1],'g-.',t,stateTLS[:,1],'y-.',t,stateK[:,2],'b:')
plt.ylabel("Viscous friction\n coefficient [N/(m/s)]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.ylim(0,robot._c*1.5)
plt.xlim(0,20)

# Grafico de experimento lineal
plt.figure()
plt.subplot(311)
plt.plot(t,(torque[:,0] + torque[:,1])/robot._r)
plt.ylabel("Force [N]")
plt.grid(True)
plt.subplot(312)
plt.plot(t,imuMeas[:,0])
plt.ylabel("Acceleration [m/s²]")
plt.grid(True)
plt.subplot(313)
plt.plot(t,odoMeas[:,0])
plt.ylabel("Speed [m/s]")
plt.xlabel("Time [s]")
plt.grid(True)

#plt.show()
#print("Masa LS: " + str(stimL.getBestFit()[0] - robot._m) + " | " + str(stimL.meanAns[0] - robot._m) + " | " + str(stimL.ansRMSE[0]) + " | " + str(stimL.myAns[0] - robot._m) + " | " + str(np.sqrt(stimL.myMSE[0])))
#print("Masa RLS: " + str(stimRL.getBestFit()[0] - robot._m) + " | " + str(stimRL.meanAns[0] - robot._m) + " | " + str(stimRL.ansRMSE[0]) + " | " + str(stimRL.myAns[0] - robot._m) + " | " + str(np.sqrt(stimRL.myMSE[0])))
#print("Masa FGLS: " + str(stimW.getBestFit()[0] - robot._m) + " | " + str(stimW.meanAns[0] - robot._m) + " | " + str(stimW.ansRMSE[0]) + " | " + str(stimW.myAns[0] - robot._m) + " | " + str(np.sqrt(stimW.myMSE[0])))
#print("Masa TLS: " + str(stimT.getBestFit()[0] - robot._m) + " | " + str(stimT.meanAns[0] - robot._m) + " | " + str(stimT.ansRMSE[0]) + " | " + str(stimT.myAns[0] - robot._m) + " | " + str(np.sqrt(stimT.myMSE[0])))
#print("Masa EKF: " + str(stimK.getBestFit()[0] - robot._m) + " | " + str(stimK.meanAns[0] - robot._m) + " | " + str(stimK.ansRMSE[0]) + " | " + str(stimK.myAns[0] - robot._m) + " | " + str(np.sqrt(stimK.myMSE[0])))
#print("Masa real: " + str(robot._m))
#print("Masa WLS: " + str(stimPL.getBestFit()[0]))
#print("⚡️⚡️⚡️⚡️")

print("Masa LS: " + str(stimL.meanAns[0]) + " | Roce LS: " + str(stimL.meanAns[1]))
print("Masa RLS: " + str(stimRL.meanAns[0]) + " | Roce RLS: " + str(stimRL.meanAns[1]))
print("Masa FGLS: " + str(stimW.meanAns[0]) + " | Roce FGLS: " + str(stimW.meanAns[1]))
print("Masa TLS: " + str(stimT.meanAns[0]) + " | Roce TLS: " + str(stimT.meanAns[1]))
print("Masa EKF: " + str(stimK.meanAns[0]) + " | Roce EKF: " + str(stimK.meanAns[1]))
print("⚡️⚡️⚡️⚡️")

'''
print("Masa LS: " + str(np.sqrt(stimL.getBestVar()[0,0])))
print("Masa RLS: " + str(np.sqrt(stimRL.getBestVar()[0,0])))
print("Masa FGLS: " + str(np.sqrt(stimW.getBestVar()[0,0])))
print("Masa TLS: " + str(np.sqrt(stimT.getBestVar()[0,0])))
print("Masa EKF: " + str(np.sqrt(stimK.getBestVar()[0,0])))
#print("Masa WLS: " + str(stimPL.getBestFit()[0]))
print("")
'''
