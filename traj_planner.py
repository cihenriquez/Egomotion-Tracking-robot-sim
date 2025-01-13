#------------------------------------------------------------------------------#
# Trajectory planner for a 2-wheeled differential drive mobile robot in a 2D world.
#
# Miguel Torres-Torriti (c) 2018
#

import matplotlib.pyplot as plt
#import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D

import numpy as np

# Open loop piecewise constant torques
# Create an open loop history of control signals manipulated variables (MVs)
# that directly set the values of the right and left wheel torques at each
# time instant.
#
# Input: [t_k*, TR_k*, TL_k*], ti, tf, Ts. 
#  [t_k*, TR_k*, TL_k*]: Array containing torque values at given times.
#  
#  t_k*: time instant at which torques TR_k* and TL_k* start to be applied.
#        The values TR_k* and TL_k* are kept constant until t_(k+1)*. 
#  TR_k*: Right wheel torque at time t_k*.
#  TL_k*: Left wheel torque at time t_k*.
#  ti: initial time
#  tf: final time
#
# Output: 
#  [t_k, TR_k, TL_k]: Array with rows containing torque values 
#         sampled every Ts, i.e. at times t_k = t_(k-1) + Ts = k*Ts
#         between times ti and tf.
#
def open_loop_pwc(t_mv_points = np.array([[0.3,2.,1.],[0.7,-2.0,-1.0],[1.1,0.5,-0.5]]), ti = 0., tf = 2.0, Ts = 0.1):
    
    Nsamples = int((tf-ti)/Ts)+1  # Compute the total number of samples
    
    t_mv = np.zeros((Nsamples,3)) # Initialize the output array
    
    # Initial segment between ti and the first reference point
    t = (np.arange(ti, t_mv_points[0,0], Ts))
    Nsegment = len(t)
    # Set time
    t_mv[0:Nsegment,0:1] = t.reshape(Nsegment,1)
    # Set torques
    t_mv[0:Nsegment,1:2] = np.zeros((Nsegment,1))
    t_mv[0:Nsegment,2:3] = np.zeros((Nsegment,1))
        
    ti_aux = t_mv[Nsegment-1,0] + Ts  #  Starting time of the new segment is set to
                                      #  the last value of the last segment + Ts.
    Nsegment_prev = Nsegment # Samples in previous segment (used to offset the array index)
    
    
    for k in range(t_mv_points.shape[0]-1):
        t = np.arange(ti_aux, t_mv_points[k+1,0], Ts)
        Nsegment = len(t)
        # Set time
        t_mv[Nsegment_prev:Nsegment_prev+Nsegment,0:1] = t.reshape(Nsegment,1)
        # Set torques
        t_mv[Nsegment_prev:Nsegment_prev+Nsegment,1:2] = t_mv_points[k,1]*np.ones((Nsegment,1))
        t_mv[Nsegment_prev:Nsegment_prev+Nsegment,2:3] = t_mv_points[k,2]*np.ones((Nsegment,1))
        
        ti_aux = t_mv[Nsegment_prev+Nsegment-1,0] + Ts
        Nsegment_prev = Nsegment_prev+Nsegment
                
    
    # Compute the values for the last segment between the last reference point and final time tf.
    t = np.arange(ti_aux, tf, Ts)
    Nsegment = len(t)
    # Set time
    t_mv[Nsegment_prev:Nsegment_prev+Nsegment,0:1] = t.reshape(Nsegment,1)
    # Set torques
    t_mv[Nsegment_prev:Nsegment_prev+Nsegment,1:2] = t_mv_points[k+1,1]*np.ones((Nsegment,1))
    t_mv[Nsegment_prev:Nsegment_prev+Nsegment,2:3] = t_mv_points[k+1,2]*np.ones((Nsegment,1))
    
    # Copy the last value if the total segment samples has not exactly reached Nsamples
    if Nsegment_prev+Nsegment < Nsamples:
        # Set time
        t_mv[Nsegment_prev+Nsegment,0:1] = t_mv[Nsegment_prev+Nsegment-1,0:1] + Ts
        # Set torque
        t_mv[Nsegment_prev+Nsegment,1:2] = t_mv[Nsegment_prev+Nsegment-1,1:2]
        t_mv[Nsegment_prev+Nsegment,2:3] = t_mv[Nsegment_prev+Nsegment-1,2:3]

         
    return t_mv


def open_loop_prbs(ti = 0., tf = 100.0, Ts = 0.1, DT = 10.0, seed_number = 0):
    Nsamples = int((tf-ti)/Ts)+1  # Compute the total number of samples
    
    t_mv = np.zeros((Nsamples,3)) # Initialize the output array
    
    # Initial segment between ti and the first reference point
    t = (np.arange(ti, tf, Ts))
    Nsegment = len(t)
    
    t_mv[0:Nsegment,0:1] = t.reshape(Nsegment,1)
    
    np.random.seed(seed_number)
    Nhigh = int(DT/Ts) +1
    Nlow  = int(0.5*DT/Ts) 
    
    # Compute TR PRBS
    n = np.random.randint(Nlow, Nhigh) # Initial skip
    Nleft = Nsamples - n
    n_prev = n
    on_flag = True
    while Nleft > Nhigh:
        n = np.random.randint(Nlow, Nhigh)
        t_mv[n_prev:n_prev+n,1:2] = float(on_flag)*np.ones((n,1)) # TR
        n_prev = n_prev + n
        Nleft = Nleft - n
        on_flag = not(on_flag)
    
    # Compute TL PRBS
    n = np.random.randint(Nlow, Nhigh) # Initial skip
    Nleft = Nsamples - n
    n_prev = n
    on_flag = True
    while Nleft > Nhigh:
        n = np.random.randint(Nlow, Nhigh)
        t_mv[n_prev:n_prev+n,2:3] = float(on_flag)*np.ones((n,1)) # TL
        n_prev = n_prev + n
        Nleft = Nleft - n
        on_flag = not(on_flag)
    
    # Copy the last value if the total segment samples has not exactly reached Nsamples
    if Nsegment < Nsamples:
        t_mv[Nsegment,0:1] = t_mv[Nsegment-1,0:1] + Ts
        # Set torque
        t_mv[Nsegment,1:2] = t_mv[Nsegment-1,1:2]
        t_mv[Nsegment,2:3] = t_mv[Nsegment-1,2:3]
    
    return t_mv
   

def xtraj(t_x_points = np.array([[0.0,2.,1.],[0.7,-2.0,-1.0],[1.1,0.5,-0.5],[2.0,2.,1.]]), Ts = 0.1, v_max = 1.0, omega_max = 1.0):
    
    DeltaP_max = Ts*v_max
    DeltaTheta_max = Ts*omega_max

    t = []
    x = []
    y = []
    theta = []
    v = []
    
    for k in range(t_x_points.shape[0]-1):
        t0 = t_x_points[k,0]
        t1 = t_x_points[k+1,0]
        p0 = t_x_points[k,1:3]
        p1 = t_x_points[k+1,1:3]
        
        DeltaP = np.linalg.norm(p1-p0)
        NpointsP = int(DeltaP/DeltaP_max)
        NpointsT = int((t1-t0)/Ts)
        Npoints = max(NpointsP, NpointsT)

        theta_p0_p1 = np.arctan2(p1[1]-p0[1],p1[0]-p0[0])
        v_ref = (DeltaP/Npoints)/Ts
        
        t = np.array([*t, *np.linspace(t0,t1,Npoints)])
        x = np.array([*x, *np.linspace(p0[0],p1[0],Npoints)])
        y = np.array([*y, *np.linspace(p0[1],p1[1],Npoints)])
        theta = np.array([*theta, *(theta_p0_p1*np.ones(Npoints))])
        v = np.array([*v, *(v_ref*np.ones(Npoints))])
        
    t_x = np.array([[*t],[*x],[*y],[*theta],[*v]]).T
        
    return t_x
    

def main():
    #t_mv1 = open_loop_pwc()
    #t_mv = open_loop_pwc(np.array([[0.3,2.,1.],[0.7,-2.0,-1.0],[1.1,0.5,-0.5]]), 0., 2.0, 0.1)
    t_mv = open_loop_prbs(ti = 0., tf = 100.0, Ts = 0.1, DT = 10.0, seed_number = 0)
    #print(t_mv)

    np.save("open_loop_ref_torques.npy", t_mv)

    t = t_mv[:,0]
    TR = t_mv[:,1]
    TL = t_mv[:,2]
    
    print("Drawing plot 1...")
    fig1 = plt.figure()
    fig1.canvas.set_window_title('Open loop piecewise constant torques: open_loop_pwc()')
    plt.plot(t, TR, t, TL)
    plt.xlabel('t [s]')
    plt.ylabel('TR [Nm], TL [Nm]')
    plt.legend(['TR [Nm]', 'TL [Nm]'])
    
    t_x = xtraj(t_x_points = np.array([[0.0,2.,1.],[0.7,-2.0,-1.0],[1.1,0.5,-0.5],[2.0,2.,1.]]), Ts = 0.1, v_max = 1.0, omega_max = 1.0)
    np.save("traj_ref.npy", t_x)
    print(t_x)

    t_ref = t_x[:,0]
    x_ref = t_x[:,1]
    y_ref = t_x[:,2]
    theta_ref = t_x[:,3]
    v_ref = t_x[:,4]
    
    print("Drawing plot 2...")
    fig2 = plt.figure()
    fig2.canvas.set_window_title('Reference trajectory vs. time: xtraj()')
    plt.plot(t_ref, x_ref, t_ref, y_ref, t_ref, theta_ref, t_ref, v_ref)
    plt.xlabel('t [s]')
    plt.ylabel('x_ref [m], y_ref [m], theta_ref [rad], v_ref [m/s]')
    plt.legend(['x_ref [m]', 'y_ref [m]', 'theta_ref [rad]', 'v_ref [m/s]'])
    
    print("Drawing plot 3...")
    fig3 = plt.figure()
    fig3.canvas.set_window_title('Reference trajectory y(t) vs. x(t): xtraj()')
    plt.plot(x_ref, y_ref)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()

if __name__ == '__main__': main()