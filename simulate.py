#------------------------------------------------------------------------------#
# Simulate a 2-wheeled differential drive mobile robot in a 2D world.
# The model includes sensors and collision checking.
# 
# The code structure is based on: watertank.py
# http://stackoverflow.com/questions/16044229/how-to-get-keyboard-input-in-pygame
# http://www.pygame.org/docs/ref/key.html
# http://www.tutorialspoint.com/python/python_numbers.htm
# https://docs.python.org/3/library/functions.html#int
#
# Miguel Torres-Torriti & Mattia Rigotti-Thompson (c) 2018

# Program flow
# - Initialize state
# - Read user inputs
# - Step model
# - Draw stuff

import sys  # find ouf constants in sys.float_info
            # https://docs.python.org/2/library/sys.html#sys.float%5Finfo

import pygame
from pygame.locals import * # Must include this line to get the OpenGL
                            # definition otherwise importing just PyGame 
                            # is fine.
from pygame import gfxdraw


import numpy as np
import time
import importlib # To import modules programatically check:
                 # https://docs.python.org/3/library/importlib.html#importlib.import_module 

from robotsim import *
from modelPredictor import Stimator

DEMO = False # Set to false only if you have an external map to load.
MAPMODNAME = 'map2' # Name of the file containing the map module   ################

if DEMO:
    # --- Graphics Variables ---
    XMAX = 640
    YMAX = 510
    screen = None
    scale = 100.0
else: 
    #import MAPMODNAME
    #import map2
    map = importlib.import_module(MAPMODNAME) # Create a module named map from file MAPMODNAME.   ####################
                                              # To import modules programatically check:
                                              # https://docs.python.org/3/library/importlib.html#importlib.import_module 

    # --- Graphics Variables ---
    XMAX = 1125
    YMAX = 700
    screen = None
    scale = 35.0

def x2screen(x):
    return int(scale*x+XMAX/2)

def y2screen(y):
    return int(-scale*y+YMAX/2)

def screen2x(x):
    return (x-XMAX/2)/scale

def screen2y(y):
    return (-y+YMAX/2)/scale


def draw_path(screen, path):
    
    Npoints = path.shape[0]
    for k in range(Npoints-1):
        R = 128 + int(127*np.cos(2*np.pi*k/Npoints))
        G = 128 + int(127*np.cos(2*np.pi*k/Npoints-2.*np.pi/3.))
        B = 128 + int(127*np.cos(2*np.pi*k/Npoints+2.*np.pi/3.))    
        pygame.draw.line(screen, (R,G,B), (x2screen(path[k,1]), y2screen(path[k,2])),(x2screen(path[k+1,1]),y2screen(path[k+1,2])), 1)
        pygame.draw.circle(screen, (R,G,B), (x2screen(path[k+1,1]),y2screen(path[k+1,2])), 5, 1)

def draw_marker(screen, x, y, radius, color = (255,255,255)):
    #print(color)
    pygame.draw.circle(screen, color , (x2screen(x), y2screen(y)), radius, 0)

def draw_markers(screen, markers_points, markers_colors,radius):
    for i in range(len(markers_points)):
        draw_marker(screen, markers_points[i][0], markers_points[i][1], radius, markers_colors[i] )

def draw_map(screen, world):
    for wall in world.walls:
        pygame.draw.line(screen, (255,255,255), (x2screen(wall[0][0]),y2screen(wall[0][1])),
            (x2screen(wall[1][0]),y2screen(wall[1][1])), 3)

    # Draw markers for the cameras
    draw_markers(screen, world.markers, world.marker_colors ,int(scale*world.marker_radius)) #marker's radius = 5


def draw_robot(screen, robot, x, y, theta, scale):
    global trace_on
    global cameras_on 
    
    pygame.draw.circle(screen, (255,255,0), (x2screen(x), y2screen(y)), int(scale*robot._R), 0)
    pygame.draw.line(screen, (255,0,0), (x2screen(x),y2screen(y)), 
        (x2screen(x+robot._R*np.cos(theta)),y2screen(y+robot._R*np.sin(theta))), 3)
    if trace_on:
        trace = robot.GetData()
        draw_path(screen, trace)

def draw_ray(screen, pose, measured_point_r_theta):

    # pose stores the robot's pose (x, y, theta)
    r = measured_point_r_theta[0]
    theta = measured_point_r_theta[1] + pose[2]
    end = (pose[0]+r*np.cos(theta), pose[1]+r*np.sin(theta))
    pygame.draw.line(screen, (0,255,0), (x2screen(pose[0]), y2screen(pose[1])), 
        (x2screen(end[0]),y2screen(end[1])), 1)

        
def draw_laserscan(screen, x, y, theta, laser_points):
    
    for point in laser_points:
        draw_ray(screen, (x, y, theta), (point[0], point[1])) # Laser points are stored in Polar coords. (r, theta)

def draw_camera(screen, x,y, theta, angle, distance):

    end_top = (x+distance*np.cos(theta+angle/2.), y+distance*np.sin(theta+angle/2.))
    end_bot = (x+distance*np.cos(theta-angle/2.), y+distance*np.sin(theta-angle/2.))

    pygame.draw.line(screen, (255,0,0), (x2screen(x), y2screen(y)), (x2screen(end_top[0]),y2screen(end_top[1])), 1)
    pygame.draw.line(screen, (255,0,0), (x2screen(x), y2screen(y)), (x2screen(end_bot[0]),y2screen(end_bot[1])), 1)
    draw_arc(screen, (x2screen(x),y2screen(y)),int(scale*distance), np.rad2deg(-theta-angle/2.),np.rad2deg(-theta+angle/2), (255,0,0))


def draw_arc(surface, center, radius, start_angle, stop_angle, color):
    x,y = center
    start_angle = int(start_angle%360)
    stop_angle = int(stop_angle%360)
    if start_angle == stop_angle:
        gfxdraw.circle(surface, x, y, radius, color)
    else:
        gfxdraw.arc(surface, x, y, radius, start_angle, stop_angle, color)




def draw_light(screen, pose, measured_point_r_theta):

    # pose stores the robot's pose (x, y, theta)
    r = measured_point_r_theta[0]
    theta = measured_point_r_theta[1] + pose[2]
    end = (pose[0]+r*np.cos(theta), pose[1]+r*np.sin(theta))
    pygame.draw.line(screen, (200,0,0), (x2screen(pose[0]), y2screen(pose[1])), 
        (x2screen(end[0]),y2screen(end[1])), 1)


def draw_cameras(screen, robot, light_points):
    if cameras_on:
        c, s = np.cos(robot._x[2]), np.sin(robot._x[2])
        R = np.array(((c, -s), (s, c)))        
        for i in range(robot._n_cameras):
            cam_pos = (robot._x[0], robot._x[1]) + R.dot(robot._rel_cam_pos[i])
            draw_camera(screen, cam_pos[0], cam_pos[1], robot._x[2], robot._angle_view, robot._max_dist_cam)
           
            #draw_cameralight(screen, cam_pos[0], cam_pos[1], robot._x[2], light_points[i])

        
def draw_cameralight(screen, x, y, theta, light_points):
    
    for point in light_points:
        draw_light(screen, (x, y, theta), (point[0], point[1])) # Light points are stored in Polar coords. (r, theta)

def draw_marker_estimated(screen, x, y, radius):
    pygame.draw.circle(screen, (255,0,0), (x2screen(x), y2screen(y)), radius, 0)

def draw_markers_estimated(screen, markers_points, radius):
    for point in markers_points.values():
        draw_marker_estimated(screen, point[0], point[1], radius)

def draw_ui(screen):

    font = pygame.font.SysFont('Arial', 24)
    screen.blit(font.render('up/down : linear speed +/-', True, (255,255,255)), (40, YMAX-30))
    screen.blit(font.render('left/right : turn speed +/-', True, (255,255,255)), (300, YMAX-30))
    screen.blit(font.render('space : stop', True, (255,255,255)), (540, YMAX-30))

def post_data(screen,tL,tR,vel = 0 ,omega = 0, masa = 0, est_m = 0, var = 0, rpm = 0,):

    font = pygame.font.SysFont('Arial', 20)
    screen.blit(font.render('Torque L:' + str(tL) + ' Torque R:' + str(tR), True, (255,255,255)), (40, YMAX-55))
    screen.blit(font.render('VelLineal:' + str(vel) + ' Omega: ' + str(omega), True, (255,255,255)), (300, YMAX-55))
    screen.blit(font.render('Real:' + str(masa) + ' Estimada:' + str(est_m) + ' Varianza:' + str(var), True, (255,255,255)),(40, YMAX-85))

    screen.blit(font.render('Motor RPM :' + str(rpm), True, (255,255,255)) , (40, YMAX-115))


def draw_image_display(screen, robot, image_data):
    CAM_XMAX = XMAX*0.4
    if cameras_on:
        #print("centers: ", image_data[0][1])

        for i in range(robot._n_cameras):
            screen.fill((255,255,255), rect=pygame.Rect(100+i*CAM_XMAX-2, 10-2, CAM_XMAX+4, 20+4))
            for j in range(len(image_data[1-i][0])):
                color = image_data[1-i][2][j]
                screen.fill(color, rect=pygame.Rect(100+i*CAM_XMAX + CAM_XMAX-(j+1)*CAM_XMAX/robot._sensor_resolution, 10, CAM_XMAX/robot._sensor_resolution+1, 20))
            for center in image_data[1-i][1]:
                screen.fill((255,0,0), rect=pygame.Rect(100+i*CAM_XMAX +CAM_XMAX-(center[0]+1)*CAM_XMAX/robot._sensor_resolution, 10, CAM_XMAX/robot._sensor_resolution+1, 20))        


    
def update_display(screen, world, robot, x, y, theta, laser_points, rocks, light_points, filtered_markers_pos, image_data, control_mode = 0, ref_traj = [] , est_m = 0, var = 0):

    global which
    screen.fill(0)
    
    #draw_laserscan(screen, x, y, theta, laser_points)
    robot.Draw(screen, scale, x2screen, y2screen)
    draw_robot(screen, robot, x, y, theta, scale)
    draw_cameras(screen, robot, light_points)
    draw_map(screen, world)
    draw_ui(screen)
    draw_markers_estimated(screen, filtered_markers_pos, int(scale*world.marker_radius))

    if control_mode == 2:
        draw_path(screen, ref_traj)

    draw_image_display(screen, robot, image_data)

    if not which:
        post_data(screen,round(robot.tau[0],2),round(robot.tau[1],2), round(robot._x[3],2), round(robot._x[4],2), round(robot._J,1), round(est_m,1), round(var,2), rpm = round(robot.GetSensorTachometer(),2))
    else:
        post_data(screen,round(robot.tau[0],2),round(robot.tau[1],2), round(robot._x[3],2), round(robot._x[4],2), round(robot._m,1), round(est_m,1), round(var,2), rpm = round(robot.GetSensorTachometer(),2))
    for rock in rocks:
        rock.Draw(screen, scale, x2screen, y2screen)
    if(var <= 0.02):
        pygame.draw.circle(screen, (0,255,0), (350, YMAX-75), 10, 0)
    pygame.display.flip()


# TODO: arreglar que se pueda remover de mas de una roca a la vez
def pickRocks(robot,rocks):
    i = 0
    rm = []
    col = False
    for rock in rocks:
        dist = np.sqrt(abs(robot._x[0] - rock.x)**2 + abs(robot._x[1] - rock.y)**2)
        if dist < robot._R + rock.r:
            col = True
            break
        i += 1
    if col:
        robot._m += rocks[i].m
        rocks.pop(i)

   
def keyboard_logic(key):
    global u_linear, u_turn, brakes_on, which, save_data, control_mode, tsim, t_cm0, trace_on, cameras_on, zoom_in, zoom_out
    
    if key == pygame.K_DOWN:
        u_linear-=0.05
        #if u_linear < -1:
        #    u_linear=-1
    if key == pygame.K_UP:
        u_linear+=0.05
        #if u_linear > 1:
        #    u_linear=1
    if key == pygame.K_RIGHT:
        u_turn-=0.05
        #if u_turn < -np.pi:
        #    u_turn=-np.pi
    if key == pygame.K_LEFT:
        u_turn+=0.05
        #if u_turn > np.pi:
        #    u_turn=np.pi
    if key == pygame.K_SPACE:
        brakes_on = True
    if key == pygame.K_q:
        u_linear = 0
        u_turn = 0
    if key == pygame.K_l:
        u_turn = 0.2
    if key == pygame.K_k:
        u_turn = -0.2
    if key == pygame.K_o:
        u_linear = 0.5
    if key == pygame.K_i:
        u_linear = -0.5
    if key == pygame.K_z:
        which = 1-which
    if key == pygame.K_s:
        save_data = True
    if key == pygame.K_1:
        t_cm0 = tsim  # Initialize the mode timer to the current simulation time
        control_mode = 1 # manual
        print('Control mode is manual')
    if key == pygame.K_2:
        t_cm0 = tsim  # Initialize the mode timer to the current simulation time
        control_mode = 2 # run open loop torque sequence
        print('Control mode is open loop torque sequence')
    if key == pygame.K_3:
        t_cm0 = tsim # Initialize the mode timer to the current simulation time
        control_mode = 3 # run closed loop reference trajectory
        print('Control mode is closed loop reference trajectory')


    if key == pygame.K_t:
        trace_on = not(trace_on)
    if key == pygame.K_c:
        cameras_on = not(cameras_on)
    if key == pygame.K_KP_MINUS:
        zoom_out=True
    if key == pygame.K_KP_PLUS:
        zoom_in=True


def handle_keyboard():
    global rocks
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # pygame.QUITis sent when the user clicks the window's "X" button, or when the system 'asks' for the process to quit
                                       # http://stackoverflow.com/questions/10080715/pygame-event-event-type-pygame-quit-confusion
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN:    # http://www.pygame.org/docs/ref/key.html
            keyboard_logic(event.key)
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            rocks.append(Rock(screen2x(pos[0]),screen2y(pos[1])))
        return True

def init_display(XMAX, YMAX, display_name):
    
    # Initialize PyGame and setup a PyGame display
    pygame.init()
    # pygame.display.set_mode()
    screen = pygame.display.set_mode((XMAX,YMAX))
    pygame.display.set_caption(display_name)
    pygame.key.set_repeat(1,50)     # Works with essentially no delay.
    #pygame.key.set_repeat(0,50)    # Doesn't work because when the delay
                                    # is set to zero, key.set_repeat is
                                    # returned to the default, disabled
                                    # state.
    return screen



def main():
    # --- Initialization ---

    
    # Initialize display
    screen = init_display(XMAX, YMAX, 'Robot Sim')

    # Initialize robot and the world
    
    # Simulation parameters
    global save_data, tsim, t_cm0
    save_data = False # SSS
    t_mv = np.load("open_loop_ref_torques.npy")
    t_mv_k = 0
    t_mv_Nsteps = t_mv.shape[0]
    
    t_x = np.load("traj_ref.npy")
    t_x_k = 0
    t_x_Nsteps = t_x.shape[0]
        
    tsim = 0.0 # Simulation time
    t_cm0 =  tsim # Initial time of the selected control mode
    
    
    # Initialize robot and the world
    
    # --- Physics Variables ---
    # System Model Parameters Needed, e.g. for controller or filter implementation
    r = 0.15 # m wheel radius
    W = 0.4  # m width
 



    global u_linear, u_turn, brakes_on, control_mode, trace_on, cameras_on, zoom_in, zoom_out , rocks, which
    u_linear = 0.0
    u_turn = 0.0
    T_l = 0
    T_r =0
    brakes_on = False
    control_mode = 1 # 1: manual, 2: semi-autonomous, 3: autonomous
    Kp_dist  =  200.0
    Kp_angle = 300#200.0
    Kd_angle = 1#200.0 
    e_dist   =   1.0
    e_angle  = np.pi
    which = 0

    trace_on = False # Turn on tracing when True
    cameras_on = True # Turn on cameras when True
    zoom_in=False # Zooming when True
    zoom_out=False



    rpm = 0
    motor_T = 0
    N = 30   # Reduction




    # Initial State
    x_init = 0.5
    y_init = -0.5
    theta_init = 0.0
    v_init = 0.0
    omega_init = 0.0

    # Initial Filter Values
    x_f = x_init
    y_f = y_init
    theta_f = theta_init
    v_f = v_init
    omega_f = omega_init
    
    # Create world
    world = World()



    if not DEMO:
        world.SetWalls([]) 
        world.SetDoors(map.doors)
        x_init = -5
        y_init = 0
        theta_init = 0
    #
    # Initialize rocks
    rocks = []
    
    # Create robot object, set its initial state and put it in the world
    robotx = Robot('CrazyBot')
    robotx.SetState(np.array([x_init, y_init, theta_init, v_init, omega_init]))
    robotx.SetWorld(world)
    robotx.SetDataLoggerOn() # SSS
    print(robotx.SetBrakesOn())

    # Initialize stimators
    lStim = Stimator(1,2,50)
    rStim = Stimator(1,2,50)
    lastEst = [0,0,0,0]
    lastVar = [0,0,0,0]

    # Initialize stimators
    lStim_cam_vel = Stimator(1,2,50)
    rStim_cam_vel = Stimator(1,2,50)
    lastEst_cam_vel = [0,0,0,0]
    lastVar_cam_vel = [0,0,0,0]

    # Initialize stimators
    lStim_cam_all = Stimator(1,2,50)
    rStim_cam_all = Stimator(1,2,50)
    lastEst_cam_all = [0,0,0,0]
    lastVar_cam_all = [0,0,0,0]        

    # Initialize time counter
    ti = time.time() # Wall-clock timer

    # Set positions of the markers for the cameras 
    markers = []
    #markers = [(4,4),(4,5),(5,4),(5,5), (-6,0),(-7,0),(-6,4), (5,-robotx._s_cam/2),(5,robotx._s_cam/2)]
    #markers.extend([(i*1.5,i*0.2-2) for i in range(-5,10)])
    markers.extend([( 7*np.sin((i/20)*2*np.pi) , 7*np.cos((i/20)*2*np.pi) ) for i in range(25)])

    world.SetMarkers(markers)
    #
    markers_cam_pos_buffer = []
    filtered_markers_pos_buffer = []
    z_vel_markers_buffer = []
    Ts_buffer = []
    filter_size = 6



    while True:

        # Check if the user pressed a key.  This can be suppresed/skipped and replaced by
        # a controller in autonomous mode.
        handle_keyboard()

        if control_mode == 1:
            T_l = u_linear+u_turn
            T_r = u_linear-u_turn
            
        elif control_mode == 2:  # run open loop torque sequence
            if tsim-t_cm0 < t_mv[t_mv_k+1,0]:
                TR_mv = t_mv[t_mv_k, 1]
                TL_mv = t_mv[t_mv_k, 2]
            else:
                t_mv_k = t_mv_k+1
                TR_mv = t_mv[t_mv_k, 1]
                TL_mv = t_mv[t_mv_k, 2]
                if t_mv_k == t_mv_Nsteps-1:
                    t_mv_k = 0
                    t_cm0 = tsim  # Reset the initial time of the task
            print('Segment: ', t_mv_k, '/', t_mv_Nsteps)
            T_l = TL_mv
            T_r = TR_mv      
        elif control_mode == 3:  # run closed loop reference trajectory
            #if tsim-t_cm0 < t_x[t_x_k+1,0]:
            if (e_dist > 0.1) or (e_angle > np.pi/10.0):
                x_ref = t_x[t_x_k, 1]
                y_ref = t_x[t_x_k, 2]
                theta_ref_traj = t_x[t_x_k, 3]
            else:
                t_x_k = t_x_k+1
                x_ref = t_x[t_x_k, 1]
                y_ref = t_x[t_x_k, 2]
                theta_ref_traj = t_x[t_x_k, 3]
                if t_x_k == t_x_Nsteps-1:
                    t_x_k = 0
                    t_cm0 = tsim   # Reset the initial time of the task
            print('Segment: ', t_x_k, '/', t_x_Nsteps)
            e_x = x_ref - x_f
            e_y = y_ref - y_f
            e_dist  = np.linalg.norm([e_x, e_y])
            theta_ref_pos_error = np.arctan2(e_y, e_x)
            theta_ref = (1.0-np.exp(-10.0*e_dist))*theta_ref_pos_error + np.exp(-10.0*e_dist)*theta_ref_traj
            #theta_ref = theta_ref_pos_error
            e_angle = theta_ref-theta_f
            if np.abs(e_angle) > np.pi/20.0:
                F_mv = 0
                robotx.SetBrakesOn()
            F_mv = Kp_dist*e_dist
            Tau_mv = Kp_angle*e_angle - Kd_angle*omega_f
            print('e_x: ', np.round(e_x, 2), 'e_y: ', np.round(e_y, 2), 'e_dist: ', np.round(e_dist, 2), 'e_angle: ', np.round(e_angle*180.0/np.pi, 2))
            # Compute wheel torques from steering forces
            # F_mv   = FR + FL         = TR_mv/r + TL_mv/r
            # Tau_mv = FR*W/2 - FL*W/2 = TR_mv*W/(2*r) - TL_mv*W/(2*r)
            # TR_mv = [  1/r       1/r  ]^{-1}[ F_mv ]  = [r/2   r/W][ F_mv ]
            # TL_mv = [W/(2*r)  -W/(2*r)]     [Tau_mv]    [r/2  -r/W][Tau_mv]
            TR_mv = (F_mv/2 + Tau_mv/W)*r
            TL_mv = (F_mv/2 - Tau_mv/W)*r
            T_l = TL_mv
            T_r = TR_mv           
        else:
            T_l = u_linear+u_turn
            T_r = u_linear-u_turn
        robotx.SetActuator(np.array([T_l,T_r]))




        # Update robot state.
        Ts = time.time() - ti # Sampling time
        robotx.SetSimulationTime(Ts) #Esto no estaba = resultados malos en tesis martin
        tsim = tsim + Ts
        robotx._t=tsim
        robotx.UpdateState()
        ti = time.time() 
        Ts_buffer.append(Ts)
        if (len(Ts_buffer)>filter_size-1):
            Ts_buffer.pop(0)
        

        T_r_m = robotx.GetSensorActuator()[0]
        T_l_m = robotx.GetSensorActuator()[1]


        epsilon_r = (T_r_m)/(robotx._tau_max) # Usando el torque medido directamente para probar
        epsilon_l = (T_l_m)/(robotx._tau_max)
        #espilon_e = 1-abs(epsilon_r)/2-abs(epsilon_l)/2

        T_r_e = epsilon_r*robotx._tau_max
        T_l_e = epsilon_l*robotx._tau_max


        # Estimate mass and viscosity coefficient
        if(abs(u_linear) > 0.1):
                d_out = [robotx.GetSensorIMU()[0], robotx.GetSensorOdometry()[0]]
                #d_out = [robotx.GetSensorIMU()[0], robotx._z_markers[0]]
                #d_out = [robotx._z_markers[2], robotx._z_markers[0]]
                #d_in = [sum(robotx.GetSensorActuator())/robotx._r]
                d_in = [(T_r_e+T_l_e)/robotx._r]
                
                #print("F1: {}".format(sum(robotx.GetSensorActuator())/robotx._r))
                
                #F = epsilon_r/2*power/(robotx._r*robotx.GetSensorEncoder('R')) + epsilon_l/2*power/(robotx._r*robotx.GetSensorEncoder('L')) # *(T_r+T_l)/(abs(T_r)+abs(T_l))
                #print("F2: {}".format(F))

                #d_in = [F]
                lStim.appendData(d_in, d_out)
                est_m, est_c = lStim.stimatePseudoInv().ravel()
                var_m, var_c = lStim.getVariability()
                lastEst[0] = est_m
                lastEst[1] = est_c
                lastVar[0] = var_m
                lastVar[1] = var_c

                d_out = [robotx.GetSensorIMU()[0], robotx._z_markers[0]]
                lStim_cam_vel.appendData(d_in, d_out)
                est_m, est_c = lStim_cam_vel.stimatePseudoInv().ravel()
                var_m, var_c = lStim_cam_vel.getVariability()
                lastEst_cam_vel[0] = est_m
                lastEst_cam_vel[1] = est_c
                lastVar_cam_vel[0] = var_m
                lastVar_cam_vel[1] = var_c

                d_out = [robotx._z_markers[2], robotx._z_markers[0]]
                lStim_cam_all.appendData(d_in, d_out)
                est_m, est_c = lStim_cam_all.stimatePseudoInv().ravel()
                var_m, var_c = lStim_cam_all.getVariability()
                lastEst_cam_all[0] = est_m
                lastEst_cam_all[1] = est_c
                lastVar_cam_all[0] = var_m
                lastVar_cam_all[1] = var_c


        # Estimate inertia and viscosity coefficient
        if(abs(u_turn) > 0.1):
                d_out = [robotx.GetSensorIMU()[1], robotx.GetSensorOdometry()[1]]
                
                #
                #d_in = [(robotx.GetSensorActuator()[0]-robotx.GetSensorActuator()[1])*robotx._W/(2*robotx._r)]  # Corregido: T_r - T_l = 2 * u_turn
                d_in = [(T_r_e-T_l_e)*robotx._W/(2*robotx._r)]

                #T_g_r = (1/(robotx._r))*(robotx._W/2)*robotx._N*robotx.GetSensorTachometer()*rpm_to_torque*epsilon_r # Usando el torque medido directamente para probar
                #T_g_l = (1/(robotx._r))*(robotx._W/2)*robotx._N*robotx.GetSensorTachometer()*rpm_to_torque*epsilon_l

                #T_g = T_g_r - T_g_l
                #d_in = [T_g]

                rStim.appendData(d_in,d_out)
                est_J, est_d = rStim.stimatePseudoInv().ravel()
                var_J, var_d = rStim.getVariability()
                lastEst[2] = est_J
                lastEst[3] = est_d
                lastVar[2] = var_J
                lastVar[3] = var_d

                d_out = [robotx.GetSensorIMU()[1], robotx._z_markers[1]]
                rStim_cam_vel.appendData(d_in,d_out)
                est_J, est_d = rStim_cam_vel.stimatePseudoInv().ravel()
                var_J, var_d = rStim_cam_vel.getVariability()
                lastEst_cam_vel[2] = est_J
                lastEst_cam_vel[3] = est_d
                lastVar_cam_vel[2] = var_J
                lastVar_cam_vel[3] = var_d

                d_out = [robotx._z_markers[3], robotx._z_markers[1]]
                rStim_cam_all.appendData(d_in,d_out)
                est_J, est_d = rStim_cam_all.stimatePseudoInv().ravel()
                var_J, var_d = rStim_cam_all.getVariability()
                lastEst_cam_all[2] = est_J
                lastEst_cam_all[3] = est_d
                lastVar_cam_all[2] = var_J
                lastVar_cam_all[3] = var_d

        # Check collision with rocks and pick them
        pickRocks(robotx, rocks)


        # Read sensors.  Get outputs from robot!
        z_gps = robotx.GetSensorGPS_Compass() # Measure [x, y, theta] 
        laser_points = robotx.GetSensorLiDAR(world, theta_start = -np.pi/2., theta_end = np.pi/2., theta_step = np.pi/20.)          # Measure range
        light_points = []
        
        image_data = []
        for i in range(robotx._n_cameras):
            cam_data = robotx.GetSensorCamera(world,i)
            light_points.append(cam_data[0])
            image_data.append(cam_data[1])

        markers_cam_pos_buffer.append(robotx.GetMarkerPos_image(image_data))
        if (len(markers_cam_pos_buffer)>filter_size):
            markers_cam_pos_buffer.pop(0)
        

        filtered_markers_pos_buffer.append(robotx.FilterMarkerPositions(markers_cam_pos_buffer,filter_size))
        if (len(filtered_markers_pos_buffer)>filter_size):
            filtered_markers_pos_buffer.pop(0)


        if len(filtered_markers_pos_buffer)==filter_size:
            z_vel_markers_buffer.append(robotx.estimateFromMarkers(filtered_markers_pos_buffer, Ts_buffer))
            #print(z_vel_markers_buffer[-1])
        if len(z_vel_markers_buffer)>filter_size:
            z_vel_markers_buffer.pop(0)
            
        
        z_IMU = robotx.GetSensorIMU()         # Measure [v_dot, omega]


        #z_markers = robotx.GetMultilateration(world)  # Measure [x, y, theta] 
        #z_dist = robotx.GetMarkerDistance(image_data)
    

        # print("Markers Measure:", z_markers)
        #print("GPS:", z_gps)
        # print("Markers Distance:", z_dist)

        x_m = z_gps[0]
        y_m = z_gps[1]
        theta_m = z_gps[2]
        a_m = z_IMU[0]
        omega_m = z_IMU[1]

        # Filter measurements.
        alpha = 0.15 # Learning rate
        x_f = (1-alpha)*x_f + alpha*x_m
        y_f = (1-alpha)*y_f + alpha*y_m
        theta_f = (1-alpha)*theta_f + alpha*theta_m
        v_f = (1-alpha)*v_f + alpha*a_m*Ts
        omega_f = (1-alpha)*omega_f + alpha*omega_m

        # Get last state and control for parameter estimation
        # before the state and the control are updated.
        x0 = robotx._x
        tau0 = robotx.tau



        # Set actuators.  Send inputs to robot!
        if brakes_on:
            robotx.SetBrakesOn()
            u_linear = 0.0
            u_turn = 0.0
            brakes_on = False
            

        if zoom_out:
            robotx.change_focal_distance(zoom_in=False)
            zoom_out=False
        elif zoom_in:
            robotx.change_focal_distance(zoom_in=True)
            zoom_in=False


        # Update the display.
        if which:
            #update_display(screen, world, robotx, z_gps[0], z_gps[1], z_gps[2], laser_points, rocks,  light_points,image_data,control_mode, t_x ,lastEst[0], lastVar[0])
            update_display(screen, world, robotx, x_f, y_f, theta_f, laser_points, rocks,  light_points, filtered_markers_pos_buffer[-1][0],image_data,control_mode, t_x ,lastEst[0], lastVar[0])

        else:
            update_display(screen, world, robotx, x_f, y_f, theta_f, laser_points, rocks,  light_points, filtered_markers_pos_buffer[-1][0],image_data,control_mode, t_x ,lastEst[2], lastVar[2])

        #time.sleep(0.01)
        # Store data
        robotx.StoreData(lastEst, lastEst_cam_vel, lastEst_cam_all) 
        if save_data:
            print("Saving data...\n")
            data = robotx.GetData()
            np.save("simdata.npy", data)
            print("File saved to 'simdata.npy'")
            #savetxt("simdata.csv",data)
            #print("File saved to 'simdata.csv'")        
            save_data = False

        #pygame.time.wait(1) # Set a wait-time to delay capture from keyboard to 1 miliseconds
                             # For very fast processes, it may be necessary to slow down the keyboard 
                             # capture rate in order to reduce fast/abrubpt responses. However, beware
                             # that this delay also reduces the sampling time of the simulator.

if __name__ == '__main__': main()
