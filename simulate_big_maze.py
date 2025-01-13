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

import numpy as np

import time

from robotsim import *
import map1

# --- Graphics Variables ---
XMAX = 1125
YMAX = 900
screen = None
scale = 35.0

def x2screen(x):
    return int(scale*x+XMAX/2)

def y2screen(y):
    return int(-scale*y+YMAX/2)

def draw_map(screen, world):
    for wall in world.walls:
        pygame.draw.line(screen, (255,255,255), (x2screen(wall[0][0]),y2screen(wall[0][1])),
            (x2screen(wall[1][0]),y2screen(wall[1][1])), 3)

def draw_robot(screen, robot, x, y, theta, scale):
    
    pygame.draw.circle(screen, (255,255,0), (x2screen(x), y2screen(y)), int(scale*robot._R), 0)
    pygame.draw.line(screen, (255,0,0), (x2screen(x),y2screen(y)), 
        (x2screen(x+robot._R*np.cos(theta)),y2screen(y+robot._R*np.sin(theta))), 3)

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


def draw_ui(screen):

    font = pygame.font.SysFont('Arial', 24)
    screen.blit(font.render('up/down : linear speed +/-', True, (255,255,255)), (20, YMAX-30))
    screen.blit(font.render('left/right : turn speed +/-', True, (255,255,255)), (280, YMAX-30))
    screen.blit(font.render('space : stop', True, (255,255,255)), (520, YMAX-30))

    
def update_display(screen, world, robot, x, y, theta, laser_points):

    screen.fill(0)
    
    draw_laserscan(screen, x, y, theta, laser_points)
    robot.Draw(screen, scale, x2screen, y2screen)
    draw_robot(screen, robot, x, y, theta, scale)
    draw_map(screen, world)
    draw_ui(screen)
    
    pygame.display.flip()

   
def keyboard_logic(key):
    global u_linear, u_turn, brakes_on
    
    if key == pygame.K_DOWN:
        u_linear-=0.1
        if u_linear < -1:
            u_linear=-1
    if key == pygame.K_UP:
        u_linear+=0.1
        if u_linear > 1:
            u_linear=1
    if key == pygame.K_RIGHT:
        u_turn-=0.1
        if u_turn < -np.pi:
            u_turn=-np.pi
    if key == pygame.K_LEFT:
        u_turn+=0.1
        if u_turn > np.pi:
            u_turn=np.pi
    if key == pygame.K_SPACE:
        brakes_on = True


def handle_keyboard():
    #global time_scaling
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # pygame.QUITis sent when the user clicks the window's "X" button, or when the system 'asks' for the process to quit
                                       # http://stackoverflow.com/questions/10080715/pygame-event-event-type-pygame-quit-confusion
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN:    # http://www.pygame.org/docs/ref/key.html
            keyboard_logic(event.key)
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
    
    # --- Physics Variables --- 
    # Controls
    global u_linear, u_turn, brakes_on
    u_linear = 0.0
    u_turn = 0
    brakes_on = False
    
    # Initial State
    x_init = 0.5
    y_init = -0.5
    theta_init = 0.0
    v_init = 0.0
    omega_init = 0.0
    
    # Create world
    world = World()
    #
    world.SetWalls(map1.walls) 
    world.SetDoors(map1.doors)
    #
    
    # Create robot object, set its initial state and put it in the world
    robotx = Robot('CrazyBot')
    robotx.SetState(np.array([x_init, y_init, theta_init, v_init, omega_init]))
    robotx.SetWorld(world)
    print(robotx.SetBrakesOn())
    
    # Initialize time counter
    ti = time.time() # Wall-clock timer

    while True:

        # Check if the user pressed a key.  This can be suppresed/skipped and replaced by
        # a controller in autonomous mode.
        handle_keyboard()            

        # Set actuators.  Send inputs to robot!
        if brakes_on:
            robotx.SetBrakesOn()
            u_linear = 0.0
            u_turn = 0.0
            brakes_on = False
            
        robotx.SetActuator(0.1*np.array([u_linear+u_turn,u_linear-u_turn]))
        
        # Update robot state.
        Ts = time.time() - ti  # Sampling time
        robotx.UpdateState()
        ti = time.time() 

        # Read sensors.  Get outputs from robot!
        z_gps = robotx.GetSensorGPS_Compass() # Measure [x, y, theta] 
        laser_points = robotx.GetSensorLiDAR(world, theta_start = -np.pi/2., theta_end = np.pi/2., theta_step = np.pi/20.)          # Measure range

        # Update the display.
        update_display(screen, world, robotx, z_gps[0], z_gps[1], z_gps[2], laser_points)

        #time.sleep(0.01)

        pygame.time.wait(1) # Set a wait-time to delay capture from keyboard to 1 miliseconds
                             # For very fast processes, it may be necessary to slow down the keyboard 
                             # capture rate in order to reduce fast/abrubpt responses. However, beware
                             # that this delay also reduces the sampling time of the simulator.

if __name__ == '__main__': main()