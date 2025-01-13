import numpy as np
#from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from functools import partial  # https://docs.python.org/3.6/library/functools.html
                               # required to pass additional parameters to solve_ivp
                               
import pygame
from pygame.locals import * # Must include this line to get the OpenGL
                            # definition otherwise importing just PyGame 

import colorsys
from collections import defaultdict

class Robot:

    def __init__(self, name='nameless_robot'):
        self.name = name
        self._x = np.array([0.,0.,0.,0.,0.]) # x,y,theta,v,omega
        self._t = 0.
        self.tau = np.array([0.,0.])
        self._tau_max = 1.0 #1000
        self._Ts = 0.01 # s
        self._step_size = self._Ts
        self._t0 = 0.
        self._tf = self._t0 + self._step_size
        self._Nsamples = 10+1
        self._tX = np.linspace(self._t0, self._tf, self._Nsamples)
        # Model parameters
        self._m = 10.0 # kg          mass
        self._J = 1.0  # kg*m^2      inertia moment
        self._c = 0.3 # N/(m/s)     linear viscous friction coefficient
        self._b = 0.3  # N*m/(rad/s) rotational viscous friction coefficient
        self._r = 0.15 # m wheel radius
        self._W = 0.4  # m width
        self._L = 0.6  # m length is not necessary
        self._H = 0.15 # m height is not necessary
        self._R = np.sqrt((self._W/2.)**2.+(self._L/2.)**2.) # m collision 
        self._Kc = 10.0 # N/m     Collision spring force constant
        self._Dc = 50.0 # N/(m/s) Collision damper force damping constant
        # Disturbance parameters
        self._sigma_v = 1e-5*np.array([0.01, 0.01])
        self._Q_k = np.diag(self._sigma_v**2)
        self._nv = [0,0]
        
        # Sensor noise parameters
        
        # GPS-Compass
        self._sigma_w_GPS_Compass = 1.0*np.array([0.1, 0.1, 3./180.*np.pi])
        self._R_k_GPS_Compass = np.diag(self._sigma_w_GPS_Compass**2)
        
        # IMU
        self._sigma_w_IMU = 1.0*np.array([0.02, 0.02*np.pi])
        self._R_k_IMU = np.diag(self._sigma_w_IMU**2)
        
        # LinAngAccelVel
        self._sigma_w_LinAngAccelVel = 1.0*np.array([0.02, 0.0001, 0.0004*np.pi, 0.02*np.pi])
        self._R_k_LinAngAccelVel = np.diag(self._sigma_w_LinAngAccelVel**2)
        

        # Odometry
        self._sigma_w_ODO = 1.0*np.array([(1./360.)*2.*np.pi, (1./360.)*2.*np.pi])
        self._R_k_ODO = np.array([[4.*(self._r/2.)**2.*self._sigma_w_ODO[0], 0.],
                                  [0., 2.*(self._r/self._W)**2.*self._sigma_w_ODO[1]]])
        
        # LiDAR
        self._sigma_w_LIDAR = 1.0*np.array([0.02, (0.05/180.0)*np.pi])
        self._R_k_LIDAR = np.diag(self._sigma_w_LIDAR**2)
        
        # World
        self._world = []

        # Data logger
        self._dl_On = False
        self._dl_N = 1000 # Number of data points to store. 
                          # The time lenght of the buffer is k_dlN*Ts ~= 1000*0.02 = 200 s.
        self._dl_k = 0    # Index to entry position of the data storage array
        self._dl_record_size = 1+max((self._x).shape)+3+max((self.tau).shape)+4+4+8+8
                # Size of each record in the data array: 
                # time+state vector+referece+manipulated variable+sensor measurements = 1+5+3+2+4
        self._dl_data = [] # The array is initialized in SetDataLogger()
                

        # Cameras
        self._sensor_resolution = 500
        self._sensor_size = 1e-3 # m 
        self._pixel_size = self._sensor_size/self._sensor_resolution
        self._focal_distance = 1.5e-3#2e-3 # m
        #print("half sensor:", self._sensor_size/2.0-self._pixel_size)
        self._angle_view = 2.0*np.arctan2(((self._sensor_size-self._pixel_size)/2.0), self._focal_distance)  # rads cameras angle-of-view



        self._n_cameras = 2	# number of cameras
        self._s_cam = 0.5	# m space between cameras
        self._rel_cam_pos = [(self._R-self._focal_distance, i*self._s_cam-self._s_cam*(self._n_cameras-1)/2) for i in range(self._n_cameras)] # m position of the cameras in the robot
        self._max_dist_cam = 20	# m max vision distance of the cameras
        self._sigma_w_Cam = 0*np.array([0.002,0.002, (0.001/180.0)*np.pi])
        self._n_closer_markers = 10
        self._z_markers = [0,0,0,0]


        # Motor
        self._motor_rpm = 0
        self._N = 1






    def change_focal_distance(self, zoom_in=True, distance=0.01e-3):
        if zoom_in:
            self._focal_distance += distance
        else:
            if self._focal_distance>distance:
                self._focal_distance -= distance
        self._angle_view = 2.0*np.arctan2(((self._sensor_size-self._pixel_size)/2.0), self._focal_distance)  # rads cameras angle-of-view
        self._rel_cam_pos = [(self._R-self._focal_distance, i*self._s_cam-self._s_cam*(self._n_cameras-1)/2) for i in range(self._n_cameras)] # m position of the cameras in the robot

    def light_intersect(self, ray, markers, markers_radius, colors):
        """
        Calcula la intersección entre un rayo y un conjunto de marcadores circulares.

        Args:
            ray (tuple): Línea definida por dos puntos (C, D), en coordenadas 2D.
            markers (list): Lista de posiciones de los centros de los marcadores.
            markers_radius (float): Radio de los marcadores.
            colors (list): Lista de colores asociados a los marcadores.

        Returns:
            tuple: (bool, Vector2, tuple)
                - bool: True si hay intersección, False en caso contrario.
                - Vector2: Punto de intersección más cercano al inicio del rayo.
                - tuple: Color del marcador correspondiente o (0, 0, 0) si no hay intersección.
        """
        # Parámetros del rayo
        r = markers_radius
        C = pygame.math.Vector2(ray[0])  # Inicio del rayo
        D = pygame.math.Vector2(ray[1])  # Fin del rayo
        V = D - C  # Dirección del rayo

        max_range = self._max_dist_cam  # Rango máximo permitido
        dist_hit = max_range  # Inicializar con la distancia máxima
        p_max = C + dist_hit * V.normalize()  # Extremo máximo del rayo

        closest_hit = None
        closest_color = (0, 0, 0)

        # Filtrar marcadores por distancia inicial (opcional para optimización)
        filtered_markers = [
            (pygame.math.Vector2(marker), color)
            for marker, color in zip(markers, colors)
            if (pygame.math.Vector2(marker) - C).length() <= max_range + r
        ]

        for marker, color in filtered_markers:
            Q = marker  # Centro del marcador

            # Ecuación cuadrática: a*t^2 + b*t + c = 0
            a = V.dot(V)
            b = 2 * V.dot(C - Q)
            c = (C - Q).dot(C - Q) - r**2

            # Discriminante
            disc = b**2 - 4 * a * c
            if disc < 1e-6:  # Umbral para evitar inestabilidad numérica
                continue

            sqrt_disc = np.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            # Filtrar valores positivos de t
            valid_t = [t for t in [t1, t2] if t >= 0]
            if not valid_t:  # Si no hay valores válidos, pasar al siguiente marcador
                continue

            t = min(valid_t)  # Seleccionar el t más cercano
            if t >= dist_hit:  # Fuera del rango permitido
                continue

            # Actualizar el punto de intersección más cercano
            dist_hit = t
            closest_hit = C + t * V
            closest_color = color

        if closest_hit is not None:
            return True, closest_hit, closest_color
        return False, p_max, (0, 0, 0)




    def GetSensorCamera(self, world, camera_number = 0):
        # x, m

        light_points = []
        image_data = []
        color_data = []
        
        w = self._sigma_w_Cam*np.random.randn(len(self._sigma_w_Cam))


        c, s = np.cos(self._x[2] + w[2] ), np.sin(self._x[2] +w[2])
        R = np.array(((c, -s), (s, c)))

        self._rel_cam_pos = [(self._R-self._focal_distance+w[0], w[1]+i*self._s_cam-self._s_cam*(self._n_cameras-1)/2) for i in range(self._n_cameras)] # m position of the cameras in the robot


        cam_pos = (self._x[0], self._x[1]) + R.dot(self._rel_cam_pos[camera_number])

        angles= np.linspace(-self._angle_view/2.0,self._angle_view/2.0, self._sensor_resolution )
        #print(angles)
        #n=0
        for theta in angles:

            
            theta_w = self._x[2]+theta+w[2] # Ray direction in world coordinates
            c_ray = np.cos(theta_w)
            s_ray = np.sin(theta_w)
            
            ray = ((cam_pos[0], cam_pos[1]), (cam_pos[0]+c_ray, cam_pos[1]+s_ray))
            intersect = self.light_intersect(ray, world.markers, world.marker_radius, world.marker_colors)
            p_hit = intersect[1]
            image_data.append(intersect[0])
            color_data.append(intersect[2])

            # laser_points.append((p_hit[0]+c_ray*w[0], p_hit[1]+s_ray*w[0])) # Cartesian output
            light_points.append((np.sqrt((p_hit[0]-cam_pos[0])**2. + (p_hit[1]-cam_pos[1])**2.), theta)) # Polar output (r,theta)
            #n+=1
            #print(theta,n)
        
        markers_centers = self.GetMarkerCenterAndColor(color_data)
        #print(light_points)
        

        return light_points, [image_data, markers_centers, color_data]




    def GetMarkerCenterAndColor(self, color_data):
        """
        Calcula los centros de los marcadores y sus colores en una lista unidimensional,
        ignorando los marcadores que están tocando los bordes.

        Args:
            color_data (list of tuple): Datos de colores unidimensionales.
                                        Cada elemento es una tupla (R, G, B).
                                        El color (0, 0, 0) indica ausencia de marcador.

        Returns:
            list of tuple: Lista de (posición del centro, color asociado).
        """
        centers_and_colors = []
        add = 0
        pixels = 0
        current_color = (0, 0, 0)
        marker_start = None  # Índice inicial del marcador

        for i in range(len(color_data)):
            if color_data[i] != (0, 0, 0):  # Si no es negro (0, 0, 0)
                if pixels == 0 or color_data[i] != current_color:  # Nuevo marcador o cambio de color
                    if pixels > 0:  # Finalizar el marcador anterior
                        if marker_start != 0 and i - 1 != len(color_data) - 1:  # Ignorar si toca los bordes
                            center = add / pixels
                            centers_and_colors.append((center, current_color))
                    # Reiniciar acumuladores para el nuevo marcador
                    add = i
                    pixels = 1
                    current_color = color_data[i]
                    marker_start = i  # Marcar inicio del nuevo marcador
                else:  # Continuar acumulando el marcador actual
                    add += i
                    pixels += 1
            else:  # Fin del marcador actual
                if pixels > 0:  # Finalizar el marcador si hay datos acumulados
                    if marker_start != 0 and i - 1 != len(color_data) - 1:  # Ignorar si toca los bordes
                        center = add / pixels
                        centers_and_colors.append((center, current_color))
                # Reiniciar acumuladores
                add = 0
                pixels = 0
                current_color = (0, 0, 0)
                marker_start = None

        # Asegurar el cálculo del último marcador
        if pixels > 0 and marker_start != 0 and len(color_data) - 1 not in range(marker_start, len(color_data)):
            center = add / pixels
            centers_and_colors.append((center, current_color))

        return centers_and_colors



    def GetMarkerDistance(self, image_data):


        centers_0 = image_data[0][1]
        centers_1 = image_data[1][1]


        centers_0 = [self._sensor_resolution/2-i for i in centers_0]
        centers_1 = [self._sensor_resolution/2-i for i in centers_1]

        #print(centers_0)
        visible_marker_dist = []
        n_markers = min(len(centers_0),len(centers_1))
        for i in range(n_markers):
            visible_marker_dist.append((self._focal_distance*self._s_cam)/(self._pixel_size*(centers_0[i]-centers_1[i]))) 

        return visible_marker_dist



    # xdot = modelo_robot_movil(t, x, u):
    def _modelo_robot_movil(self, t, x, u):
        TR = u[0]
        TL = u[1]
            
        xdot = np.array([
                        x[3]*np.cos(x[2]),  # x_dot
                        x[3]*np.sin(x[2]),  # y_dot
                        x[4],               # theta_dot
                        1./(self._m*self._r)*(TR+TL)-(self._c/self._m)*x[3],          # v_dot 
                        self._W/(2.*self._J*self._r)*(TR-TL)-(self._b/self._J)*x[4]]) # omega_dot
        return xdot
    
    def __str__(self):
        return self.name
        
    def __repr__(self):
        return self.name
        
    def SetState(self, x):
        self._x = x
    
    def SetSimulationTime(self, T):
        self._Ts = T
        self._step_size = self._Ts
        self._t0 = 0.
        self._tf = self._t0 + self._step_size

    def SetWorld(self, world):
        self._world = world
        
    def UpdateState(self):
    
        #print('Updating robot state')
        # t, x = step_model(model, t0, x0, u, step_size)
    
        # odeint - Perform integration using Fortran's LSODA (Adams & BDF methods)
        # This function has been deprecated.  It is slower and the integration methods
        # cannot handle ODEs with noise, producing diverging results.
        #x = odeint(lambda x, t, *args : model(t, x, *args), x0, tX, args=(u, sigma_v))   
        #t = tfinal
        #return t, x.T
    
        # solve_ivp
        x0 = self._x
        x = solve_ivp(partial(self._modelo_robot_movil, u=self.tau), (self._t0,self._tf), x0, method='BDF', teval=self._tX)
        #return x.t, x.y
        self._x = ((x.y).T)[-1,:]
        
        # x[4] = w -> n_omega = W/(2.*J*r)*(v[0]-v[1])*T
        # x[3] = v -> n_v = 1./(m*r)*(v[0]+v[1])*T    
        # x[2] = theta -> n_theta = n_omega*T^2/2 = n_omega*T/2
        # x[1] = n_y -> n_v*(T/2)*np.sin(x[2]+n_theta)
        # x[0] = n_x -> n_v*(T/2)*np.cos(x[2]+n_theta)
        
        v = self._sigma_v*np.random.randn(len(self._sigma_v))   # '*' is the element-wise multiplication
                                                                # when using NumPy arrays.
        #v = [0,0]
        self._nv = v
        n_omega = self._W/(2.*self._J*self._r)*(v[0]-v[1])*self._Ts
        n_v = 1./(self._m*self._r)*(v[0]+v[1])*self._Ts
        n_theta = n_omega*self._Ts/2.
        n_y = n_v*(self._Ts/2.)*np.sin(self._x[2]+n_theta)
        n_x = n_v*(self._Ts/2.)*np.sin(self._x[2]+n_theta)
        
        self._x = self._x + np.array([n_x, n_y, n_theta, n_v, n_omega])
        
        if self._world:
            #collision, F_collision = self.GetSensorCollision(self._x, self._world)
            collision, F_collision = self.GetSensorCollisionPL(self._x, self._world)
            #print(collision)
            if collision:
                self._x[0] = x0[0] + F_collision[0]/self._m*self._Ts**2.
                self._x[1] = x0[1] + F_collision[1]/self._m*self._Ts**2. 
    
    def SetActuator(self, tau):

        if np.abs(tau[0]) > self._tau_max:
            tau[0] = self._tau_max*np.sign(tau[0])
        if np.abs(tau[1]) > self._tau_max:
            tau[1] = self._tau_max*np.sign(tau[1])

        self.tau = tau

        #self._motor_rpm = 0 


 



    def SetBrakesOn(self):

        self.tau = np.array([0.,0.])
        self._motor_rpm = 0
        self.SetState(np.array([self._x[0], self._x[1], self._x[2], 0., 0.]))

        return 22
        
    def GetSensorCollision(self, x, world):

        # Intersect wall and circle
        #
        #   Wall is line segment between points A and B
        #   Circle is center at pos = (x, y) = (x[0], x[1]) with radius _R
        # 
        # Line 1:  (B-A)*t1 + A = (Ax+(Bx-Ax)*t, Ay+(By-Ay)*t)
        # Line 2:  (x-Cx)^2+(y-Cy)^2 = _R^2
        #        (Ax-Cx+(Bx-Ax)*t)^2 + (Ay-Cy+(By-Ay)*t)^2 = _R^2
        #      (Ax-Cx)^2+2*(Ax-Cx)*(Bx-Ax)*t+(Bx-Ax)^2*t^2 +
        #      (Ay-Cy)^2+2*(Ay-Cy)*(By-Ay)*t+(By-Ay)^2*t^2
        #
        # Intersection:
        #
        #       t = -2*[(Ax-Cx)*(Bx-Ax)+(Ay-Cy)*(By-Ay)]+/- sqrt(D)
        #           -----------------------------------------------
        #                       2*[(Bx-Ax)^2+(By-Ay)^2]
        #
        #           D = 4*[(Ax-Cx)*(Bx-Ax)+(Ay-Cy)*(By-Ay)]-4[(Bx-Ax)^2+(By-Ay)^2]*
        #                                                   *[(Ax-Cx)^2+(Ay-Cy)^2-_R^2]
        #
        #       t \in [0,1], D>0.
        #
        #       The previous solution can be written defining
        #             a = (Bx-Ax)^2+(By-Ay)^2,
        #             b = 2*[(Ax-Cx)*(Bx-Ax)+(Ay-Cy)*(By-Ay)],
        #             c = (Ax-Cx)^2+(Ay-Cy)^2-_R^2,
        #       as t = (-b +/- sqrt(b^2-4*a*c))/(2*a).
        #

        walls = world.walls
        R = self._R

        Cx = x[0] 
        Cy = x[1]         
        
        F_collision = (0, 0)
        
        for wall in walls:
            Ax = wall[0][0]
            Ay = wall[0][1]
            Bx = wall[1][0]
            By = wall[1][1]
            
            a = (Bx-Ax)**2+(By-Ay)**2
            b = 2*(Bx-Ax)*(Ax-Cx)+2*(By-Ay)*(Ay-Cy)
            c = (Ax-Cx)**2+(Ay-Cy)**2-R**2

            if b**2-4*a*c >= 0:
                t1 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
                t2 = (-b-np.sqrt(b**2-4*a*c))/(2*a)
                if ((t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1)): return True, F_collision
        
        return False, F_collision
    
    def GetSensorCollisionPL(self, x, world):

        # Compute distance between point and line
        #
        #   Wall is line segment between points A and B
        #   Point is the centroid of the robot
        # 
        # Line 1:  (y-Ay)/(x-Ax) = (By-Ay)/(Bx-Ax)
        #          (y-Ay)*(Bx-Ax) = (x-Ax)*(By-Ay)
        #          (Ay-By)*x + (Bx-Ax)*y = Ax*(Ay-By)+Ay*(Bx-Ax)
        #
        #           n = [(By-Ay), -(Bx-Ax)]
        # 
        #           n_ = n / ||n||
        #
        #                      B
        #                pl ./
        #                  /  . .
        #                /     ___p
        #           q. /    ___  .
        #            /   ___   .
        #           A ___    .
        #         /.       .
        #         .      .
        #     y  .     .
        #     | .   .
        #     |.  .
        #     |.________x
        #    0
        #
        # Line 1: q = v*t + A,  v = B-A
        #         (p-A) + A  = p    ->  segment pA is (p-A)
        #        
        # The distance to the line is the orthogonal projection
        # of (p-A) on the normal n_ of line segment AB, i.e.:
        # 
        #                  d = (p-A).n_
        #
        # where n_ = (nx, ny)/nn, with nx = By-Ay, ny = -(Bx-Ax)
        #       nn = |(nx, ny)|
        #
        # On the other hand
        #
        #                 pl = p -d n_
        #
        # pl in the line equation ->
        #  
        #  pl = v*t + A
        #  p  -d n_ = (B-A)*t + A
        # (p-A) - d n_ = (B-A)*t
        # (p-A) - (p-A).n_ n_ = (B-A)*t
        # Noting that (B-A) is orthogonal to n_, and dot-multiplying
        # by (B-A),
        #
        # (B-A).(p-A)  = (B-A).(B-A)*t
        #
        # Thus           t = (B-A).(p-A)/(B-A).(B-A)
        #
        # Since (B-A) = (-ny, nx)
        #                t = (-ny, nx)/(nx^2+ny^2).(pAx, pAy)
        # t must belong to [0,1] to pl to lie between A and B.
        #
        walls = world.walls
        R = self._R 
        R2 = R**2

        Cx = x[0] 
        Cy = x[1]         
        
        F_collision = (0, 0)
        
        for wall in walls:
            Ax = wall[0][0]
            Ay = wall[0][1]
            Bx = wall[1][0]
            By = wall[1][1]

            nx =   By-Ay
            ny = -(Bx-Ax)
            nn2 = nx**2.+ny**2.

            pAx = Cx - Ax
            pAy = Cy - Ay
            
            d2 = (pAx*nx + pAy*ny)**2./nn2
            # t = (-ny, nx)/(nx^2+ny^2).(pAx, pAy)
            t = (nx*pAy-ny*pAx)/nn2 
            
            if (d2 <= 1.02*R2) and (t >= 0 and t <= 1):
                #print('collision', np.sqrt(d2))
                d_collision = np.sqrt(1.02*R2)-np.sqrt(d2)
                # Collision direction: a = ( np.cos(x[2]), np.sin(x[2]) )
                # Reflection direction:
                #   b = b1*n_ortho - b2*n_
                #   where n_ortho = v/|v| is orthogonal to n_
                #   and b1 = d.n_ortho, b2 = d.n_
                # b1 = (-np.cos(x[2])*ny + np.sin(x[2])*nx)/nn
                # b2 = ( np.cos(x[2])*nx + np.sin(x[2])*ny)/nn
                # b = b1*(-ny, nx)/nn - b2*(nx, ny)/nn 
                # b = (-d1*ny, d1*nx)/nn + (-d2*nx, -d2*ny)/nn
                # b = (-d1*ny -d2*nx, d1*nx -d2*ny)/nn
                # b = ( np.cos(x[2])*(ny**2.-nx**2.) -2.*np.sin(x[2])*nx*ny,
                #      -2.*np.cos(x[2])*nx*ny + np.sin(x[2])*(nx**2.-ny**2.) 
                #      )/nn2
                # where nn = sqrt(nx^2+ny^2) and nn2 = nx^2+ny^2
                nxny = nx*ny
                ny_nx = ny**2. - nx**2.
                ax = np.cos(x[2])
                ay = np.sin(x[2])
                b = (ax*ny_nx - 2.*ay*nxny, -2.*ax*nxny-ay*ny_nx)
                b_mag = np.sqrt(b[0]**2.+b[1]**2.)
                F_mag = self._Kc*d_collision + self._Dc*x[3]
                F_collision = (F_mag*b[0]/b_mag, F_mag*b[1]/b_mag)
                #incidence_angle = np.arctan2(np.abs(b1), np.abs(b2))
                #print(incidence_angle*180./np.pi)
                return True, F_collision
        return False, F_collision
        
    def GetSensorDoor(self, world):
        doors = world.doors
        R = self._R

        Cx = self._x[0] 
        Cy = self._x[1] 
        
        dist_door = -1.
        p_door = ()
        
        for door in doors:
            dx = door[0]
            dy = door[1]

            dist = np.sqrt((dx-Cx)**2+(dy-Cy)**2)
            if dist < 2*R:
                dist_door = dist
                p_door = (dx, dy)
        
        return dist_door, p_door

    def GetSensorGPS_Compass(self):
        w = self._sigma_w_GPS_Compass*np.random.randn(len(self._sigma_w_GPS_Compass))
    
        z = np.array([self._x[0],
                      self._x[1],
                      self._x[2]]) + w
        return z
    
    def GetSensorIMU(self):
        w = self._sigma_w_IMU*np.random.randn(len(self._sigma_w_IMU))
    
        z = np.array([
            1./(self._m*self._r)*(self.tau[0]+self.tau[1])-(self._c/self._m)*self._x[3], # v_dot
            self._W/(2.*self._J*self._r)*(self.tau[0]-self.tau[1])-(self._b/self._J)*self._x[4]]) + w  # omega_dot
        return z
        

    def GetSensorLinAngAccelVel(self):
        w = self._sigma_w_LinAngAccelVel*np.random.randn(len(self._sigma_w_LinAngAccelVel))
    
        z = np.array([
            1./(self._m*self._r)*(self.tau[0]+self.tau[1])-(self._c/self._m)*self._x[3], # v_dot
            self._x[3], # v
            self._W/(2.*self._J*self._r)*(self.tau[0]-self.tau[1])-(self._b/self._J)*self._x[4], # omega_dot
            self._x[4]  # omega
            ]) + w
        return z

    def GetSensorEncoder(self, wheel):
        if wheel == 'R':
            return (self._x[3] + (self._x[4]*self._W/2))/self._r
        if wheel == 'L':
            return (self._x[3] - (self._x[4]*self._W/2))/self._r
        return 0 

        

    def GetSensorOdometry(self):
        w = np.random.multivariate_normal(np.array([0.,0.]), self._R_k_ODO)
        
        z = np.array([
            self._x[3], # v
            self._x[4]]) + w  # omega
        return z


    def GetSensorActuator(self):

        w = np.random.randn(2)*self._sigma_v
        z = self.tau + w

        return z 

    def GetSensorTachometer(self):
        w = np.random.randn()*0
        z = self._motor_rpm + w
        return z
    
    def laser_intersect(self, ray, walls):
        
        # Intersect wall and ray
        #
        #   Wall is line segment between points A and B
        #   Ray  is line segment between points C and D
        # 
        # Line 1:  (B-A)*t1 + A
        # Line 2:  (D-C)*t2 + C
        #
        # Intersection: (B-A)*t1 + A = (D-C)*t2 + C
        #               (B-A)*t1 + (C-D)*t2  = C-A
        #               [(B-A) (C-D)]*[t1 t2]^T  = C-A
        #               [a b][t1] = [e]
        #               [c d][t2]   [f]
        #            
        #              [t1] = (  1 /      [d -b][e]
        #              [t2]     a*d-b*c ) [-c a][f]
        #
        #        where
        #             a = Bx-Ax,
        #             c = By-Ay,
        #             b = Cx-Dx,
        #             d = Cy-Dy,
        #             e = Cx-Ax,
        #             f = Cy-Ay.
        #
        # For other ways to solve the intersection, check:
        # https://math.stackexchange.com/questions/25171/intersection-of-two-lines-in-2d
        # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect?noredirect=1&lq=1
        # Google: intersection of two lines in 2d, line intersction 2d.

        # Ray points
        Cx = ray[0][0]
        Cy = ray[0][1]
        Dx = ray[1][0]
        Dy = ray[1][1]
        
        max_range = 10      # Set laser scanner range
        dist_hit  = max_range # first hit (distance to closest wall, i.e. minimum distance)
        #p_hit = C + dist_hit*(D-C)
        p_hit = (Cx+dist_hit*(Dx-Cx), Cy+dist_hit*(Dy-Cy))
                
        for wall in walls:
            # Wall points
            Ax = wall[0][0]
            Ay = wall[0][1]
            Bx = wall[1][0]
            By = wall[1][1]
            
            # B-A
            a = Bx-Ax 
            c = By-Ay
            # C-D
            b = Cx-Dx
            d = Cy-Dy
            # C-A
            e = Cx-Ax
            f = Cy-Ay
            
            g = a*d-b*c
            if g != 0:
                t1 = (d*e-b*f)/g
                t2 = (a*f-c*e)/g
                # Check if ray and wall intersect.
                # t1 \in [0,1] -> point between [A,B] (wall)
                # t2 > 0       -> laser hit is ahead
                if ((t1 >= 0 and t1 <= 1) and (t2 >= 0)): 
                    p_hit_x = Ax + t1*(Bx - Ax)
                    p_hit_y = Ay + t1*(By - Ay)
                    dist = np.sqrt((p_hit_x-Cx)**2+(p_hit_y-Cy)**2)
                    if dist < dist_hit:
                        dist_hit = dist
                        p_hit = (p_hit_x, p_hit_y) 
        
        return p_hit
        
    def GetSensorLiDAR(self, world, theta_start = -np.pi/2., theta_end = np.pi/2., theta_step = np.pi/20.):
        # x, m

        laser_points = []

        theta = theta_start
        while theta <= theta_end:
            w = self._sigma_w_LIDAR*np.random.randn(len(self._sigma_w_LIDAR))
            theta_w = self._x[2]+theta+w[1] # Ray direction in world coordinates
            c_ray = np.cos(theta_w)
            s_ray = np.sin(theta_w)
            ray = ((self._x[0], self._x[1]), (self._x[0]+c_ray, self._x[1]+s_ray))
            p_hit = self.laser_intersect(ray, world.walls)
            # laser_points.append((p_hit[0]+c_ray*w[0], p_hit[1]+s_ray*w[0])) # Cartesian output
            laser_points.append((np.sqrt((p_hit[0]-self._x[0])**2. + (p_hit[1]-self._x[1])**2.)+0.*w[0], theta)) # Polar output (r,theta)
            theta += theta_step
        
        z = laser_points
        return z

    def Draw(self, screen, scale, x2screen, y2screen):
            x = self._x[0];
            y = self._x[1];
            theta = self._x[2];
            pygame.draw.circle(screen, (0,0,255), (x2screen(x), y2screen(y)), int(scale*self._R), 0)
            pygame.draw.line(screen, (255,255,0), (x2screen(x),y2screen(y)), 
                (x2screen(x+self._R*np.cos(theta)),y2screen(y+self._R*np.sin(theta))), 3)
        
    def GetMarkerPos(self, world):

        markers = world.markers
        visible_markers_pos = [marker for marker in markers if (self.in_vision_range(marker, 0) and self.in_vision_range(marker, 1))]

        # (Angle of the marker relative to the robot, distance)
        v_m_rel_polar_pos = [(np.arctan2(marker[1]-self._x[1], marker[0]-self._x[0])-self.reduced_theta(),np.sqrt((marker[0]-self._x[0])**2. + (marker[1]-self._x[1])**2.)) for marker in visible_markers_pos]  # MEDICIoN, FALTA AGREGAR RUIDO

        #angle_marker = np.arctan2(marker[1]-self._x[1], marker[0]-self._x[0])-self.reduced_theta() 

        #distance_marker = np.sqrt((marker[0]-self._x[0])**2. + (marker[1]-self._x[1])**2.) 

        return visible_markers_pos, v_m_rel_polar_pos

    def GetMarkerPos_image(self, image_data):
        """
        Calcula las posiciones de los marcadores detectados con respecto a las cámaras 
        y al sistema de coordenadas global verificando correspondencia de colores,
        y filtra los `self._n_closer_markers` más cercanos.
        """
        # Extraer los centros y colores de ambas cámaras
        markers_centers_0 = image_data[0][1]
        markers_centers_1 = image_data[1][1]



        # Convertir píxeles a coordenadas reales en el sensor
        markers_centers_0 = [
            (self._pixel_size * (center[0]+0.5) - self._sensor_size / 2.0, center[1])
            for center in markers_centers_0
        ]
        markers_centers_1 = [
            (self._pixel_size * (center[0]+0.5) - self._sensor_size / 2.0, center[1])
            for center in markers_centers_1
        ]



        # Asociación por color
        matched_markers = []
        for center_0, color_0 in markers_centers_0:
            for center_1, color_1 in markers_centers_1:
                if color_0 == color_1:
                    matched_markers.append((center_0, center_1, color_0))

        #print(matched_markers)

        # Calcular posiciones relativas
        markers_with_distance = []
        for center_0_x, center_1_x, color in matched_markers:
            # Imprimir valores intermedios
            #print(f"Center 0 X: {center_0_x}, Center 1 X: {center_1_x}, Color: {color}")
            #print(f"Focal Distance: {self._focal_distance}, S_Cam: {self._s_cam}")

            # Calcular coordenadas relativas
            x = self._focal_distance * self._s_cam / (center_0_x - center_1_x)
            y = center_1_x * self._s_cam/ (center_0_x - center_1_x)  + self._s_cam / 2.0
            y2= center_0_x * self._s_cam/ (center_0_x - center_1_x)  - self._s_cam / 2.0
            #print(x)

            #print(f"Calculated Relative X: {x}, Y: {y}")
            markers_with_distance.append((x, y, color))

        # Ordenar por distancia y seleccionar los más cercanos
        markers_with_distance = sorted(markers_with_distance, key=lambda m: (m[0]**2 + m[1]**2)**0.5)[:self._n_closer_markers]

        # Cálculo de posiciones finales
        rel_markers_pos = [(x, y, color) for x, y, color in markers_with_distance]
        markers_pos = []

        c, s = np.cos(self._x[2]), np.sin(self._x[2])
        R = np.array(((c, -s), (s, c)))

        for x, y, color in rel_markers_pos:
            global_pos = [self._x[0], self._x[1]] + R.dot([x+(self._R-self._focal_distance), y])
            #print(f"Calculated Global X: {global_pos[0]}, Y: {global_pos[1]}")
            markers_pos.append((global_pos[0], global_pos[1], color))

        return markers_pos, rel_markers_pos


    def FilterMarkerPositions(self, marker_positions, n):
        """
        Filtra las posiciones de marcadores promediando las últimas `n` mediciones
        para cada color, tanto en coordenadas globales como relativas.

        Args:
            marker_positions (list of tuple): Lista de tuplas, donde cada tupla contiene:
                - Lista de posiciones globales: [(x_global, y_global, color)]
                - Lista de posiciones relativas: [(x_rel, y_rel, color)]
            n (int): Número de mediciones a promediar.

        Returns:
            tuple: (filtered_global_positions, filtered_relative_positions)
                - filtered_global_positions: Diccionario con posiciones globales promedio.
                - filtered_relative_positions: Diccionario con posiciones relativas promedio.
        """
        # Diccionarios para almacenar posiciones por color
        global_positions = defaultdict(list)
        relative_positions = defaultdict(list)

        # Agrupar las posiciones globales y relativas por color
        for global_frame, relative_frame in marker_positions:
            for x_global, y_global, color in global_frame:
                global_positions[color].append((x_global, y_global))
            for x_rel, y_rel, color in relative_frame:
                relative_positions[color].append((x_rel, y_rel))

        # Función para calcular el promedio de las últimas `n` posiciones
        def average_positions(positions, n):
            filtered_positions = {}
            for color, pos_list in positions.items():
                if len(pos_list) >= n:
                    recent_positions = pos_list[-n:]  # Últimas `n` posiciones
                else:
                    recent_positions = pos_list  # Todas las posiciones disponibles

                # Promediar coordenadas x e y
                x_avg = np.mean([pos[0] for pos in recent_positions])
                y_avg = np.mean([pos[1] for pos in recent_positions])
                filtered_positions[color] = (x_avg, y_avg)
            return filtered_positions

        # Filtrar las posiciones globales y relativas
        filtered_global_positions = average_positions(global_positions, n)
        filtered_relative_positions = average_positions(relative_positions, n)

        return filtered_global_positions, filtered_relative_positions



    def estimateFromMarkers(self, markers_pos_list, Ts_list):
        """
        Calcula la velocidad y aceleración lineal (avance en el eje X) y angular del robot diferencial
        a partir de las posiciones relativas de los marcadores con respecto al robot y 
        los intervalos de tiempo, ponderando por cercanía.

        Args:
            markers_pos_list (list of tuple): Lista de tuplas, donde cada tupla contiene:
                                              - Posiciones globales (ignorado aquí).
                                              - Diccionario de posiciones relativas.
            Ts_list (list of float): Lista de tiempos entre mediciones consecutivas, de largo `n`.

        Returns:
            tuple: (v, w, a_v, a_w)
                v (float): Velocidad lineal promedio del robot (en el eje X).
                w (float): Velocidad angular promedio del robot (en rad/s).
                a_v (float): Aceleración lineal promedio del robot (en el eje X).
                a_w (float): Aceleración angular promedio del robot (en rad/s^2).
        """
        if len(markers_pos_list) != len(Ts_list) + 1:
            raise ValueError("La lista de tiempos debe tener un elemento menos que la lista de posiciones.")
        
        # Inicialización de acumuladores
        linear_velocities = []
        angular_velocities = []
        time_intervals = Ts_list

        for i in range(len(Ts_list)):
            markers_prev = markers_pos_list[i][1]  # Diccionario de posiciones relativas (medición anterior)
            markers_curr = markers_pos_list[i + 1][1]  # Diccionario de posiciones relativas (medición actual)
            dt = Ts_list[i]  # Intervalo de tiempo entre las mediciones

            # Encontrar marcadores comunes entre mediciones consecutivas
            common_colors = set(markers_prev.keys()) & set(markers_curr.keys())
            if not common_colors:
                continue

            # Variables para acumulación
            weighted_sum_vx = 0
            weighted_sum_angular = 0
            total_weight_vx = 0
            total_weight_angular = 0

            for color in common_colors:
                x_prev, y_prev = markers_prev[color]
                x_curr, y_curr = markers_curr[color]

                # Calcular velocidades relativas
                dx = x_curr - x_prev
                linear_velocity = -dx / dt  # Negativo porque es relativo al robot

                # Calcular velocidades angulares relativas
                theta_prev = np.arctan2(y_prev, x_prev)
                theta_curr = np.arctan2(y_curr, x_curr)
                dtheta = theta_curr - theta_prev
                angular_velocity = -dtheta / dt  # Negativo porque es relativo al robot

                # Calcular distancia para ponderar
                distance = (x_curr**2 + y_curr**2)**0.5
                if distance == 0:
                    continue  # Evitar divisiones por cero

                weight = 1 / distance  # Peso inverso a la distancia

                # Acumular ponderaciones
                weighted_sum_vx += weight * linear_velocity
                total_weight_vx += weight

                weighted_sum_angular += weight * angular_velocity
                total_weight_angular += weight

            # Promediar velocidades ponderadas
            if total_weight_vx > 0:
                avg_linear_velocity = weighted_sum_vx / total_weight_vx
                linear_velocities.append(avg_linear_velocity)

            if total_weight_angular > 0:
                avg_angular_velocity = weighted_sum_angular / total_weight_angular
                angular_velocities.append(avg_angular_velocity)

        # Calcular aceleraciones a partir de las velocidades
        linear_accelerations = []
        angular_accelerations = []

        for i in range(len(linear_velocities) - 1):
            dt = time_intervals[i + 1]  # Intervalo de tiempo entre las velocidades
            a_v = (linear_velocities[i + 1] - linear_velocities[i]) / dt
            a_w = (angular_velocities[i + 1] - angular_velocities[i]) / dt
            linear_accelerations.append(a_v)
            angular_accelerations.append(a_w)

        # Promediar las velocidades y aceleraciones
        avg_linear_velocity = np.mean(linear_velocities) if linear_velocities else 0
        avg_angular_velocity = np.mean(angular_velocities) if angular_velocities else 0
        avg_linear_acceleration = np.mean(linear_accelerations) if linear_accelerations else 0
        avg_angular_acceleration = np.mean(angular_accelerations) if angular_accelerations else 0

        self._z_markers[0] = avg_linear_velocity
        self._z_markers[1] = avg_angular_velocity
        self._z_markers[2] = avg_linear_acceleration
        self._z_markers[3] = avg_angular_acceleration

        return avg_linear_velocity, avg_angular_velocity, avg_linear_acceleration, avg_angular_acceleration


    def GetMultilateration(self, world):
        if self._n_cameras>1:
            seen_markers, seen_markers_rel_pos = self.GetMarkerPos(world)
            #print("Marker pos:", seen_markers)
            #print("Marker relative_pos:", seen_markers_rel_pos)
            if len(seen_markers)>=2:
            	m1 = seen_markers[1]
            	m2 = seen_markers[0]
            	d1 = seen_markers_rel_pos[1][1]
            	d2 = seen_markers_rel_pos[0][1]
            	separation_markers = np.sqrt((m1[0]-m2[0])**2. + (m1[1]-m2[1])**2.)
            	U = separation_markers
            	x = ((d1**2)-(d2**2)+U**2)/(2*U)
            	y1 = np.sqrt(d1**2-x**2)#+m1[1]s
            	y2 = -np.sqrt(d1**2-x**2)#+m1[1]

            	ang1 = np.arctan2(y1,x)+ seen_markers_rel_pos[1][0]
            	ang2= np.arctan2(y2,x)+seen_markers_rel_pos[1][0]
            	ang3 = np.arctan2(y1,x)+ seen_markers_rel_pos[0][0]
            	ang4= np.arctan2(y2,x)+seen_markers_rel_pos[0][0]
            	return [(x+m1[0],y1+m1[1]),(x+m1[0],y2+m1[1])]#, (ang1,ang2), (ang3,ang4)]
        return None

    def reduced_theta(self):

    	angle = self._x[2] % (2.*np.pi)
    	angle = (angle + 2.*np.pi) % (2.*np.pi)
    	if (angle > np.pi):
    		angle -= (np.pi*2.)

    	return angle

    def in_vision_range(self, marker, camera_number=0):
        x = self._x[0]
        y = self._x[1]
        theta = self.reduced_theta()

        c, s = np.cos(self._x[2] ), np.sin(self._x[2] )
        R = np.array(((c, -s), (s, c)))

        cam_pos = (self._x[0], self._x[1]) + R.dot(self._rel_cam_pos[camera_number])
        #print("cam_pos:", cam_pos)

        angle_marker = np.arctan2(marker[1]-cam_pos[1], marker[0]-cam_pos[0])

        #print("y:", marker[1]-cam_pos[1], "       x:", marker[0]-cam_pos[0])
        #print("Angle marker:", angle_marker, "Angle robot:", theta)


        distance_marker = np.sqrt((marker[0]-cam_pos[0])**2. + (marker[1]-cam_pos[1])**2.)
        #print("Dist:", distance_marker)

        if (angle_marker <= theta+self._angle_view/2.) and (angle_marker>= theta-self._angle_view/2.) and distance_marker<=self._max_dist_cam:   #Revisar problemas en angulos cercanos a pi y -pi
            return True

        return False




    def SetDataLoggerOn(self):
        self._dl_On = True
        self._dl_data = np.zeros((self._dl_N,self._dl_record_size)) 
                # Store in each row: time, state vector (row-wise), reference, manipulated variable
                # http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html

    def StoreData(self, est1, est2, est3):
        # Store in row 'k' of the 'self._dl_data' array of size 
        # (rows, cols)= (self._dl_N, self._dl_record_size), 
        # the following values are arranged column wise:
        # time t, state x, referece ref and manipulated variables mv.
        # The record size is thus: size(t)+size(x)+size(ref)+size(mv)+size(sensor),
        # which for a planar differential drive mobile robot is
        # 1+5+3+2+4 = 15
        #        
        # Data is stored in a circular buffer, aka ring buffer or cyclic buffer
        z = self.GetSensorLinAngAccelVel()    
        
        self._dl_data[self._dl_k,:] = np.r_[self._t, np.r_[self._x, np.r_[np.array([0.,0.,0.]), np.r_[self.tau,np.r_[z,np.r_[self._z_markers, np.r_[est1, np.r_[est2,np.r_[est3, np.array([self._m, self._c, self._J, self._b])]]]]]]]]]
        self._dl_k = self._dl_k+1
        if self._dl_k == self._dl_N:
            self._dl_k = 0
        
    def ArrangeData(self):
        # Arrange the order of the data in the circular buffer from oldest to newest sample
        
        if self._dl_k > 0:
            aux1 = self._dl_data[:self._dl_k,:]               # Pick the first k_data values
            aux2 = self._dl_data[-(self._dl_N-self._dl_k):,:] # Pick the remaining values in the data array
            arranged_dl_data = np.r_[aux2,aux1]               # Vertically stack the rows of aux2 and aux1
        else:
            arranged_dl_data = self._dl_data
        # Check http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html
        # for info on array handling.
        
        return arranged_dl_data

    def GetData(self):
        
        arranged_data = self.ArrangeData()
        return arranged_data
    
    def GetRawData(self):
        
        raw_data = self._dl_data
        
        return raw_data

    def GetParameterEstimationData(self):
        # The record size is thus: size(t)+size(x)+size(ref)+size(mv)+size(sensor),
        # which for a planar differential drive mobile robot is
        # 1+5+3+2+4 = 15
        TR = self._dl_data[:,-6:-5]
        TL = self._dl_data[:,-5:-4]
        Ftot = (TR+TL)/(self._r)
        Tau_tot = self._W*(TR-TL)/(self._r)
        
        raw_data = np.c_[self._dl_data[:,-4:],np.c_[Ftot,Tau_tot]]
        
        return raw_data



class World:
    # map
    def __init__(self):
        self.max_x = 6
        self.max_y = 4
        self.walls = [[(0,0),(2,0)],
                      [(2,0),(2,1)],
                      [(2,1),(1,1)],
                      [(-1,1),(0,1)],
                      [(0,1),(0,-1)],
                      [(0,-1),(2,-1)],
                      [(1,-1),(1,-2)],
                      [(-2,0),(-1,0)],
                      [(-1,0),(-1,-1)],
                      [(-2,2),(-2,0)],
                      [(-3,-1),(-2,-1)],
                      [(-3,-2),(-3,2)],
                      [(3,-2),(-3,-2)],
                      [(3,2),(3,-2)],
                      [(-3,2),(3,2)]]
        self.doors = [(-3,-2),
                      (-3, 2)]
        self.markers = []
        self.marker_colors = []
        self.marker_radius = 0.1# m
    
    def SetWalls(self, walls_map):
        self.walls = walls_map

    def SetDoors(self, doors_list):
        self.doors = doors_list

    def SetMarkers(self, markers_list):
        self.markers = markers_list
        self.marker_colors = [colorsys.hsv_to_rgb(np.random.rand(),1,1) for i in self.markers]
        
        self.marker_colors = [tuple([int(channel * 255) for channel in color]) for color in self.marker_colors]




class Rock:
    # Rocks that can be picked up
    def __init__(self,x,y,rad = 0.05, mass = 0.5):
        self.x = x
        self.y = y
        self.r = rad
        self.m = mass

    def Draw(self, screen, scale, x2screen, y2screen):
            x = self.x
            y = self.y
            pygame.draw.circle(screen, (188,143,143), (x2screen(x), y2screen(y)), int(scale*self.r), 0)
            
def main():

    robotx = Robot()
    robotx.variable = "pio"
    print(robotx.variable)
    print(robotx._x)
    robotx.UpdateState()
    print(robotx._x)
    print(robotx.tau)
    robotx.SetActuator(np.array([3.0,-20.0]))
    print(robotx.tau)
    robotx.UpdateState()
    print(robotx._x)
    robotx.SetState(np.array([0.,0.,0.,0.,0.]))
    robotx.SetSimulationTime(1.)
    robotx.UpdateState()
    print(robotx._x)
    print(robotx.GetSensorGPS_Compass())
    print(robotx.GetSensorIMU())
    print(robotx.GetSensorOdometry())
    print(robotx.tau)
    world = World()
    print(world.walls)
    print(robotx.GetSensorLiDAR(world, theta_start = -np.pi/2., theta_end = np.pi/2., theta_step = np.pi/20.))
    print(robotx.GetSensorCollision(np.array([0.,0.]), world)) # Intersection between circle and line
                                                               # only requires point and radius.  
                                                               # Repulsion is proportional to collision
                                                               # depth of the circle within the wall.
    print(robotx.GetSensorCollisionPL(np.array([0.,0.,np.pi/4,0.1,0.0]), world)) # Intersection between
                                                                                 # line and wall.
                                                                                 # Requires line starting
                                                                                 # point, but also line
                                                                                 # direction.  The repulsion
                                                                                 # force is proportional
                                                                                 # to the velocity and
                                                                                 # thus the full state vector
                                                                                 # is needed unlike the
                                                                                 # simpler collision
                                                                                 # detection.
    
    walls = [[(-1,-1),(1,-1)],
             [(1,-1),(1,1)],
             [(1,1),(-1,1)],
             [(-1,1),(-1,-1)],
             [(0,0),(0,1)]]
    doors = [(1,0),
             (-1, 1)]
    world.SetWalls(walls)
    world.SetDoors(doors)
    print(world.walls)
    print(world.doors)
    
    robotx.SetWorld(world)

    robotx.SetState(np.array([.1,2.,0.,0.,0.]))
    print(robotx.GetSensorDoor(world))
    robotx.SetState(np.array([-0.9,0.7,0.,0.,0.]))
    print(robotx.GetSensorDoor(world))
    robotx.SetBrakesOn()
    

    print(len(robotx._dl_data))
    robotx.SetDataLoggerOn()
    print((robotx._dl_data).shape)

    
if __name__ == '__main__':
    main()
