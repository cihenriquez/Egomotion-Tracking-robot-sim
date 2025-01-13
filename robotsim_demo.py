#------------------------------------------------------------------------------#
# robotsim_demo.py:
#
# Este código demuestra la funcionalidad básica del módulo robotsim.py.  El módulo 
# contiene la funcionalidad necesaria para simular un robot móvil en un mundo 2D.
#
# El código simulate.py demuestra como usar la funcionalidad básica en una 
# simulación interactiva, en la que el usuario puede modificar los torque de
# las ruedas.
#
# Para entender la funcionalidad básica es importante pensar que el robot como
# objeto físico existe en un mundo, recibe comandos que generar cambios en el
# estado del robot (o el mundo) y posee sensores que permiten medir ciertas
# variables asociadas al estado del robot o el mundo. En este sentido 
# corresponde estructura el módulo de simulación en dos grandes categorías (clases):
#    - el mundo: clase World()
#    - el robot: clase Robot()
#
# El mundo se describe a través de un mapa.  Aquí hablar de mundo o mapa es 
# lo mismo.  Las funciones (métodos) asociados a la clase mundo permiten 
# definir nuevos mapas, que se conforman por una colección de paredes y puertas
# como se muestra abajo.
# 
# El robot es un objeto que recibe inputs (comandos) y entrega outputs (mediciones).
# Los inputs se fijan con comandos que empiezan con la palabra "Set".
# Los outputs se obtiene con comandos que empiezan con la palabra "Get".
# La actualización del estado conlleva simular el robot por un  lapso de tiempo
# de acuerdo al modelo dinámico.  Esto involucra varias funciones asociadas a la
# actualización del estado y la simulación.
#
# Diagramáticamente, el proceso de simulación ('update state') involucra:
#  update state: (t_L, t_R) -> Robot -> (x,y,theta) --> Sensor -> Measurement
#
# Lo primero que observamos es que robot tiene:
#   Un estado incial
#   Señales de entrada
#   Señales de salida
#   Un mecanismo para simular el cambio de estado
# 
# Por lo anterior las funciones se resumen en:
#  
#  1. Configuración del estado incial
#       - Robot.SetState
#
#  2. Señales de entrada - Set
#       - Robot.SetActuator
#       - Robot.SetBrakesOn
#
#  3. Señales de salida  - Get
#       - Robot.GetSensorGPS_Compass
#       - Robot.GetSensorIMU
#       - Robot.GetSensorOdometry
#       - Robot.GetSensorLiDAR
#       - Robot.GetSensorCollision
#       - Robot.GetSensorDoor
#
#  4. Funciones de actualización
#       - Robot.SetSimulationTime
#       - Robot.UpdateState
#
# El uso de cada una de estas funciones fundamentales se ilustra a continuación.
#
# Miguel Torres-Torriti, Mattia Rigotti (c) 2018

from robotsim import *

def main():

    # --- Creación de un mundo ---
    print('--- Creación de un mundo ---')
    # Creación de un mundo.  En este contexto crear un mundo  es básicamente
    # crear un mapa con paredes y objetos.  La clase mundo puede ser ampliada 
    # con otros objetos definidos por el usuario.
    world = World() # Crea un mundo por defecto con paredes y puertas.

    # Las paredes son listas de segmentos entre dos puntos
    walls = [[(-1,-1),(1,-1)],
             [(1,-1),(1,1)],
             [(1,1),(-1,1)],
             [(-1,1),(-1,-1)],
             [(0,0),(0,1)]]
    # Las puertas son listas de puntos donde se ubican las puertas
    doors = [(1,0),
             (-1, 1)]
    
    # Las paredes y puertas del mundo  por defecto se pueden cambiar
    # como se muestra a continuación
    world.SetWalls(walls) 
    world.SetDoors(doors)
    print('Un mundo con paredes y puertas fue creado.')
    print(walls)
    print(doors)
    input('Presione una tecla para continuar...\n')
    
    # --- Creación de un robot ---
    print('--- Creación de un robot ---')
    robotx = Robot() # Crea un robot móbil de tracción diferencial con estado
                     # y controles inicializados en cero por defecto.
                     # Si no se pasa un string a Robot(), el nombre por defecto
                     # del robot es 'nameless_robot'
    print('Se creo el robot llamado: ', robotx)
    input('Presione una tecla para continuar...\n')
        
    # Asignación del robot creado a un mundo.                 
    robotx.SetWorld(world)  # Esta línea es opcional.  Si no se asigna un mundo,
                            # se puede simular el robot sin un mundo asignado.
                            # Sin embargo, aún cuando se haya creado un mundo
                            # el robot no verificará colisiones porque inicialmente
                            # el robot está inicializado con un mundo vacío, 
                            # en otras palabras el robot inicialmente se origna
                            # y vive en el eterno vacío infinito.
                     
    # --- Sensores --- (Outputs del robot)
    print('--- Sensores ---')
    # Los sensores del robot incluyen GPS, IMU, Odometria, LiDAR, Colisión y Puerta.
    # Cada uno de estos retorna una medición, la cual podría guardarse en 
    # una variable haciendo:
    #     z = GetSensorNAME()
    # donde NAME es el nombre del sensor.
    # A continuación se muestra el resultado de lo que entrega cada sensor.
    #
    # GPS
    print('GPS & Brújula: ', robotx.GetSensorGPS_Compass())
    # IMU
    print('IMU: ', robotx.GetSensorIMU())
    # Odometría
    print('Odometría: ', robotx.GetSensorOdometry())
    # LiDAR
    print('LiDAR: ', robotx.GetSensorLiDAR(world, theta_start = -np.pi/2., theta_end = np.pi/2., theta_step = np.pi/20.))
    # Collision
    # .. Dada la posición del robot (x,y) en un array y el mapa, retorna verdadero
    #    o falso si el robot está chocando con una pared.
    print('Colision: ', robotx.GetSensorCollision(np.array([0.,0.]), world))
    # Puerta
    # .. Dada la posición del robot (x,y) (estado interno) y el mapa, retorna 
    #    la distancia a la puerta más próxima y la coordenada (xd,yd)
    #    de la puerta.  Si no hay una puerta próxima entrega una distancia -1
    #    y una tupla vacía ().
    #    A contunación se muestran dos ejemplos. El primero interroga la
    #    existencia de una puerta desde una posición lejana a las puertas
    #    cargadas en el mapa.  El segundo desde una posición cercana a la puerta
    #    cargada en el mapa con el comando world.SetDoors(doors) del ejemplo anterior.
    print('Puerta lejana : ', robotx.GetSensorDoor(world))
    world.SetDoors([(-0.1,0.2)])
    print('Puerta cercana: ', robotx.GetSensorDoor(world))
    print('Se leyeron sensores GPS, IMU, odometria, LiDAR, colision y puerta.')
    input('Presione una tecla para continuar...\n')
    
    # --- Actuadores --- (Inputs al robot y acciones)
    print('--- Actuadores ---')
    # Los torques del robot de cada rueda se guardan en la variable tau.
    # La variable tau es un array de dos elementos, el primero guarda el torque
    # de la rueda deracha, el segundo el de la rueda izquierda.
    # Inicialmente el torque es cero para ambos motores como se puede ver 
    # a través del siguiente comando.
    print('Torque actual: ', robotx.tau)
    print('Posición inicial medida: ', robotx.GetSensorGPS_Compass())                 
    print('Se actualiza el estado... no debería cambiar porque los torques aplicados son cero.')
    robotx.UpdateState()
    print('Posición medida después de aplicar torques nulos: ', robotx.GetSensorGPS_Compass()) 
    print('\n')
    
    print('Se aplican torques...')
    robotx.SetActuator(np.array([3.0,-20.0]))
    print('Torques modificados: ', robotx.tau)
    print('Se actualiza el estado... ahora debería cambiar porque los torques no son nulos.')
    robotx.UpdateState()
    print('Posición medida después de aplicar torques: ', robotx.GetSensorGPS_Compass()) 
    print('No cambio mucho porque el periodo de simulación fue el valor por defecto Ts = 0.01 s')
    print('\n')
    
    print('Se cambia el tiempo de muestreo y por consiguiente de simulación a 1 s.')
    robotx.SetSimulationTime(1.)
    robotx.UpdateState()
    print('Posición medida después de aplicar torques por 1 s: ', robotx.GetSensorGPS_Compass())
    print('\n')
    
    print('Se puede inicializar el estado del robot para ponerlo en una posición distinta del mapa.  El estado por defecto es (x, y, theta, v, omega) = (0, 0, 0, 0, 0)')
    robotx.SetState(np.array([8.5, -2.5, -0.7854, 2., -1.]))
    print('Posición medida en nueva ubicación: ', robotx.GetSensorGPS_Compass())
    print('Medición de la IMU: ', robotx.GetSensorIMU())
    print('\n')
    
    print('Se puede frenar el robot poniendo torques y velocidades en cero a través del comando de frenado.')
    robotx.SetBrakesOn()
    print('Posición medida luego de aplicar frenos: ', robotx.GetSensorGPS_Compass())
    print('Medición de la IMU luego de aplicar frenos: ', robotx.GetSensorIMU())
    print('\n')
    
    print('Por último se muestra como agregar variables definidas por el usuario a la clase robot.')
    robotx.variable = "Beep beep!"
    print(robotx.variable)
    print('\n')

    print('Fin del ejemplo!')
    
if __name__ == '__main__': main()