# Las paredes son listas de segmentos entre dos puntos
walls = [# Outer walls
         #[(,),(,)],
         [(-15,-12),(-15,12)],
         [(-15,12),(15,12)],
         [(15,12),(15,-12)],
         [(15,-12),(-15,-12)]
        ]
         
# Las puertas son listas de puntos donde se ubican las puertas
doors = [(0,3),  # Altar Stellarium
         (10,5), # Craddle of Krhyno
         (7, -8)]# Cave of the Magician