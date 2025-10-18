#Necesidades basicas para correr codigo

Version de python necesaria: 3.9.13

Instrucciones: La version Tetis3.py es la final, esta nomas entrena al modelo y Tetris_prueba.py muestra a la maquina jugando el juego. 
Todos los demas codigos entrenan el modelo y luego lo muestran jugando, sin embargo para mejores propositos se separo el entrenamiento 
de la visualizacion. El modelo se ejecuta sin visualizacion porque se entrena mas rapido y consume menos recursos de la computadora. 


# Verificar Python 3.9
py -3.9 --version

# Instalar librerías con Python 3.9
py -3.9 -m pip install gymnasium stable-baselines3 tetris-gymnasium numpy torch

# Ejecutar tu código con Python 3.9
py -3.9 "c:/Users/david/Documents/Machine Learning/Proyecto Agente de Tetris/Tetris1.py"
py -3.9 "c:/Users/david/Documents/Machine Learning/Proyecto Agente de Tetris/Tetris2.py"
py -3.9 "c:/Users/david/Documents/Machine Learning/Proyecto Agente de Tetris/Tetris3.py"
py -3.9 "c:/Users/david/Documents/Machine Learning/Proyecto Agente de Tetris/Tetris_prueba.py"

# Opcionales

# Establecer Python 3.9 como predeterminado
py -3.9 -m pip install --upgrade pip

# Ahora pip usará Python 3.9 automáticamente
pip install gymnasium stable-baselines3 tetris-gymnasium numpy torch



Nota: Para detener el codigo pulsar Ctrl + C (o Ctrl + F2 si utiliza PyCharm como editor)




Specs utilizados:

CPU --> Ryzen 5 5600x
GPU --> NVIDIA 4070 Super 12GB
RAM --> 16GB 3600MHz
DISK --> 1TB HDD

#Descripciones Generales de Cada Codigo:

Tetris --> Codigo de entrenamiento que utiliza GPU de NVIDIA y RAM al maximo.

Tetris2 --> Codigo de entrenamiento la cual no utiliza GPU de NVIDIA
Mas simple que los demas codigos.

Tetris3 --> Codigo de entramiento que utiliza GPU de NVIDIA de manera moderada
Tetris3 es muy parecido a Tetris, sin embargo este no acapara todos los recursos de la computadora.

Tetris4 --> Codigo de entrenamiento que utiliza GPU de NVIDIA al maximo y "refresca" al agente

Este codigo es identico a Tetris en cuanto a la utilizacion de recursos, sin embargo este contiene una
funcion "REHEAT", la cual realiza una copia de un agente entranado y lo fuerza a que gire las piezas en sus primeros
movimientos para que luego aprenda de estos nuevos movimientos.
(SOLO UTILIZAR CUANDO EL AGENTE NO MUESTRA CAMBIOS DE COMPORTAMIENTO DESPUES DE LARGOS INTERVALOS DE ENTRENAMIENTO)

Tetris5 (Opcional) --> Codigo completamente identico que Tetris, unica diferencia son los checkpoints cambian de nombre despues de REHEAT.
Debido a cambios del modelo del paremetro "learning_rate" de 3e-5 de vuelta a 5e-5 es recomendable vigilar el parametro "loss" para que no se dispare,
mantener copia del agente REHEAT terminado por si ocurre perdida de aprendizaje.

Tetris_Prueba:
Toma al agente actual "dqn_tetris.zip" y corre una renderizacion de lo que ha aprendido el agente hasta el momento,
esto codigo NO entrena al agente solo vizualiza lo aprendido.