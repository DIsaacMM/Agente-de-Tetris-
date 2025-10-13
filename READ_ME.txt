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



# Opcionales

# Establecer Python 3.9 como predeterminado
py -3.9 -m pip install --upgrade pip

# Ahora pip usará Python 3.9 automáticamente
pip install gymnasium stable-baselines3 tetris-gymnasium numpy torch



Nota: Para detener el codigo pulsar Ctrl + C