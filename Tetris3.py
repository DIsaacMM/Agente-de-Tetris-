import gymnasium as gym                                     # Entrenamiento de modelo por refuerzo
from tetris_gymnasium.envs.tetris import Tetris             # Ambiente de tetris creado para gymnasium
from stable_baselines3 import DQN                           # Libreria para usar algoritmo DQN
from stable_baselines3.common.vec_env import DummyVecEnv    # Convertir el ambiente en matrices

print("Entrenamiento Agente DQN para TETRIS ")

# Crear entorno de entrenamiento (sin renderizado)
env = Tetris(render_mode=None)
print("Entorno Tetris creado ")
print(f" - Acciones posibles: {env.action_space.n}")            # Entrega el numero de acciones que puede hacer el modelo que son 8
print(f" - Espacio de observación: {env.observation_space}")    # Entrega cómo se estructura la observación del entorno


# Creacion del modelo si es que no existe
try:
    model = DQN.load("dqn_tetris", env=env, verbose=1)
    print("Modelo existente cargado")
except:
    print("Creando nuevo modelo ")
    model = DQN("MultiInputPolicy", env, 
                learning_rate=1e-4,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                exploration_fraction=0.1,
                exploration_final_eps=0.01,
                verbose=1)
    print("Modelo Creado")

# Entrenamiento
print("\n Iniciando Entrenamiento ")
print("Presiona Ctrl+C para interrumpir si es necesario")

try:
    model.learn(total_timesteps=50000, log_interval=100)
    model.save("dqn_tetris")
    print("Entrenamiento completado y modelo guardado ")
except KeyboardInterrupt:
    print("Entrenamiento interrumpido por el usuario ")
    model.save("dqn_tetris_interrupted ")
    print("Modelo guardado (entrenamiento interrumpido) ")
except Exception as e:
    print(f"Error en entrenamiento: {e} ")

env.close()