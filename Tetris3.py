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
                learning_rate=5e-5,          # más suave, reduce pérdida alta
                buffer_size=200000,          # mejor aprovechamiento de RAM
                learning_starts=5000,        # empieza a entrenar con más datos
                batch_size=64,               # más estable, aprovecha RAM
                gamma=0.99,
                exploration_fraction=0.2,    # baja el epsilon más lento
                exploration_final_eps=0.02,  # final un poco mayor, evita atascarse
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
    model.save("dqn_tetris")
    print("Modelo guardado (entrenamiento interrumpido) ")
except Exception as e:
    print(f"Error en entrenamiento: {e} ")

env.close()