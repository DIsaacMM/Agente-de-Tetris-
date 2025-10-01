import gymnasium as gym
import tetris_gymnasium
from stable_baselines3 import DQN

# Entrenamiento 
env = gym.make('Tetris-v1', render_mode='None')

# INTELIGENTE: Intenta cargar modelo existente, sino crea uno nuevo
try:
    model = DQN.load("dqn_tetris", env=env, verbose=1)
    print("Modelo cargado, continuando entrenamiento...")
except:
    # Si no existe el archivo, crea un modelo nuevo
    model = DQN("MlpPolicy", env, verbose=1)
    print("Creando nuevo modelo...")

# Continúa el entrenamiento (añade 100,000 pasos más)
model.learn(total_timesteps=100000)
model.save("dqn_tetris")  # Guarda el modelo mejorado

# Prueba
env = gym.make('Tetris-v1', render_mode='human')
model = DQN.load("dqn_tetris")
obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()