import gymnasium as gym
import tetris_gymnasium
from stable_baselines3 import DQN

# Entrenar

# Crea el entorno de Tetris sin renderizado (más rápido para entrenar)
env = gym.make('Tetris-v1', render_mode='None')

# Inicializa el modelo DQN (Deep Q-Network):
# - "MlpPolicy": Usa una red neuronal de múltiples capas (perceptrón)
# - env: El entorno donde se entrenará
# - verbose=1: Muestra información del progreso durante el entrenamiento
model = DQN("MlpPolicy", env, verbose=1)

# Entrena el modelo por 100,000 pasos de tiempo
# Cada "timestep" representa una acción tomada en el entorno
model.learn(total_timesteps=100000)

# Guarda el modelo entrenado para usarlo después
model.save("dqn_tetris")


# Probar

# Crea un nuevo entorno PERO ahora con renderizado humano para ver el juego
env = gym.make('Tetris-v1', render_mode='human')

# Carga el modelo previamente entrenado
model = DQN.load("dqn_tetris")


# Reinicia el entorno y obtiene la primera observación (estado del juego)
obs, info = env.reset()

# Ejecuta 1000 pasos de juego para probar el modelo
for _ in range(1000):
    # El modelo predice la mejor acción basándose en la observación actual
    # deterministic=True: Siempre elige la acción con mayor Q-value
    action, _ = model.predict(obs, deterministic=True)

    # Ejecuta la acción en el entorno y obtiene:
    # - obs: Nuevo estado del juego
    # - reward: Recompensa obtenida por la acción
    # - terminated: Si el juego terminó (game over)
    # - truncated: Si el episodio fue truncado (límite de tiempo)
    # - info: Información adicional del entorno
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Si el juego terminó o fue truncado, reinicia el entorno
    if terminated or truncated:
        obs, info = env.reset()