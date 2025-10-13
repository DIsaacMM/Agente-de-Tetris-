import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

print("Entrenamiento Agente DQN para TETRIS ")

# Entrenamiento

# Crear entorno sin renderizado
env = DummyVecEnv([lambda: Tetris(render_mode=None)])
print("Entorno de Tetris creado")

# Inicializar modelo DQN
model = DQN("MlpPolicy", env, verbose=1)

# Entrenar el modelo
print("Entrenando modelo (100,000 pasos)...")
model.learn(total_timesteps=100000)
print("Entrenamiento completado ")

# Guardar modelo
model.save("dqn_tetris")
print("Modelo guardado como 'dqn_tetris.zip'")

# === PRUEBA DEL MODELO ENTRENADO ===
print("\n=== PROBANDO MODELO ENTRENADO ===")

# Crear entorno con renderizado para ver el juego
test_env = Tetris(render_mode='human')

# Cargar el modelo entrenado
model = DQN.load("dqn_tetris")

# Reiniciar entorno
obs, info = test_env.reset()

# Ejecutar 1000 pasos para probar
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, info = test_env.reset()
