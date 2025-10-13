import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Prueba con interfaz visual
print("\n Iniciando Prueba Visual ")
try:
    env_test = Tetris(render_mode='human')
    model = DQN.load("dqn_tetris")
    obs, info = env_test.reset()
    
    print("Presiona Ctrl+C para detener la prueba ")
    games_played = 0
    
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        
        if terminated or truncated:
            games_played += 1
            print(f"Juego {games_played} terminado - Reiniciando...")
            obs, info = env_test.reset()
            
except KeyboardInterrupt:
    print("\n Prueba interrumpida por el usuario")
except Exception as e:
    print(f"Error en prueba: {e}")
finally:
    env_test.close()
    print("Programa finalizado ")