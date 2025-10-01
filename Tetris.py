import gymnasium as gym
import tetris_gymnasium
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# Callback para monitorear el progreso
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(rewards)
                print(f"Step {self.n_calls}, Mean Reward: {mean_reward:.2f}")
        return True

def main():
    print("ENTRENAMIENTO AGENTE TETRIS ")
    
    # 1. Crear entorno de Tetris
    
    env = gym.make('Tetris-v1', render_mode='None', obs_type='ram')
    
    print(f"Espacio de observación: {env.observation_space}")
    print(f"Espacio de acciones: {env.action_space}")
    
    # 2. Crear modelo DQN
    print("\nInicializando modelo DQN...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        verbose=1
    )
    
    # 3. Callback para monitoreo
    callback = ProgressCallback(check_freq=5000)
    
    # 4. Entrenar el modelo
    print("\n Iniciando entrenamiento...")
    print("Presiona Ctrl+C para interrumpir")
    
    try:
        model.learn(total_timesteps=100000, callback=callback)
        
        # Guardar modelo final
        model.save("dqn_tetris_final")
        print("Modelo final guardado: dqn_tetris_final")
        
    except KeyboardInterrupt:
        print("\n Entrenamiento interrumpido por usuario")
        model.save("dqn_tetris_interrupted")
        print("Modelo guardado: dqn_tetris_interrupted")
    
    # 5. Cerrar entorno
    env.close()
    
    # 6. Evaluar el modelo
    print("\n Evaluando modelo entrenado...")
    evaluate_model()


def evaluate_model(model_path ="dqn_tetris_final", num_episodes=3):
    
    # Crear entorno con renderizado
    eval_env = gym.make('Tetris-v1', render_mode='human', obs_type='ram')
    
    try:
        # Cargar modelo
        model = DQN.load(model_path)
        print(f" Modelo cargado: {model_path}")
        
        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\n--- Episodio {episode + 1} ---")
            
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    lines_cleared = info.get('lines_cleared', 0)
                    print(f"Recompensa: {total_reward:.1f}, "
                          f"Líneas: {lines_cleared}, "
                          f"Pasos: {steps}")
                    break
        
    except FileNotFoundError:
        print(f" Modelo no encontrado: {model_path}")
    finally:
        eval_env.close()


main()