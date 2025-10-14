import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from stable_baselines3 import DQN
import cv2
import numpy as np
import time



print("\nIniciando prueba del modelo entrenado...")

try:
    # Inicialización del entorno y del modelo
    # Se crea el entorno de Tetris en modo visual (RGB)
    env_test = Tetris(render_mode='rgb_array')

    # Se carga el modelo DQN previamente entrenado (archivo "dqn_tetris.zip")
    model = DQN.load("dqn_tetris")

    # Reinicia el entorno y obtiene la primera observación
    obs, info = env_test.reset()

    print("Presiona Q o Ctrl+C para detener la prueba")
    games_played = 0  # Contador de juegos jugados

    # Bucle principal de simulación
    while True:
        # El modelo predice la mejor acción posible según su política entrenada
        action, _ = model.predict(obs, deterministic=True)

        # Se ejecuta la acción en el entorno y se obtienen los resultados
        obs, reward, terminated, truncated, info = env_test.step(action)

        # Renderiza el frame actual del entorno
        frame = env_test.render()

        # Si el frame existe, se convierte y se muestra en una ventana
        if frame is not None:
            # Convertir de RGB a BGR (formato que usa OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Mostrar el frame en una ventana de OpenCV
            cv2.imshow("Tetris - Agente DQN", frame)

            # Pausa para ralentizar la simulación 
            time.sleep(0.1)

            # Si el usuario presiona la tecla 'q', se detiene la simulación
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Si el juego termina (por perder o alcanzar un límite de pasos)
        if terminated or truncated:
            games_played += 1
            print(f"Juego {games_played} terminado - Reiniciando...")
            obs, info = env_test.reset()


# Manejo de interrupciones o errores
except KeyboardInterrupt:
    # Si el usuario presiona Ctrl+C, se interrumpe de forma segura
    print("\nPrueba interrumpida por el usuario.")
except Exception as e:
    # Captura cualquier error inesperado
    print(f"Error en prueba: {e}")

# Limpieza final (siempre se ejecuta)
finally:
    env_test.close()        # Cierra el entorno correctamente
    cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV
    print("Programa finalizado.")
