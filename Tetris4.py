import os
import signal
import sys

import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, BaseCallback, EvalCallback
)
from stable_baselines3.common.vec_env import SubprocVecEnv

# ========= Helpers =========
CKPT_DIR = "checkpoints"
BASE_NAME = "dqn_tetris"
N_ENVS_TRAIN = 12
N_ENVS_EVAL  = 4
RB_PATH = os.path.join(CKPT_DIR, "rb_before_reheat.pkl")  # replay buffer temporal para transferencia

class ActionCounter(BaseCallback):
    def __init__(self, print_every=10000, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self.counts = None
    def _on_training_start(self) -> None:
        n = self.training_env.action_space.n
        self.counts = [0]*n
    def _on_step(self) -> bool:
        # última acción tomada por cada env vectorizado
        actions = self.locals.get("actions", None)
        if actions is not None:
            for a in actions:
                self.counts[int(a)] += 1
        if self.model.num_timesteps % self.print_every == 0:
            total = sum(self.counts) or 1
            dist = [c/total for c in self.counts]
            print(f"[act] steps={self.model.num_timesteps:,} dist={dist}")
        return True


class EpsilonLogger(BaseCallback):
    """Imprime epsilon y timesteps cada print_freq calls."""
    def __init__(self, print_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._n = 0
    def _on_step(self) -> bool:
        self._n += 1
        if self._n % self.print_freq == 0:
            eps = getattr(self.model, "exploration_rate", None)
            if eps is not None:
                print(f"[dbg] steps={self.model.num_timesteps:,}  epsilon={eps:.4f}")
        return True

def make_env(seed_offset=0):
    """Factory picklable para SubprocVecEnv (debe estar a nivel módulo)."""
    def _thunk():
        e = Tetris(render_mode=None)
        e = Monitor(e)
        e.reset(seed=42 + seed_offset)
        return e
    return _thunk

def main():
    # ======== CUDA / DEVICE ========
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        USE_CUDA = torch.cuda.is_available()
        DEVICE = "cuda" if USE_CUDA else "cpu"
        if USE_CUDA:
            torch.backends.cudnn.benchmark = True
            print(f"[INFO] CUDA disponible: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        else:
            print("[INFO] CUDA no disponible. Usando CPU.")
    except Exception:
        DEVICE = "cpu"
        print("[WARN] PyTorch no encontrado o error al consultar CUDA. Usando CPU.")

    print("Entrenamiento Agente DQN para TETRIS")

    # =============== Entornos (multiproceso) ===============
    os.makedirs(CKPT_DIR, exist_ok=True)
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS_TRAIN)])
    eval_env  = SubprocVecEnv([make_env(100 + i) for i in range(N_ENVS_EVAL)])
    print("Entornos Tetris creados (SubprocVecEnv)")
    print(f" - Entrenamiento: {N_ENVS_TRAIN} envs")
    print(f" - Evaluación:    {N_ENVS_EVAL} envs")
    print(f"[INFO] Dispositivo de entrenamiento: {DEVICE}")

    # =============== Recalentamiento de exploración (finetune) ===============
    BATCH_SIZE = 1024 if DEVICE == "cuda" else 64  # aprovecha GPU

    # 1) Intentamos cargar el modelo actual como base
    base_model = None
    try:
        base_model = DQN.load(BASE_NAME, env=train_env, device=DEVICE, verbose=1)
        print("[REHEAT] Modelo base cargado correctamente.")
    except Exception as e:
        print(f"[REHEAT] No se pudo cargar {BASE_NAME}.zip, se entrenará desde cero. Detalle: {e}")

    # 2) Creamos SIEMPRE un modelo nuevo con schedule de exploración “recalentado”
    #    (si no hay base_model, este será tu modelo de arranque)
    print("[REHEAT] Creando nuevo modelo con epsilon más alto y LR más bajo (finetune).")
    new_model = DQN(
        "MultiInputPolicy",
        train_env,
        device=DEVICE,
        learning_rate=3e-5,              # LR más suave para finetune estable
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=BATCH_SIZE,
        gamma=0.99,
        train_freq=8,                    # más pasos por update en envs paralelos
        gradient_steps=4,                # más trabajo consecutivo en GPU
        target_update_interval=10_000,
        max_grad_norm=10.0,
        # Recalentamos exploración:
        exploration_initial_eps=0.30,    # sube epsilon para explorar rotaciones
        exploration_fraction=0.10,       # cae rápido al 10% del horizonte de learn()
        exploration_final_eps=0.02,
        optimize_memory_usage=False,
        verbose=1,
        seed=42
    )

    # 3) Si había modelo base, copiamos PESOS y REPLAY BUFFER
    if base_model is not None:
        try:
            # Copiar pesos (Q y target)
            new_model.set_parameters(base_model.get_parameters())
            print("[REHEAT] Pesos transferidos del modelo base.")

            # Intentamos transferir el replay buffer
            # (SB3 recomienda usar save/load para asegurar compatibilidad)
            try:
                base_model.save_replay_buffer(RB_PATH)
                print(f"[REHEAT] Replay buffer del modelo base guardado en {RB_PATH}.")
                new_model.load_replay_buffer(RB_PATH)
                print("[REHEAT] Replay buffer cargado en el nuevo modelo.")
            except Exception as e_rb:
                print(f"[REHEAT] No se pudo transferir el replay buffer: {e_rb}")
        except Exception as e_param:
            print(f"[REHEAT] No se pudieron transferir parámetros: {e_param}")

    # A partir de aquí entrenamos siempre con el modelo “recalentado”
    model = new_model

    # =============== Callbacks ===============
    checkpoint_cb = CheckpointCallback(
        save_freq=1_000_000,               # checkpoint cada 1M pasos (menos I/O)
        save_path=CKPT_DIR,
        name_prefix=BASE_NAME,
        save_replay_buffer=True
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CKPT_DIR,     # guarda checkpoints/best_model.zip
        log_path=CKPT_DIR,
        eval_freq=100_000,                 # eval cada 100k pasos
        deterministic=True,
        render=False
    )
    epslog_cb = EpsilonLogger(print_freq=2000)
    action_counter_cb = ActionCounter(print_every=50_000)
    callbacks = CallbackList([checkpoint_cb, eval_cb, epslog_cb, action_counter_cb])

    # =============== Manejador Ctrl+C ===============
    def handle_sigint(sig, frame):
        print("\n[WARN] Interrupción detectada. Guardando snapshot...")
        try:
            model.save(f"{BASE_NAME}_interrupted")
            print(f"[OK] Guardado: {BASE_NAME}_interrupted.zip")
        finally:
            try:
                train_env.close()
                eval_env.close()
            finally:
                sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # =============== Entrenamiento ===============
    print("\nIniciando Entrenamiento (REHEAT / finetune)")
    print("Presiona Ctrl+C para interrumpir si es necesario")
    try:
        # Nota: no reseteamos timesteps para mantener contadores estables
        model.learn(
            total_timesteps=10_000_000,    # 5–10M es buen finetune para salir del óptimo local
            callback=callbacks,
            log_interval=100,
            reset_num_timesteps=False
        )
        model.save(BASE_NAME)
        print(f"Entrenamiento completado y modelo guardado en {BASE_NAME}.zip")
    except KeyboardInterrupt:
        print("Entrenamiento interrumpido por el usuario")
        model.save(BASE_NAME)
        print(f"Modelo guardado (entrenamiento interrumpido): {BASE_NAME}.zip")
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
    finally:
        train_env.close()
        eval_env.close()
        print("Entornos cerrados.")

# ======== Arranque seguro para Windows ========
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
