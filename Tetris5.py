import os
import signal
import sys

import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback, EveryNTimesteps
)

from functools import partial

# --- LR constante (pickleable) ---
def constant_lr(progress_remaining: float, value: float) -> float:
    # Ignora el progreso; mantiene LR constante
    return float(value)

# ========= Config =========
CKPT_DIR = "checkpoints"
BASE_NAME = "dqn_tetris"
N_ENVS_TRAIN = 12
N_ENVS_EVAL = 4
TOTAL_STEPS = 50_000_000  # horizonte de entrenamiento

# ========= Callbacks utilitarios =========
class EpsilonLogger(BaseCallback):
    def __init__(self, print_freq: int = 2000, verbose=0):
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
    """Factory picklable para SubprocVecEnv (nivel módulo)."""
    def _thunk():
        env = Tetris(render_mode=None)
        env = Monitor(env)
        env.reset(seed=42 + seed_offset)
        return env
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

    # =============== Modelo ===============
    BATCH_SIZE = 2048 if DEVICE == "cuda" else 64  # ajusta según VRAM

    loaded = False
    try:
        model = DQN.load(BASE_NAME, env=train_env, device=DEVICE, verbose=1)
        loaded = True
        print("Modelo existente cargado")
    except Exception:
        print("Creando nuevo modelo")
        model = DQN(
            "MultiInputPolicy",
            train_env,
            device=DEVICE,
            learning_rate=5e-5,            # valor que fijaremos también como constante
            buffer_size=200_000,
            learning_starts=5_000,
            batch_size=BATCH_SIZE,
            gamma=0.99,
            train_freq=8,
            gradient_steps=4,
            target_update_interval=30_000,
            max_grad_norm=10.0,
            exploration_initial_eps=0.20,
            exploration_fraction=1.00,
            exploration_final_eps=0.02,
            optimize_memory_usage=False,   # DictReplayBuffer no soporta True
            verbose=1,
            seed=42,
            tensorboard_log="tb_logs"
        )
        print("Modelo creado")

    # === Forzar LR constante = 5e-5 (tanto si cargaste como si creaste) ===
    model.lr_schedule = partial(constant_lr, value=5e-5)  # pickle-safe
    model.learning_rate = model.lr_schedule               # para logging interno
    # Actualizar optimizer en caliente
    if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
        for g in model.policy.optimizer.param_groups:
            g["lr"] = 5e-5
    print("[lr] fijado a 5e-5")

    # =============== Callbacks ===============
    # Checkpoint cada 2M timesteps totales (usando envoltura EveryNTimesteps)
    ckpt_inner = CheckpointCallback(
        save_freq=1,                         # lo controla EveryNTimesteps
        save_path=CKPT_DIR,
        name_prefix="dqn_tetris_postreheat",
        save_replay_buffer=True
    )
    checkpoint_every_2m = EveryNTimesteps(n_steps=2_000_000, callback=ckpt_inner)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CKPT_DIR,
        log_path=CKPT_DIR,
        eval_freq=200_000,
        deterministic=True,
        render=False
    )
    epslog_cb = EpsilonLogger(print_freq=2000)

    callbacks = CallbackList([checkpoint_every_2m, eval_cb, epslog_cb])

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
    print("\nIniciando Entrenamiento")
    print("Presiona Ctrl+C para interrumpir si es necesario")
    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
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
