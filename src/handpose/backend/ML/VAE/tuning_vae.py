import logging

import optuna
import torch
import torch.optim as optim
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader, random_split

from handpose.config_loader import get_settings

from . import train_vae
from .dataset_vae import HandPoseDataset
from .model_vae import HandPoseVae

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.Trial,
    WINDOW_SIZE=None,
    STRIDE=None,
    DEVICE=None,
    BATCH_SIZE=None,
    FEATURES_DIR_TRAINING=None,
    VALIDATION_SPLIT=None,
    N_TRIALS=None,
) -> float:
    vae_cfg = get_settings().vae
    WINDOW_SIZE = WINDOW_SIZE if WINDOW_SIZE is not None else vae_cfg.window_size
    STRIDE = STRIDE if STRIDE is not None else vae_cfg.stride
    DEVICE = DEVICE if DEVICE is not None else vae_cfg.actual_device
    BATCH_SIZE = BATCH_SIZE if BATCH_SIZE is not None else vae_cfg.batch_size
    FEATURES_DIR_TRAINING = (
        FEATURES_DIR_TRAINING
        if FEATURES_DIR_TRAINING is not None
        else vae_cfg.paths.get("features_dir_training")
    )
    VALIDATION_SPLIT = (
        VALIDATION_SPLIT if VALIDATION_SPLIT is not None else vae_cfg.validation_split
    )

    # Hyperparameter
    window_size = WINDOW_SIZE
    stride = STRIDE
    batch_size = BATCH_SIZE
    epochs = 15

    # Architektur & Daten
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)

    # Training
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    kl_weight = trial.suggest_float("kl_weight", 1e-6, 1e-3, log=True)

    # Datenset
    dataset = HandPoseDataset(
        features_dir=FEATURES_DIR_TRAINING, window_size=window_size, stride=stride
    )

    if len(dataset) < 2 * batch_size:
        logger.warning(
            f"Trial {trial.number}: Zu wenig Daten ({len(dataset)}). Pruned."
        )
        raise optuna.exceptions.TrialPruned()

    # Daten in Trainings- und Validierungssets aufteilen
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )

    # Eingabe-Dimension aus dem Dataset ableiten
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[1]

    # Modell initialisieren
    model = HandPoseVae(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=window_size,
        dropout=dropout,
    ).to(DEVICE)

    # AdamW als Optimizer verwenden
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        for inputs, _ in train_loader:
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(inputs)

            # Loss mit Optuna-KL-Weight
            loss = train_vae.vae_loss_function(
                recon_batch, inputs, mu, logvar, kl_weight
            )

            loss.backward()
            optimizer.step()

        # Validierung
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(DEVICE)
                recon, mu, logvar = model(inputs)
                loss = train_vae.vae_loss_function(recon, inputs, mu, logvar, kl_weight)
                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)

        # Optuna Reporting & Pruning
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return best_val_loss


def run_optuna_study():
    vae_cfg = get_settings().vae
    DEVICE = vae_cfg.actual_device
    N_TRIALS = vae_cfg.n_trials

    # Pruner Setup
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0, interval_steps=1)

    study = optuna.create_study(
        direction="minimize", study_name="HandPose_VAE_Tuning", pruner=pruner
    )

    logger.info(f"Starte Optimierung auf {DEVICE}...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    logger.info(f"\n--- Beste Parameter ---\nValue (Loss): {study.best_value}")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")
