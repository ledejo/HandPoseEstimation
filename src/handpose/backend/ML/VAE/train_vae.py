import logging
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from handpose.config_loader import get_absolute_path, get_settings

from .dataset_vae import HandPoseDataset
from .model_vae import HandPoseVae

logger = logging.getLogger(__name__)


def vae_loss_function(recon_x, x, mu, logvar, KL_WEIGHT: float):
    """
    VAE Verlustfunktion: Rekonstruktionsverlust + KL-Divergenz

    Args:
        recon_x: Rekonstruierte Eingaben
        x: Originale Eingaben
        mu: Mittelwert aus dem Encoder
        logvar: Log-Varianz aus dem Encoder
        KL_WEIGHT: Gewichtung der KL-Divergenz im Verlust

    Returns:
        Gesamter Verlustwert
    """
    MSE = F.mse_loss(recon_x, x, reduction="mean")
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (KL_WEIGHT * KLD)


def train(
    MODEL_SAVE_PATH=None,
    WINDOW_SIZE=None,
    STRIDE=None,
    LEARNING_RATE=None,
    EPOCHS=None,
    BATCH_SIZE=None,
    PATIENCE=None,
    DEVICE=None,
    FEATURES_DIR_TRAINING=None,
    HIDDEN_DIM=None,
    LATENT_DIM=None,
    DROPOUT=None,
    VALIDATION_SPLIT=None,
):
    # Lade Config-Defaults falls nicht übergeben
    vae_cfg = get_settings().vae
    MODEL_SAVE_PATH = (
        MODEL_SAVE_PATH
        if MODEL_SAVE_PATH is not None
        else get_absolute_path(vae_cfg.model_save_path)
    )
    WINDOW_SIZE = WINDOW_SIZE if WINDOW_SIZE is not None else vae_cfg.window_size
    STRIDE = STRIDE if STRIDE is not None else vae_cfg.stride
    LEARNING_RATE = (
        LEARNING_RATE if LEARNING_RATE is not None else vae_cfg.learning_rate
    )
    EPOCHS = EPOCHS if EPOCHS is not None else vae_cfg.epochs
    BATCH_SIZE = BATCH_SIZE if BATCH_SIZE is not None else vae_cfg.batch_size
    PATIENCE = PATIENCE if PATIENCE is not None else vae_cfg.patience
    DEVICE = DEVICE if DEVICE is not None else vae_cfg.actual_device
    FEATURES_DIR_TRAINING = (
        FEATURES_DIR_TRAINING
        if FEATURES_DIR_TRAINING is not None
        else get_absolute_path(vae_cfg.paths.get("features_training"))
    )
    HIDDEN_DIM = HIDDEN_DIM if HIDDEN_DIM is not None else vae_cfg.hidden_dim
    LATENT_DIM = LATENT_DIM if LATENT_DIM is not None else vae_cfg.latent_dim
    DROPOUT = DROPOUT if DROPOUT is not None else vae_cfg.dropout
    VALIDATION_SPLIT = (
        VALIDATION_SPLIT if VALIDATION_SPLIT is not None else vae_cfg.validation_split
    )
    KL_WEIGHT = vae_cfg.kl_weight
    # os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Training VAE auf: {DEVICE} | Konfiguration: (Window: {WINDOW_SIZE}, LR: {LEARNING_RATE:.2e})"
    )

    # Dataset
    full_dataset = HandPoseDataset(
        FEATURES_DIR_TRAINING,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    if len(full_dataset) == 0:
        logger.error("Abbruch: Das Dataset ist leer.")
        return

    # Splitte das Dataset in Trainings- und Validierungsdatensätze basierend auf dem Validierungsanteil
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Lade die Daten in den DataLoader
    # -> Num_workers=2 verhindert Freezes auf Windows/CPU
    # -> Pin_memory=True beschleunigt Datenübertragung zur GPU und False bei CPU
    # -> Persistent_workers=True hält die Worker-Prozesse aktiv
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
        persistent_workers=True,
    )

    # Input Dimension bestimmen
    sample_x, _ = full_dataset[0]
    input_dim = sample_x.shape[1]

    # Modell initialisieren
    model = HandPoseVae(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        seq_len=WINDOW_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)
    # Optimizer initialisieren -> Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    logger.info(f"Starte VAE Training (max. {EPOCHS} Epochen, Patience {PATIENCE})...")

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for inputs, _ in train_loader:
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = vae_loss_function(recon, inputs, mu, logvar, KL_WEIGHT)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validierung des Modells
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(DEVICE)
                recon, mu, logvar = model(inputs)
                loss = vae_loss_function(recon, inputs, mu, logvar, KL_WEIGHT)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Logging & Checkpointing
        log_msg = f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Fehler {math.sqrt(avg_val_loss)}"

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_improve = 0
            logger.info(log_msg + " -> Aktuell bestes Modell gespeichert.")
        else:
            logger.info(log_msg)
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info("Early Stopping.")
                break

    logger.info(
        f"Fertig! Bester Validation Loss: {best_val_loss:.6f}\nDas beste Modell liegt unter: {MODEL_SAVE_PATH}"
    )
