from ultralytics import YOLO
import torch
import sys
import argparse
import logging

# Konfiguration für das Logging
logging.basicConfig(
    level=logging.INFO,  # Setzt das Logging-Level auf INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format für Log-Meldungen
    datefmt="%Y-%m-%d %H:%M:%S"  # Datumsformat für die Logs
)

# Logger-Instanz erstellen
logger = logging.getLogger(__name__)

def load_model(model_path: str) -> YOLO:
    """
    Lädt ein YOLO-Modell von einem gegebenen Pfad.

    Args:
        model_path (str): Der Pfad zum Modell.

    Returns:
        YOLO: Das geladene YOLO-Modell.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    
    try:
        model = YOLO(model_path).to(device)
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        sys.exit(1)
    
    return model

def validate_model(model: YOLO) -> None:
    """
    Validiert das Modell und gibt Metriken aus.

    Args:
        model (YOLO): Das geladene YOLO-Modell.

    Returns:
        None
    """
    try:
        metrics = model.val()
        logger.info(f"Validierungsmetriken: {metrics}")
    except Exception as e:
        logger.error(f"Fehler bei der Validierung des Modells: {e}")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Modell Validierungs Script")
    parser.add_argument("--model_path", default="runs/detect/train/weights/best.pt", help="Pfad zum Modell")
    args = parser.parse_args()

    model = load_model(args.model_path)
    validate_model(model)

if __name__ == "__main__":
    main()
