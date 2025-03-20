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

# Device erkennen und setzen
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def load_model(task: str='det', size: str='n') -> YOLO:
    """
    Lädt das YOLO-Modell basierend auf dem gegebenen Task (Segmentierung oder Detektion) und der Modellgröße.

    Args:
        task (str): Der Task-Typ, entweder 'seg' für Segmentierung oder 'det' für Detektion.
        size (str): Die Modellgröße, entweder 'n', 's', 'm', 'l' oder 'x'.

    Returns:
        YOLO: Das geladene YOLO-Modell.
    """
    if size not in {"n", "s", "m", "l", "x"}:
        logger.error(f"Ungültige Modellgröße: {size}. Erlaubt sind: n, s, m, l, x.")
        sys.exit(1)

    # Ordnerpfad abhängig vom Task setzen
    model_folder = "segment" if task == "seg" else "detect"

    # Modellname abhängig vom Task setzen
    model_suffix = "-seg" if task == "seg" else ""
    model_path = f"models/{model_folder}/yolo11{size}{model_suffix}.pt"

    try:
        model = YOLO(model_path).to(device)
        logger.info(f"Modell erfolgreich geladen: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells '{model_path}': {e}")
        sys.exit(1)

def train_model(model: YOLO, data: str, epochs: int, imgsz: int) -> None:
    """
    Trainiert das YOLO-Modell mit einem Datensatz.

    Args:
        model (YOLO): Das geladene YOLO-Modell.
        data (str): Der Pfad zur Datensatz-Konfigurationsdatei.
        epochs (int): Anzahl der Trainings-Epochen.
        imgsz (int): Bildgröße für das Training.

    Returns:
        None
    """
    try:
        results = model.train(data=data, epochs=epochs, imgsz=imgsz, device=[0] if "cuda" in device else "cpu", workers=0)
        logger.info(f"Training abgeschlossen: {results}")
    except Exception as e:
        logger.error(f"Fehler beim Training des Modells: {e}")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Modell Training Script")
    parser.add_argument("--task", choices=["seg", "det"], default='det', help="Modelltyp: 'seg' für Segmentierung, 'det' für Detektion")
    parser.add_argument("--size", choices=["n", "s", "m", "l", "x"], default="n", help="Modellgröße: n, s, m, l oder x (Standard: n)")
    parser.add_argument("--data", type=str, default="coco8-seg.yaml", help="Pfad zur Datensatz-Konfigurationsdatei")
    parser.add_argument("--epochs", type=int, default=10, help="Anzahl der Trainings-Epochen (Standard: 10)")
    parser.add_argument("--imgsz", type=int, default=640, help="Bildgröße für das Training (Standard: 640)")
    args = parser.parse_args()

    # Modell laden
    model = load_model(task=args.task, size=args.size)

    # Modell trainieren
    train_model(model, data=args.data, epochs=args.epochs, imgsz=args.imgsz)

if __name__ == "__main__":
    main()