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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')

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

def predict(model: YOLO, image_path: str) -> None:
    """
    Führt eine Vorhersage auf dem gegebenen Bild durch und speichert die Ergebnisse.

    Args:
        model (YOLO): Das vorab geladene YOLO-Modell.
        image_path (str): Der Pfad zum Eingabebild.

    Returns:
        None
    """
    try:
        results = model(image_path)
        for result in results:
            result.show()
            result.save(filename="result.jpg")
    except Exception as e:
        logger.error(f"Fehler bei der Vorhersage: {e}")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Vorhersage Script")
    parser.add_argument("--task", choices=["seg", "det"], default='det', help="Modelltyp: 'seg' für Segmentierung, 'det' für Detektion")
    parser.add_argument("--size", choices=["n", "s", "m", "l", "x"], default="n", help="Modellgröße: n, s, m, l oder x (Standard: n)")
    parser.add_argument("--image_path", default="datasets/bus.jpg", help="Pfad zum Eingabebild")
    args = parser.parse_args()

    model = load_model(task=args.task, size=args.size)
    predict(model, args.image_path)

if __name__ == "__main__":
    main()
