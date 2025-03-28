import time
import argparse
from ultralytics import YOLO
import torch
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_model(task: str='det', size: str='n') -> YOLO:
    """
    Lädt das YOLO-Modell basierend auf dem gegebenen Task (Segmentierung oder Detektion) und der Modellgröße.
    
    Args:
        task (str): 'seg' für Segmentierung oder 'det' für Detektion.
        size (str): Modellgröße ('n', 's', 'm', 'l', 'x').
    
    Returns:
        YOLO: Das geladene Modell.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    model_folder = "segment" if task == "seg" else "detect"
    model_suffix = "-seg" if task == "seg" else ""
    model_path = f"models/{model_folder}/yolo11{size}{model_suffix}.pt"
    
    try:
        model = YOLO(model_path).to(device)
        logger.info(f"Modell erfolgreich geladen: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells '{model_path}': {e}")
        exit(1)

def benchmark(model: YOLO, image_path: str, runs: int = 10) -> None:
    """
    Testet die Inferenzzeit des YOLO-Modells über mehrere Durchläufe.
    
    Args:
        model (YOLO): Das vorab geladene YOLO-Modell.
        image_path (str): Der Pfad zum Eingabebild.
        runs (int): Anzahl der Durchläufe zur Mittelwertbildung.
    """
    logger.info(f"Starte Benchmark mit {runs} Durchläufen...")

    # CUDA-Warmup (1 Durchlauf ignorieren)
    model(image_path)
    torch.cuda.synchronize()

    times = []
    for i in range(runs):
        start_time = time.time()
        results = model(image_path)  # Vorhersage ausführen
        torch.cuda.synchronize()  # Sicherstellen, dass alles abgeschlossen ist
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000  # in ms
        times.append(elapsed_time)
        logger.info(f"Durchlauf {i+1}: {elapsed_time:.2f} ms")

    avg_time = sum(times) / len(times)
    logger.info(f"Durchschnittliche Inferenzzeit: {avg_time:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="YOLO Inferenz Benchmark")
    parser.add_argument("--task", choices=["seg", "det"], default='det', help="Modelltyp: 'seg' oder 'det'")
    parser.add_argument("--size", choices=["n", "s", "m", "l", "x"], default="n", help="Modellgröße: n, s, m, l oder x")
    parser.add_argument("--image_path", default="datasets/bus.jpg", help="Pfad zum Eingabebild")
    parser.add_argument("--runs", type=int, default=100, help="Anzahl der Durchläufe für das Benchmarking")
    args = parser.parse_args()
    
    model = load_model(task=args.task, size=args.size)
    benchmark(model, args.image_path, args.runs)

if __name__ == "__main__":
    main()
