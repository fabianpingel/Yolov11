import os
import random
import cv2
import numpy as np
from typing import List, Optional, Tuple



def load_coco_labels(file_path: str) -> List[str]:
    """
    Lädt die COCO-Labels aus einer Datei, in der jedes Label in einer Zeile steht.
    
    :param file_path: Pfad zur Datei mit COCO-Labels.
    :return: Liste der Labels.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            labels = [line.strip() for line in file if line.strip()]
        return labels
    except FileNotFoundError:
        print(f"Fehler: Die Datei {file_path} wurde nicht gefunden.")
        return []
    except Exception as e:
        print(f"Fehler beim Laden der COCO-Labels: {e}")
        return []
    
    
    
def get_img_path_batches(batch_size: int, img_dir: str) -> List[List[str]]:
    """
    Durchsucht das angegebene Verzeichnis rekursiv nach Bilddateien und teilt sie in Batches auf.

    :param batch_size: Anzahl der Bilder pro Batch.
    :param img_dir: Pfad zum Verzeichnis mit den Bildern.
    :return: Liste von Batches, wobei jeder Batch eine Liste von Bildpfaden enthält.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Fehler: batch_size muss eine positive ganze Zahl sein.")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Fehler: Das Verzeichnis '{img_dir}' existiert nicht oder ist nicht zugänglich.")

    batches: List[List[str]] = []  # Liste zur Speicherung der Batches
    batch: List[str] = []  # Temporäre Liste für den aktuellen Batch

    # Durchsucht das Verzeichnis rekursiv nach Dateien
    for root, _, files in os.walk(img_dir):
        for name in sorted(files):  # Sortiert die Dateien für eine konsistente Reihenfolge
            file_path = os.path.join(root, name)
            
            # Wenn der aktuelle Batch voll ist, wird er gespeichert und eine neue Liste gestartet
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
            
            batch.append(file_path)

    # Falls noch Bilder übrig sind, den letzten Batch hinzufügen
    if batch:
        batches.append(batch)

    return batches



def plot_one_box(
    x: List[int],
    img: np.ndarray,
    color: Optional[Tuple[int, int, int]] = None,
    label: Optional[str] = None,
    line_thickness: Optional[int] = None
) -> None:
    """
    Zeichnet eine Bounding Box auf ein Bild.

    :param x: Liste mit den Koordinaten der Bounding Box [x1, y1, x2, y2].
    :param img: OpenCV-Bild (NumPy-Array), auf das die Bounding Box gezeichnet wird.
    :param color: Farbe der Bounding Box als (R, G, B)-Tupel, z. B. (0, 255, 0) für Grün.
    :param label: Text, der über der Bounding Box angezeigt wird (optional).
    :param line_thickness: Dicke der Bounding-Box-Linie (optional).
    :return: Keine Rückgabe. Die Funktion verändert das Bild direkt.
    """

    # Fehlerbehandlung für ungültige Eingaben
    if not isinstance(x, list) or len(x) != 4:
        raise ValueError("Fehler: Die Bounding Box-Koordinaten müssen eine Liste mit genau 4 Werten sein.")

    if not isinstance(img, np.ndarray):
        raise TypeError("Fehler: Das Bild muss ein NumPy-Array sein (OpenCV-Format).")

    if color is not None and (not isinstance(color, tuple) or len(color) != 3):
        raise ValueError("Fehler: Die Farbe muss ein Tupel mit drei Werten (R, G, B) sein.")

    # Bestimme die Dicke der Linien, falls nicht angegeben
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

    # Falls keine Farbe angegeben wurde, wähle eine zufällige Farbe
    color = color or tuple(random.randint(0, 255) for _ in range(3))

    # Definiere die Eckpunkte der Bounding Box
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    # Zeichne das Rechteck
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        # Bestimme die Schriftgröße und -dicke
        tf = max(tl - 1, 1)  # Minimale Linienstärke = 1
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 3, thickness=tf)[0]
        
        # Definiere die Position der Textbox
        label_pos = (c1[0], c1[1] - 2)
        text_bg_pos = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

        # Zeichne eine gefüllte Box hinter den Text für bessere Lesbarkeit
        cv2.rectangle(img, c1, text_bg_pos, color, -1, cv2.LINE_AA)

        # Zeichne den Label-Text in Weiß
        cv2.putText(
            img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, tl / 3, (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA
        )
