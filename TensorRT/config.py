# config.py - Konfigurationsvariablen für das YOLO-Modell
from typing import Final


# Konfidenz-Schwellenwert für die Erkennung (nur Objekte mit höherer Wahrscheinlichkeit als dieser Wert werden akzeptiert)
CONF_THRESH: Final[float] = 0.5  
# Intersection over Union (IoU) Schwellenwert für Non-Maximum Suppression (NMS)
IOU_THRESHOLD: Final[float] = 0.4  
# Anzahl der Keypoints für die Posen-Schätzung (17 Keypoints * 3 Werte pro Punkt: x, y, confidence)
POSE_NUM: Final[int] = 17 * 3  
# Anzahl der Parameter für jede Detektion (z. B. x, y, Breite, Höhe, Konfidenz, Klassen-ID)
DET_NUM: Final[int] = 6  
# Anzahl der Segmente für die Segmentierung (z. B. Masken-Punkte oder Features)
SEG_NUM: Final[int] = 32 