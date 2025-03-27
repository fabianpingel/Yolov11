# 🚀 1. Vorbereitungen

### 1.1. Python installieren
Version 3.12.9: https://www.python.org/downloads/release/python-3129/

⚠️ Add python.exe to path anwählen!

### 1.2. Anaconda installieren
Download Version für Python 3.12: https://www.anaconda.com/download/success

### 1.3. CMake installieren
Download Version 3.31.6: https://cmake.org/download/

⚠️ Add CMAKE to the PATH environment variable anwählen!

Prüfen, ob CMake richtig installiert ist mit  ```cmake --version```

### 1.4. Visual Studio 2022 installieren/updaten
Version 17.13.4

Desktopentwicklung mit C++ muss installiert sein! (optional: Python-Entwicklung)

Dauert ein wenig... Zeit für einen ☕!

### 1.5. OpenCV 4.11.0 herunterladen
OpenCV: https://github.com/opencv/opencv.git

zusätzlich extra Module herunterladen:

OpenCV Contrib: https://github.com/opencv/opencv_contrib.git

⚠️ Versionen müssen zueinander passen (selbsterklärend...!)

### 1.6. OpenCV Downloads in folgende Verzeichnisstruktur entpacken
```
    C:\
    └── OpenCV\
        ├── build\
        ├── opencv_contrib-4.11.0\
        └── opencv-4.11.0\
```

### 1.7. NVIDIA CUDA Toolkit installieren
Download CUDA Toolkit 12.6.3: https://developer.nvidia.com/cuda-toolkit-archive

Download cuDNN 9.8.0: https://developer.nvidia.com/cudnn-downloads

### 1.8. Systemumgebungsvariable setzen/prüfen
Unter Systemvariablen sollte 

```CUDA_PATH --> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6```

zu finden sein.

Außerdem unter ```Path --> C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6``` hinzufügen

### 1.9. Rechner neu starten
Anschließend Eingabeaufforderung öffnen und folgendes prüfen:

```
python -V
``` 
✅ Python 3.12.9

```
nvcc -V
``` 
✅ Build cuda_12.6.r12.6/compiler.35059454_0

```
nvidia-smi
``` 
✅ Driver Version 561.17 & Cuda Version: 12.6

### 1.10. PIP upgraden
```
python.exe -m pip install --upgrade pip
```

### 1.11. PyTorch installieren
https://pytorch.org/get-started/locally/

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Hierbei wird gleich die richtige `numpy` Version installiert. 

Wenn die Installation abgeschlossen ist, können die Pakete mit `pip list` überprüft werden.

Hier sollten sich wiederfinden:
```
numpy         2.1.2
torch         2.6.0+cu126
torchaudio    2.6.0+cu126
torchvision   0.21.0+cu126
```

#### Optional: Python Installation überprüfen 
```
python -c "import torch; print(torch.cuda.is_available())"
```
 --> True, wenn CUDA(GPU) verfügbar


# 💻👁️ 2. OpenCV-DNN-Moduls mit CUDA-Backend-Unterstützung einrichten
🥳 Jetzt beginnt der spaßige Teil...!!!

### 2.1 CMake öffen
Source Code Verzeichnis wählen:  ```C:/opencv/opencv-4.11.0```

Build Verzeichnis wählen:  ```C:/opencv/build```

`Configure` Button drücken

Jetzt müssen iterativ Optionen angewählt werden. Hierzu zunächst das Häkchen bei `Advanced` und `Grouped` setzen.

Folgene Optionen anwählen (können unter Search gesucht werden):
- CMAKE_CONFIGURATION_TYPES --> (hier kann Debug entfernt werden, so dass nur release bleibt)
- WITH_CUDA
- OPENCV_DNN_CUDA
- ENABLE_FAST_MATH
- BUILD_opencv_world
- OPENCV_EXTRA_MODULES_PATH definieren --> C:/opencv/opencv_contrib-4.11.0/modules (entweder in der Zeile gaaaanz rechts über das kleine, fast unsichtbare Kästchen mit den drei Punkten oder Pfad manuell reinkopieren. Auf '/' achten!)

Danach erneut `Configure` Button drücken

Jetzt tauchen neue Optionen (rot hinterlegt) auf. Folgende Optionen hinzufügen:
- CUDA_FAST_MATH
- CUDA_ARCH_BIN --> für RTX 3080 >= 8.6 drin lassen
Compute Capability je GPU kann hier https://developer.nvidia.com/cuda-gpus nachgesehen werden.

Außerdem tauchen ein paar Fehler auf, da die cuDNN Libraries nicht gefunden werden. Diese kann man manuell setzen (sofern die Option 'advanced' aktiviert ist)
- CUDNN_LIBRARY: C:/Program Files/NVIDIA/CUDNN/v9.8/lib/12.8/x64/cudnn.lib

Und wieder erneut `Configure` Button drücken.

Zuletzt noch das 'Include'-Verzeichnis angeben und nicht gefundene Video CVodecs entfernen

- CUDNN_INCLUDE_DIR: C:/Program Files/NVIDIA/CUDNN/v9.8/include/12.8
- WITH_NVCUVID=OFF
- WITH_NVCUVENC=OFF

Jetzt sollte die Configuration fehlerfrei sein.

#### Python Support (optional)
Notwendige Python-Pfade können mit folgendem Befehl herausgefunden werden:
```
python -c "from sysconfig import get_paths; info = get_paths(); print('\n'.join(f'{k}: {v}' for k, v in info.items()))"
````
 Für Python anschließend folgende Pfade angeben:
 - PYTHON3_EXECUTABLE: C:/Users/<user_name>>/AppData/Local/Programs/Python/Python312/python.exe 
 (kann in Konsole mit Befehl ```where python``` heausgefunden werden)
- PYTHON3_INCLUDE_DIR: C:/Users/<user_name>>/AppData/Local/Programs/Python/Python312/include 
- PYTHON3_LIBRARY: C:/Users/<user_name>/AppData/Local/Programs/Python/Python312/libs/python312.lib
- PYTHON3_NUMPY_INCLUDE_DIRS: C:/Users/<user_name>/AppData/Local/Programs/Python/Python312/Lib/site-packages/numpy/_core/include 
(```python -c "import numpy as np; print(np.get_include())"```)
- PYTHON3_PACKAGES_PATH: C:/Users/<user_name>/AppData/Local/Programs/Python/Python312/Lib/site-packages


Abschließend `Generate` drücken.

### 2.2 Visual Studio öffen

Im build-Verzeichnis `OpenCV.sln` mit Visual Studio öffnen.

`Release` und `x64` unter Projektmappenkonfiguration/-plattformen anwählen.

`CMakeTargets` im Projektmappen-Explorer öffnen und
- `ALL_BUILD` sowie anschließend
- `INSTALL`

erstellen. Der erste Vorgang dauert je nach Hardware >1h. 

Genügend Zeit, um das Mittagessen 🍲 vorzubereiten 

###### 🥳 Herzlichen Glückwunsch! Es ist vollbracht 🎉


# 3. Github Repo klonen
```
git clone https://github.com/fabianpingel/Yolov11.git
```


# 4. Ultralytics installieren
https://docs.ultralytics.com/de/models/yolo11/

### 4.1. virtuelle Umgebung erzeugen
```
python -m venv env
```
und aktivieren mit
```
env\Scripts\activate
```

Jetzt sollte ein `(env)` vor jeder Kommendozeile stehen. 

⚠️ Alle weiteren Schritt im geklonten, lokalen Repo durchführen oder im Terminal des Code Editors.

### 4.2 PIP upgraden 
(falls nicht bereits oben geschehen)
```
python.exe -m pip install --upgrade pip
```

### 4.3. Requirements installieren
```
pip install -r requirements.txt
```

### 4.4 PyTorch in virtueller Umgebung installieren 
(ich weiß, ist doppelt, aber notwendig...)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 4.5 Torchvision
Torchvision muss für GPU support manuell nachinstalliert werden!

Download latest python whl: https://download.pytorch.org/whl/torchvision/

```
pip install <PATH-TO-TORCHVISION.WHL>\torchvision-0.21.0+cu126-cp312-cp312-win_amd64.whl
```

Mit `pip list` prüfen, ob die '+cu126' Build-Variante hinter den torch-Versionen steht.

### 4.6. Modelle herunterladen
Die Yolo-Modelle können je nach Aufgabe [hier](https://docs.ultralytics.com/de/models/yolo11/#supported-tasks-and-modes) heruntergeladen werden.

- Detect: https://docs.ultralytics.com/de/tasks/detect/
- Segment: https://docs.ultralytics.com/de/tasks/segment/#models

Es bietet sich an folgende Ordnerstruktur für die Modelle anzulegen:

```
    Yolov11:\
    └── env\
    └── models\
        ├── detect\
        |       └── yolo11n.pt
        |       └── yolo11n.onnx      
        ├── segment\
        |       └── yolo11n-seg.pt
        |       └── yolo11n-seg.onnx  
```

### 4.7. Testen
Im Verzeichnis `Ultralytics` befinden sich Python-Skripte, um Training, Export und Inferenz zu testen.
```
    Yolov11:\
    └── env\
    └── models\
    └── Ultralytics\
        ├── export.py
        ├── predict.py
        ├── train.py
        ├── validate.py
```

# 5. NVIDIA | TensorRT

## 5.1. TensorRT installieren

Download TensorRT 10.9 zip-Package für Windows (GA Version für 'General Availability'!): https://developer.nvidia.com/tensorrt/download/10x

Entpacken und Inhalt von *TensorRT-10.9.0.34* unter ```C:\Program Files\NVIDIA``` kopieren.

TensorRT Bibliotheksdateien zum Systempfad *Path* als Umgebungsvariable hinzufügen. 

```
C:\Program Files\NVIDIA\TensorRT-10.9.0.34\lib
```

Weitere Infos unter: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html



## 5.2 Anwenden der Engine

Infos zur Einrichtung und Anwendung der Engine siehe gesonderte [readme.md](./TensorRT/readme.md)


### 🚧 Weitere Abschnitte sind noch im Aufbau...!



