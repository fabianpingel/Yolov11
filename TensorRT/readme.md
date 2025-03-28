## 1. Einführung

Coda in Anlehnung an: 
- [tensorrtx/yolo11](https://github.com/wang-xinyu/tensorrtx/tree/master/yolo11)
- [YOLOv10-TensorRT10](https://github.com/mpj1234/YOLOv10-TensorRT10)

Die Yolo11-Modelle unterstützten TensorRT-8, FP32/FP16/INT8 und Python/C++ API.

## 2. getestete Environment

* CUDA 12.6
* cuDNN 9.8.0
* TensorRT 10.9.0.34 / 8.6.1.6
* OpenCV 4.11.0.86
* Ultralytics 8.3.91

## 3. Build and Run

### 3.1 Konfiguration

* YOLO11-Modellgröße n/s/m/l/x aus den Kommandozeilenargumenten wählen.
* Weitere Konfigurationsmöglichkeiten unter [include/config.h](include/config.h).

### 3.2 *.wts Datei

1. .wts Dateien erzeugen (aus PyTorch mit .pt)

```shell
cd <PATH-TO-TENSORRT>

python gen_wts.py -w ../models/detect/yolo11n.pt -o yolo11n.wts -t det

python gen_wts.py -w ../models/segment/yolo11n-seg.pt -o yolo11n-seg.wts -t seg

# Hiermit wird eine Datei „yolo11n.wts“ erzeugt.
```

2. tensorrtx/yolo11 bauen und ausführen

```shell
mkdir build
cd build
cmake ..
make
```


### 3.3 Detection
```shell
cp [PATH-TO-WTS_File]/yolo11n.wts .

# Im build-Verzeichnis ausführen

# TensorRT-Engine aufbauen und serialisieren
Release\yolo11_det.exe -s ..\yolo11n.wts ..\yolo11n.engine n # [n/s/m/l/x]

# Inferenz ausführen
Release\yolo11_det.exe -d ..\yolo11n.engine ..\..\datasets c # [c/g]

# Ergebnisse werden im Build-Verzeichnis (aktuelles Arbeitsverzeichnis) gespeichert.
```



### 3.4 Segmentation
```shell
cp [PATH-TO-WTS_File]/yolo11n-seg.wts .

# Im build-Verzeichnis ausführen

# TensorRT-Engine aufbauen und serialisieren
Release\yolo11_seg.exe -s ..\yolo11n-seg.wts ..\yolo11n-seg.engine n # [n/s/m/l/x]

# Herunterladen der COCO-Labels
python -m wget -o ..\datasets\coco.names https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt

# Inferenz ausführen
Release\yolo11_seg.exe -d ..\yolo11n-seg.engine ..\..\datasets c coco.names # [c/g]
```



### 3.5 *Optional*, Tensorrt-Modells in Python Laden und Ausführen 

```shell
# tensorrt & pycuda müssen installiert sein
# sicherstellen, dass yolo11n.engine existiert

# Detection
python yolo11_det_trt.py .\build\yolo11n.engine .\build\Release\myplugins.dll

# Segmentation
python yolo11_seg_trt.py .\build\yolo11n-seg.engine .\build\Release\myplugins.dll
```

#### INT8 Quantization

1. Kalibrierungsbilder vorbereiten. 1000 Bilder können aus dem Trainingssatz zufällig auswählen werden. Für Coco können auch Kalibrierungsbilder `coco_calib` von [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) oder [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) heruntergeladen werden. (pwd: a9wh)
2. Entpacken nach yolo11/build
3. Makro `USE_INT8` in src/config.h setzen und erneut kompilieren
4. Modell serialisieren und testen

