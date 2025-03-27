"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from pathlib import Path
from typing import Any, List, Tuple
from utils import load_coco_labels, get_img_path_batches, plot_one_box 
import config # Variablen importieren 
import logging


# Konfiguration für das Logging
logging.basicConfig(
    level=logging.INFO,  # Setzt das Logging-Level auf INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format für Log-Meldungen
    datefmt="%Y-%m-%d %H:%M:%S"  # Datumsformat für die Logs
)
# Logger-Instanz erstellen
logger = logging.getLogger(__name__)


# Lade COCO-Kategorien aus einer Datei
coco_labels_path = Path.cwd() / "datasets" / "coco.names"  # Datei muss im gleichen Verzeichnis sein oder absoluten Pfad nutzen
categories = load_coco_labels(coco_labels_path)
if not categories:
    logger.warning("Warnung: Keine COCO-Labels geladen. Überprüfe die Datei.")
else:
    logger.info(f"Kategorien: {categories}")    



class YoLo11TRT(object):
    """
    description: A YOLO11 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        input_binding_names = []
        output_binding_names = []

        for binding_name in engine:
            shape = engine.get_tensor_shape(binding_name)
            print('binding_name:', binding_name, shape)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            # Append to the appropriate list.
            if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                input_binding_names.append(binding_name)
                self.input_w = shape[-1]
                self.input_h = shape[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            elif engine.get_tensor_mode(binding_name) == trt.TensorIOMode.OUTPUT:
                output_binding_names.append(binding_name)
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
            else:
                print('unknow:', binding_name)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.input_binding_names = input_binding_names
        self.output_binding_names = output_binding_names
        self.batch_size = engine.get_tensor_shape(input_binding_names[0])[0]
        print('batch_size:', self.batch_size)
        self.det_output_length = host_outputs[0].shape[0]

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        input_binding_names = self.input_binding_names
        output_binding_names = self.output_binding_names
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.set_tensor_address(input_binding_names[0], cuda_inputs[0])
        context.set_tensor_address(output_binding_names[0], cuda_outputs[0])
        context.execute_async_v3(stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * self.det_output_length: (i + 1) * self.det_output_length], batch_origin_h[i],
                batch_origin_w[i]
            )
            # Draw rectangles and labels on the original image
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                plot_one_box(
                    box,
                    batch_image_raw[i],
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[j])], result_scores[j]
                    ),
                )
        return batch_image_raw, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0]
            y[:, 2] = x[:, 2]
            y[:, 1] = x[:, 1] - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 3] - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 2] - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1]
            y[:, 3] = x[:, 3]
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        num_values_per_detection = config.DET_NUM + config.SEG_NUM + config.POSE_NUM
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        # pred = np.reshape(output[1:], (-1, 38))[:num, :]
        pred = np.reshape(output[1:], (-1, num_values_per_detection))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=config.CONF_THRESH, nms_thres=config.IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = (np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None)
                      * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None))
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes



class InferThread(threading.Thread):
    """
    Diese Klasse führt die Inferenz mit dem YOLO11-Modell in einem separaten Thread durch.
    Sie verarbeitet einen Batch von Bildern, speichert die Ergebnisse und gibt die Laufzeit aus.
    """
    
    def __init__(self, yolo11_wrapper: Any, image_path_batch: List[str]) -> None:
        """
        Initialisiert den Inferenz-Thread.

        :param yolo11_wrapper: Eine Instanz des YOLO11-Objekts zur Durchführung der Inferenz.
        :param image_path_batch: Liste mit Pfaden zu den Eingabebildern.
        """
        super().__init__()  # Initialisierung der Thread-Klasse
        self.yolo11_wrapper = yolo11_wrapper
        self.image_path_batch = image_path_batch

    def run(self) -> None:
        """
        Führt die Inferenz für einen Batch von Bildern durch und speichert die Ergebnisse.
        """
        try:
            # Lade Bilder und führe die Inferenz durch
            batch_image_raw, use_time = self.yolo11_wrapper.infer(self.yolo11_wrapper.get_raw_image(self.image_path_batch))

            for i, img_path in enumerate(self.image_path_batch):
                try:
                    parent, filename = os.path.split(img_path)
                    save_name = os.path.join('output', filename)

                    # Speichern des inferierten Bildes
                    if batch_image_raw[i] is not None:
                        cv2.imwrite(save_name, batch_image_raw[i])
                    else:
                        logger.warning(f"Warnung: Keine gültige Inferenz für {img_path}, Bild wird nicht gespeichert.")

                except Exception as e:
                    logger.error(f"Fehler beim Speichern des Bildes {img_path}: {e}")

            logger.info(f"Eingaben -> {self.image_path_batch}, Zeit -> {use_time * 1000:.2f} ms, Bilder gespeichert in 'output/'")

        except AttributeError as e:
            logger.error(f"Fehler: Die YOLO-Wrapper-Instanz hat nicht die erwarteten Methoden. {e}")

        except FileNotFoundError as e:
            logger.error(f"Fehler: Eine der Eingabedateien wurde nicht gefunden. {e}")

        except Exception as e:
            logger.error(f"Unerwarteter Fehler während der Inferenz: {e}")



class WarmUpThread(threading.Thread):
    """
    Diese Klasse führt das Warm-up für das YOLO-Modell in einem separaten Thread aus.
    Das Warm-up hilft, die Laufzeit des ersten Inferenzdurchlaufs zu optimieren.
    """
    
    def __init__(self, yolo11_wrapper: Any) -> None:
        """
        Initialisiert den Warm-up-Thread.

        :param yolo11_wrapper: Eine Instanz des YOLO11-Objekts, das die Inferenz ausführt.
        """
        super().__init__()  # Ersetzt `threading.Thread.__init__(self)`
        self.yolo11_wrapper = yolo11_wrapper

    def run(self) -> None:
        """
        Führt das Warm-up durch, indem eine leere Eingabe an das Modell übergeben wird.
        Dies sorgt dafür, dass TensorRT das Modell lädt und optimiert, bevor echte Bilder verarbeitet werden.
        """
        try:
            # Erstelle ein Dummy-Bild (Nullmatrix) und führe die Inferenz durch
            batch_image_raw, use_time = self.yolo11_wrapper.infer(self.yolo11_wrapper.get_raw_image_zeros())

            # Ausgabe der Ergebnisse
            logger.info(f"Warm-up -> {batch_image_raw[0].shape}, Zeit -> {use_time * 1000:.2f} ms")

        except AttributeError as e:
            logger.error(f"Fehler: Die YOLO-Wrapper-Instanz hat nicht die erwarteten Methoden. {e}")

        except Exception as e:
            logger.error(f"Unerwarteter Fehler während des Warm-ups: {e}")



def main() -> None:
    """
    Hauptfunktion zum Laden von TensorRT-Plugins, Initialisieren des YOLO-Modells
    und Ausführen von Inferenz auf Bildern.
    """
    try:
        # Lade benutzerdefiniertes Plugin und Engine-Datei
        PLUGIN_LIBRARY = Path.cwd() / "TensorRT" / "build" / "Release" / "myplugins.dll"
        logger.info(PLUGIN_LIBRARY)
        engine_file_path = Path.cwd() / "TensorRT" / "build" / "yolo11n.engine"
        logger.info(engine_file_path)

        # Falls Kommandozeilenargumente übergeben wurden, verwende diese
        if len(sys.argv) > 1:
            engine_file_path = Path(sys.argv[1])
        if len(sys.argv) > 2:
            PLUGIN_LIBRARY = Path(sys.argv[2])

        # Lade das Plugin
        if not PLUGIN_LIBRARY.exists():
            raise FileNotFoundError(f"Plugin-Datei nicht gefunden: {PLUGIN_LIBRARY}")
        ctypes.CDLL(str(PLUGIN_LIBRARY))
        
        # Output-Verzeichnis vorbereiten
        output_dir = Path("output")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()
        
        # YOLO-Modell initialisieren
        if not engine_file_path.exists():
            raise FileNotFoundError(f"Engine-Datei nicht gefunden: {engine_file_path}")
        
        yolo11_wrapper = YoLo11TRT(str(engine_file_path))

        try:
            logger.info(f'Batch Size: {yolo11_wrapper.batch_size}')

            image_dir = Path.cwd() / "TensorRT" / "images"
            image_path_batches = get_img_path_batches(yolo11_wrapper.batch_size, str(image_dir))

            # Warm-up Durchläufe für das Modell
            for _ in range(10):
                warmup_thread = WarmUpThread(yolo11_wrapper)
                warmup_thread.start()
                warmup_thread.join()
                
            # Inferenz durchführen   
            for batch in image_path_batches:
                infer_thread = InferThread(yolo11_wrapper, batch)
                infer_thread.start()
                infer_thread.join()
        
        finally:
            # YOLO-Instanz zerstören, um Ressourcen freizugeben
            yolo11_wrapper.destroy()
            
    except Exception as e:
        logger.error(f"Fehler aufgetreten: {e}")
    
    logger.info("Fertig!")



if __name__ == "__main__":
    main()
