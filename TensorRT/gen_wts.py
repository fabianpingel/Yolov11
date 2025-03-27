import sys  # noqa: F401
import argparse
import os
import struct
import torch
import logging

# Konfiguration für das Logging
logging.basicConfig(
    level=logging.INFO,  # Setzt das Logging-Level auf INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format für Log-Meldungen
    datefmt="%Y-%m-%d %H:%M:%S"  # Datumsformat für die Logs
)

# Logger-Instanz erstellen
logger = logging.getLogger(__name__)

def parse_args() -> tuple:
    """
    Analysiert die Kommandozeilenargumente.

    Returns:
        tuple: Tupel mit Pfad zur Gewichtsdatei (.pt), Ausgabedatei (.wts) und Modelltyp.
    """
    parser = argparse.ArgumentParser(description='Konvertiert .pt Datei nach .wts')
    parser.add_argument('-w', '--weights', required=True,
                        help='Eingabepfad zur Gewichtsdatei (.pt) (erforderlich)')
    parser.add_argument(
        '-o', '--output', help='Ausgabepfad zur Datei (.wts) (optional)')
    parser.add_argument(
        '-t', '--type', type=str, default='detect', choices=['detect', 'cls', 'seg', 'pose'],
        help='bestimmt den Modelltyp: Detection/Classification/Segmentation/Pose')
    args = parser.parse_args()

    # Überprüfen, ob die Gewichtsdatei existiert
    if not os.path.isfile(args.weights):
        raise FileNotFoundError('Ungültige Eingabedatei')

    # Festlegen der Ausgabedatei, wenn nicht angegeben
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')

    return args.weights, args.output, args.type


try:
    pt_file, wts_file, m_type = parse_args()

    logger.info(f'Generiere .wts für das {m_type}-Modell')

    # Modell laden
    logger.info(f'Lade {pt_file}')

    # Gerät festlegen
    device = 'cpu'

    # Modell laden und auf FP32 konvertieren
    #model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
    # Aenderung FP 20.03.2025
    # PyTorch 2.6 changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. 
    model = torch.load(pt_file, map_location=device, weights_only=False)['model'].float()  # load to FP32

    if m_type in ['detect', 'seg', 'pose']:
        anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]

        delattr(model.model[-1], 'anchors')

    model.to(device).eval()
    
    # .wts-Datei schreiben
    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
            
    logger.info(f'{wts_file} erfolgreich erstellt.')
    logger.info(f'Fertig!')

except Exception as e:
    logger.error(f'Fehler beim Konvertieren der Datei: {e}')

