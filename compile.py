import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch
from ultralytics import YOLO

def download_model(model_name: str, save_path: Path):
    """
    Descarga el modelo YOLOv8n.pt utilizando la API de Ultralytics.
    """
    try:
        print(f"Descargando el modelo {model_name}...")
        model = YOLO(model_name)
        model.to('cpu')  # Asegurarse de que el modelo esté en CPU antes de exportar
        pt_path = save_path / f"{model_name}.pt"
        model.save(pt_path)
        print(f"Modelo descargado y guardado en {pt_path}")
        return pt_path
    except Exception as e:
        print(f"Error al descargar el modelo: {e}")
        sys.exit(1)

def export_to_onnx(pt_model_path: Path, onnx_model_path: Path):
    """
    Exporta el modelo PyTorch a formato ONNX usando la API de Ultralytics.
    """
    try:
        print(f"Exportando el modelo {pt_model_path} a ONNX...")
        model = YOLO(pt_model_path)
        # Definir las opciones de exportación
        export_options = {
            "format": "onnx",
            "simplify": True,
            "opset": 14,
            "include": ["onnx"],
            "verbose": False
        }
        # Exportar el modelo
        model.export(format="onnx", simplify=True, opset=14, verbose=False)
        exported_onnx = Path("yolov8n.onnx")
        if exported_onnx.exists():
            exported_onnx.rename(onnx_model_path)
            print(f"Modelo exportado a ONNX y guardado en {onnx_model_path}")
        else:
            print("Exportación a ONNX falló: archivo .onnx no encontrado.")
            sys.exit(1)
    except Exception as e:
        print(f"Error al exportar a ONNX: {e}")
        sys.exit(1)

def convert_onnx_to_engine(onnx_model_path: Path, engine_path: Path, precision: str = "fp16"):
    """
    Convierte el modelo ONNX a TensorRT .engine usando trtexec.
    Requiere que trtexec esté en el PATH.
    
    :param onnx_model_path: Ruta al modelo ONNX.
    :param engine_path: Ruta donde se guardará el modelo TensorRT.
    :param precision: Precisión para la conversión ('fp32', 'fp16', 'int8').
    """
    try:
        print(f"Convirtiendo {onnx_model_path} a TensorRT .engine con precisión {precision}...")
        # Verificar si trtexec está disponible
        result = subprocess.run(["trtexec", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: trtexec no está disponible. Asegúrate de que TensorRT esté instalado y que trtexec esté en tu PATH.")
            sys.exit(1)
        
        # Construir el comando trtexec
        trtexec_cmd = [
            "trtexec",
            f"--onnx={onnx_model_path}",
            f"--saveEngine={engine_path}",
            f"--{precision}"
        ]
        
        # Ejecutar trtexec
        print(f"Ejecutando comando: {' '.join(trtexec_cmd)}")
        process = subprocess.run(trtexec_cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"Error en trtexec:\n{process.stderr}")
            sys.exit(1)
        
        print(f"Conversión a TensorRT .engine completada y guardada en {engine_path}")
    except Exception as e:
        print(f"Error al convertir a TensorRT: {e}")
        sys.exit(1)

def main():
    # Definir nombres y rutas
    model_name = "yolov8n"  # Nombre del modelo en la API de Ultralytics
    save_directory = Path("./models")
    save_directory.mkdir(parents=True, exist_ok=True)
    
    pt_model_path = save_directory / f"{model_name}.pt"
    onnx_model_path = save_directory / f"{model_name}.onnx"
    engine_model_path = save_directory / f"{model_name}.engine"
    
    # Paso 1: Descargar el modelo YOLOv8n.pt
    if not pt_model_path.exists():
        download_model(model_name, save_directory)
    else:
        print(f"El modelo PyTorch ya existe en {pt_model_path}. Saltando descarga.")
    
    # Paso 2: Exportar a ONNX
    if not onnx_model_path.exists():
        export_to_onnx(pt_model_path, onnx_model_path)
    else:
        print(f"El modelo ONNX ya existe en {onnx_model_path}. Saltando exportación.")
    
    # Paso 3: Convertir ONNX a TensorRT .engine
    if not engine_model_path.exists():
        convert_onnx_to_engine(onnx_model_path, engine_model_path, precision="fp16")
    else:
        print(f"El modelo TensorRT .engine ya existe en {engine_model_path}. Saltando conversión.")

if __name__ == "__main__":
    main()
