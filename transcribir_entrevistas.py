#!/usr/bin/env python3
"""Pipeline de transcripción de entrevistas de campo con WhisperX.

Configuración por defecto pensada para entrevistas 1 a 1:
  - modelo large-v3, compute_type=float16 en GPU (int8 en CPU)
  - diarización activada con min_speakers=2 y max_speakers=2

Si sabes que hay más voces, sube el máximo:
  python transcribir_entrevistas.py audio.mp3 --max-speakers 4

Requisitos:
  pip install whisperx
  Token de Hugging Face con acceso aceptado a pyannote/speaker-diarization-3.1
  (variable de entorno HF_TOKEN o flag --hf-token).
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import whisperx

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


def liberar(modelo) -> None:
    """Libera VRAM entre etapas para poder encadenar los tres modelos en una GPU modesta."""
    del modelo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cargar_diarizador(hf_token: str, device: str):
    # La clase cambió de módulo entre versiones de whisperx
    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError:
        DiarizationPipeline = whisperx.DiarizationPipeline
    return DiarizationPipeline(use_auth_token=hf_token, device=device)


def formato_tiempo(segundos: float) -> str:
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def guardar_salidas(result: dict, destino: Path) -> None:
    segmentos = result["segments"]

    # JSON completo (palabras, tiempos, hablantes) para análisis posterior
    destino.with_suffix(".json").write_text(
        json.dumps(segmentos, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # TXT legible con hablante por turno
    lineas = []
    hablante_previo = None
    for seg in segmentos:
        hablante = seg.get("speaker", "DESCONOCIDO")
        texto = seg["text"].strip()
        if hablante != hablante_previo:
            lineas.append(f"\n[{hablante}]")
            hablante_previo = hablante
        lineas.append(texto)
    destino.with_suffix(".txt").write_text("\n".join(lineas).strip() + "\n", encoding="utf-8")

    # SRT con marcas de tiempo y hablante
    bloques = []
    for i, seg in enumerate(segmentos, start=1):
        hablante = seg.get("speaker", "DESCONOCIDO")
        bloques.append(
            f"{i}\n{formato_tiempo(seg['start'])} --> {formato_tiempo(seg['end'])}\n"
            f"[{hablante}] {seg['text'].strip()}\n"
        )
    destino.with_suffix(".srt").write_text("\n".join(bloques), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entrada", help="Archivo de audio o carpeta con audios")
    parser.add_argument("-o", "--salida", default="transcripciones", help="Carpeta de salida")
    parser.add_argument("--modelo", default="large-v3")
    parser.add_argument("--idioma", default=None, help="Código ISO (ej. 'es'); si se omite, se detecta")
    parser.add_argument("--batch-size", type=int, default=16, help="Bajar a 8 o 4 si falta VRAM")
    parser.add_argument("--no-diarize", action="store_true", help="Desactivar diarización")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=2, help="Subir cuando haya más de 2 voces")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    diarizar = not args.no_diarize
    if diarizar and not args.hf_token:
        parser.error("La diarización requiere un token de Hugging Face (HF_TOKEN o --hf-token)")

    entrada = Path(args.entrada)
    if entrada.is_dir():
        audios = sorted(p for p in entrada.iterdir() if p.suffix.lower() in AUDIO_EXTS)
        if not audios:
            parser.error(f"No se encontraron audios en {entrada}")
    elif entrada.is_file():
        audios = [entrada]
    else:
        parser.error(f"No existe: {entrada}")

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Dispositivo: {device} | compute_type: {compute_type} | modelo: {args.modelo}")
    if diarizar:
        print(f"Diarización: min_speakers={args.min_speakers}, max_speakers={args.max_speakers}")

    # 1) Transcripción (backend faster-whisper con batching, mucho más rápido que transformers)
    model = whisperx.load_model(args.modelo, device, compute_type=compute_type, language=args.idioma)
    transcripciones = {}
    for ruta in audios:
        print(f"\n→ Transcribiendo {ruta.name}")
        audio = whisperx.load_audio(str(ruta))
        transcripciones[ruta] = model.transcribe(audio, batch_size=args.batch_size), audio
    liberar(model)

    # 2) Alineación por palabra (necesaria para asignar hablantes con precisión)
    resultados = {}
    align_cache = {}
    for ruta, (result, audio) in transcripciones.items():
        idioma = result["language"]
        if idioma not in align_cache:
            align_cache[idioma] = whisperx.load_align_model(language_code=idioma, device=device)
        model_a, metadata = align_cache[idioma]
        resultados[ruta] = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        ), audio
    for model_a, _ in align_cache.values():
        liberar(model_a)

    # 3) Diarización y asignación de hablantes
    if diarizar:
        diarize_model = cargar_diarizador(args.hf_token, device)
        for ruta, (result, audio) in resultados.items():
            print(f"→ Diarizando {ruta.name}")
            diarize_segments = diarize_model(
                audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers
            )
            resultados[ruta] = whisperx.assign_word_speakers(diarize_segments, result), audio
        liberar(diarize_model)

    for ruta, (result, _) in resultados.items():
        destino = salida / ruta.stem
        guardar_salidas(result, destino)
        print(f"✓ {ruta.name} → {destino}.{{txt,srt,json}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
