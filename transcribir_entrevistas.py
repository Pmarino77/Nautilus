#!/usr/bin/env python3
"""Pipeline de transcripción de grupos focales y entrevistas con WhisperX.

Pensado para audios largos en español, en una GPU de 12 GB (RTX 5070).

Modos:
  grupo      (por defecto) diarización con min_speakers=3, max_speakers=8
  entrevista 1 a 1: min_speakers=2, max_speakers=2

Ejemplos:
  python transcribir_entrevistas.py sesiones/                      # grupos focales
  python transcribir_entrevistas.py sesion3.mp3 --max-speakers 6   # sabes que son 6
  python transcribir_entrevistas.py audio.mp3 --modo entrevista    # dos hablantes

Requisitos:
  pip install whisperx
  La RTX 5070 (Blackwell) requiere PyTorch compilado con CUDA 12.8+:
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
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

# min/max hablantes por modo; acotar el rango mejora mucho la diarización
MODOS = {
    "grupo": (3, 8),
    "entrevista": (2, 2),
}


def liberar(modelo) -> None:
    """Libera VRAM entre etapas: en 12 GB no caben los tres modelos a la vez con large-v3."""
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

    # TXT legible con hablante y marca de tiempo por turno
    lineas = []
    hablante_previo = None
    for seg in segmentos:
        hablante = seg.get("speaker", "DESCONOCIDO")
        texto = seg["text"].strip()
        if hablante != hablante_previo:
            lineas.append(f"\n[{formato_tiempo(seg['start'])[:8]}] [{hablante}]")
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
    parser.add_argument("--modo", choices=MODOS, default="grupo",
                        help="'grupo' (focal, por defecto) o 'entrevista' (2 hablantes)")
    parser.add_argument("--modelo", default="large-v3")
    parser.add_argument("--idioma", default="es", help="Código ISO; 'auto' para detectar")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="8 va holgado en 12 GB; bajar a 4 si hay OOM")
    parser.add_argument("--no-diarize", action="store_true", help="Desactivar diarización")
    parser.add_argument("--min-speakers", type=int, default=None, help="Anula el mínimo del modo")
    parser.add_argument("--max-speakers", type=int, default=None, help="Anula el máximo del modo")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    diarizar = not args.no_diarize
    if diarizar and not args.hf_token:
        parser.error("La diarización requiere un token de Hugging Face (HF_TOKEN o --hf-token)")

    min_def, max_def = MODOS[args.modo]
    min_speakers = args.min_speakers if args.min_speakers is not None else min_def
    max_speakers = args.max_speakers if args.max_speakers is not None else max_def
    if min_speakers > max_speakers:
        parser.error(f"min_speakers ({min_speakers}) no puede superar max_speakers ({max_speakers})")

    idioma = None if args.idioma == "auto" else args.idioma

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
    print(f"Modo: {args.modo} | idioma: {idioma or 'auto'}")
    if diarizar:
        print(f"Diarización: min_speakers={min_speakers}, max_speakers={max_speakers}")

    # Cada etapa carga su modelo, recorre todos los audios y libera VRAM antes de la
    # siguiente. El audio se recarga por etapa en vez de retenerlo: con sesiones de
    # 1-2 horas, mantener todos los arrays en RAM costaría varios GB por archivo.

    # 1) Transcripción (backend faster-whisper con batching)
    model = whisperx.load_model(args.modelo, device, compute_type=compute_type, language=idioma)
    resultados = {}
    for ruta in audios:
        print(f"\n→ Transcribiendo {ruta.name}")
        audio = whisperx.load_audio(str(ruta))
        resultados[ruta] = model.transcribe(audio, batch_size=args.batch_size)
        del audio
    liberar(model)

    # 2) Alineación por palabra (necesaria para asignar hablantes con precisión)
    align_cache = {}
    for ruta, result in resultados.items():
        lang = result["language"]
        if lang not in align_cache:
            align_cache[lang] = whisperx.load_align_model(language_code=lang, device=device)
        model_a, metadata = align_cache[lang]
        print(f"→ Alineando {ruta.name}")
        audio = whisperx.load_audio(str(ruta))
        resultados[ruta] = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        )
        del audio
    for model_a, _ in align_cache.values():
        liberar(model_a)

    # 3) Diarización y asignación de hablantes
    if diarizar:
        diarize_model = cargar_diarizador(args.hf_token, device)
        for ruta, result in resultados.items():
            print(f"→ Diarizando {ruta.name}")
            audio = whisperx.load_audio(str(ruta))
            diarize_segments = diarize_model(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )
            resultados[ruta] = whisperx.assign_word_speakers(diarize_segments, result)
            del audio
        liberar(diarize_model)

    for ruta, result in resultados.items():
        destino = salida / ruta.stem
        guardar_salidas(result, destino)
        print(f"✓ {ruta.name} → {destino}.{{txt,srt,json}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
