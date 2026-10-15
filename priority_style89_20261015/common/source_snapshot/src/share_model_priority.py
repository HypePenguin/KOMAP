"""Verify or run the KOMAP priority style89 model on a folder of images."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'model'


def verify_files():
    manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise RuntimeError(f'Missing or changed file: {entry["path"]}')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != entry['sha256']:
            raise RuntimeError(f'Checksum mismatch: {entry["path"]}')
    print(f'PACKAGE_VERIFIED: {len(manifest["files"])} files', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'check-model', 'infer'])
    parser.add_argument('--input', type=Path, help='Image file or folder (non-recursive)')
    parser.add_argument('--output', type=Path, help='New output folder; existing folders are rejected')
    args = parser.parse_args()
    verify_files()
    if args.command == 'verify':
        return
    if args.command == 'infer' and (args.input is None or not args.input.exists()):
        parser.error('infer requires --input pointing to an image file or folder')
    import torch
    import experiment as core
    from PIL import Image
    import numpy as np
    torch.set_num_threads(4)
    config = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(RUN / 'best.pt', map_location='cpu', weights_only=False)
    model = core.make_model(config, pretrained=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    result = json.loads((RUN / 'result.json').read_text(encoding='utf-8'))
    if parameters != config['expected_parameters'] or checkpoint['epoch'] != result['best_epoch']:
        raise RuntimeError('Checkpoint architecture/epoch differs from recorded model')
    if checkpoint.get('model_kind') != 'EMA':
        raise RuntimeError('Expected the selected EMA model')
    if args.command == 'check-model':
        print(json.dumps({'model_loaded_on': 'cpu', 'parameters': parameters,
            'selected_epoch': checkpoint['epoch'], 'model_kind': checkpoint['model_kind']}))
        return
    if os.name != 'nt' or not torch.cuda.is_available():
        raise RuntimeError('This inference entry point requires Windows and CUDA-enabled NVIDIA PyTorch.')
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not files:
        raise RuntimeError('No supported images found in input folder')
    names = [p.stem.removesuffix('_image') + '_mask.png' for p in files]
    if len({name.casefold() for name in names}) != len(names):
        raise RuntimeError('Input image names would produce duplicate output masks')
    output = args.output or ROOT / 'predictions' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = model.cuda().to(memory_format=torch.channels_last).eval()
    from training_awake import keep_system_awake
    with keep_system_awake():
        for path, name in zip(files, names):
            with Image.open(path) as image:
                rgb = np.array(image.convert('RGB'))
            pred = core.predict(model, rgb, config, tta=True, tile=768)
            Image.fromarray(core.PALETTE[pred]).save(output / name)
            print(name, flush=True)
    core.save_json(output / 'inference_info.json', {'count': len(files), 'tile': 768,
        'tta': 'D4', 'selected_epoch': checkpoint['epoch'], 'mIoU': None,
        'note': 'Unlabeled input images; no ground-truth metric was measured.'})
    print('OUTPUT:', output, flush=True)


if __name__ == '__main__':
    main()

