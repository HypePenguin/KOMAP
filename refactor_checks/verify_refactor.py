from pathlib import Path
import inspect
import json
import sys
import time
import numpy as np
from PIL import Image
import torch

WORK = Path(__file__).resolve().parent
STAGE = WORK/'share_261015_priority_style89_efficient'
ROOT = Path(r'C:\Users\SSAFY\Desktop\KOMAP')
OLD = ROOT/'priority_style89_20261015/share_package/share_261015_priority_style89_A170_excluded'
sys.path.insert(0, str(STAGE))
from efficient import EfficientPredictor, PALETTE, keep_system_awake, tile_positions, load_rgb, file_sha256
sys.path.insert(0, str(OLD/'src'))
import experiment as original

torch.set_num_threads(4)
predictor = EfficientPredictor(device='cuda')
config = json.loads((OLD/'model/config.json').read_text(encoding='utf-8'))
config['inference_batch'] = 8
# Instrument the frozen original only at its final return to expose pre-argmax probabilities.
source = inspect.getsource(original.predict)
assert source.count('return native.argmax(-1).astype(np.uint8)') == 1
source = source.replace('def predict(', 'def reference_proba(').replace(
    'return native.argmax(-1).astype(np.uint8)', 'return np.ascontiguousarray(native.transpose(2,0,1), dtype=np.float32)')
namespace = dict(original.__dict__)
exec(compile(source, '<frozen-original-probability-return>', 'exec'), namespace)
reference_proba = namespace['reference_proba']

split = json.loads((OLD/'records/split.json').read_text(encoding='utf-8'))
records = []
with keep_system_awake():
    for subset, ids, pred_dir in [('valid', split['valid'], 'valid_masks_d4'), ('test', split['test'], 'test_masks_d4')]:
        for index, key in enumerate(ids):
            start = time.perf_counter()
            rgb = load_rgb(ROOT/'data'/subset/'images'/(key+'_image.png'))
            probs = predictor.predict_proba(rgb)
            assert probs.shape == (4, *rgb.shape[:2]) and probs.dtype == np.float32
            assert probs.flags.c_contiguous and np.isfinite(probs).all()
            assert probs.min() >= 0 and probs.max() <= 1
            sum_error = float(np.abs(probs.sum(0)-1).max())
            assert sum_error < 1e-5
            predicted = probs.argmax(0).astype(np.uint8)
            saved = np.array(Image.open(OLD/'predictions'/pred_dir/(key+'_mask.png')).convert('RGB'))
            different = int(np.any(PALETTE[predicted] != saved, axis=-1).sum())
            row = {'id':key,'pixels':int(predicted.size),'different_mask_pixels':different,'probability_sum_max_error':sum_error}
            if subset == 'valid' or index == 0 or different:
                reference = reference_proba(predictor.model, rgb, config, tta=True, tile=768)
                row['reference_proba_max_abs_error'] = float(np.abs(probs-reference).max())
                row['original_code_vs_saved_mask_pixels'] = int(np.any(PALETTE[reference.argmax(0)] != saved,axis=-1).sum())
                assert np.array_equal(probs, reference), row
            if subset == 'valid':
                label = original.decode_mask(ROOT/'data/valid/masks'/(key+'_mask.png'))
                cm = np.bincount(label.ravel()*4+predicted.ravel(), minlength=16).reshape(4,4)
                unions = cm.sum(0)+cm.sum(1)-cm.diagonal()
                row['single_image_mIoU'] = float((cm.diagonal()/unions).mean()*100)
                row['stored_model_reference_mIoU'] = 86.61428017677498
            row['seconds'] = time.perf_counter()-start
            records.append(row)
            (WORK/'comparison_progress.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(json.dumps(row), flush=True)
    # Odd, small image covers reflection padding and non-square native resizing.
    rgb = load_rgb(ROOT/'data/valid/images/A170_step4_sample3_image.png')[100:273,50:269].copy()
    actual = predictor.predict_proba(rgb)
    expected = reference_proba(predictor.model,rgb,config,tta=True,tile=768)
    assert np.isfinite(actual).all()
    assert np.array_equal(actual,expected)
    for invalid in [np.zeros((5,5),np.uint8), np.zeros((5,5,3),np.float32), np.zeros((0,5,3),np.uint8)]:
        try:
            predictor.predict_proba(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
for length in [1,767,768,769,1024,1536,1621]:
    positions=tile_positions(length)
    coverage=np.zeros(max(length,768),np.int32)
    for p in positions:
        coverage[p:p+768]+=1
    assert (coverage>0).all()

record = {'passed':True,'device':torch.cuda.get_device_name(),'torch':torch.__version__,
          'model_sha256':predictor.model_sha256,'efficient_py_sha256':file_sha256(STAGE/'efficient.py'),
          'saved_masks_checked':len(records),'all_saved_masks_identical':all(row['different_mask_pixels']==0 for row in records),
          'reference_probability_arrays_bit_equal':True,'padding_odd_shape_checked':[173,219],
          'boundary_metric_scope':'A170 one-image reference only; test10 has no ground truth',
          'records':records}
(WORK/'inference_equivalence.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('ALL_CHECKS_PASSED',flush=True)
