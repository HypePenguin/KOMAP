from pathlib import Path
import json
import sys
import time
import numpy as np
import torch

work=Path(__file__).resolve().parent
sys.path.insert(0,str(work/'share_261015_priority_style89_efficient'))
from efficient import EfficientPredictor

torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
predictor=EfficientPredictor(device='cpu',batch_size=1)
assert all(parameter.dtype == torch.float32 for parameter in predictor.model.parameters())
rgb=np.random.default_rng(42).integers(0,256,(17,29,3),dtype=np.uint8)
start=time.perf_counter()
with torch.autocast('cpu',dtype=torch.bfloat16):
    probs=predictor.predict_proba(rgb)
assert probs.shape==(4,17,29) and probs.dtype==np.float32 and np.isfinite(probs).all()
assert np.max(np.abs(probs.sum(0)-1))<1e-5
torch.set_default_dtype(torch.float32)
record={'passed':True,'device':'cpu','external_default_dtype':'float64',
        'external_autocast':'bfloat16','actual_model_dtype':'float32',
        'output_shape':list(probs.shape),'output_dtype':str(probs.dtype),
        'seconds':time.perf_counter()-start}
(work/'cpu_api_check.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
