from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image

work=Path(__file__).resolve().parent
stage=work/'share_261015_priority_style89_efficient'
minimal=work/'minimal_run'
minimal.mkdir(exist_ok=False)
(minimal/'model').mkdir()
shutil.copy2(stage/'efficient.py',minimal/'efficient.py')
shutil.copy2(stage/'model/best.pt',minimal/'model/best.pt')
output=work/'cli_output'
completed=subprocess.run([sys.executable,'-B',str(minimal/'efficient.py'),
    '--input',r'C:\Users\SSAFY\Desktop\KOMAP\data\valid\images\A170_step4_sample3_image.png',
    '--output',str(output),'--save-probs'],cwd=work,capture_output=True,text=True,encoding='utf-8',timeout=300)
(work/'cli_execution.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
assert completed.returncode==0, completed.stdout+completed.stderr
key='A170_step4_sample3'
probs=np.load(output/'probabilities'/(key+'_prob.npy'),allow_pickle=False)
labels=np.array(Image.open(output/'labels'/(key+'_label.png')))
colors=np.array(Image.open(output/(key+'_mask.png')).convert('RGB'))
palette=np.array([[153,127,76],[76,178,76],[25,76,153],[204,204,204]],np.uint8)
assert probs.shape==(4,*labels.shape) and probs.dtype==np.float32 and np.isfinite(probs).all()
assert np.array_equal(labels,probs.argmax(0)) and np.array_equal(colors,palette[labels])
info=json.loads((output/'inference_info.json').read_text(encoding='utf-8'))
assert info['count']==1 and info['mIoU'] is None and info['selected_epoch']==25
with (minimal/'model/best.pt').open('rb') as stream:
    model_sha=hashlib.file_digest(stream,'sha256').hexdigest()
assert model_sha=='e464af75e27dc7e8204287b2dc6f11fca8184c05aad09db5050dd0a320b0c626'
record={'passed':True,'standalone_files':['efficient.py','model/best.pt'],
        'other_local_modules_or_configs_present':False,'output_shape':list(probs.shape),
        'probabilities_labels_and_rgb_masks_consistent':True,'model_sha256':model_sha,
        'inference_info':info}
(work/'cli_check.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(completed.stdout,flush=True)
print('STANDALONE_CLI_VERIFIED',flush=True)
