"""Independent EMA arithmetic and recorded-data invariants before a long run."""
import copy,json
from pathlib import Path
import torch
import experiment as core

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w=torch.nn.Parameter(torch.tensor([1.]))
        self.register_buffer('running',torch.tensor([.2])); self.register_buffer('counter',torch.tensor(0,dtype=torch.int64))

online=Tiny(); ema=copy.deepcopy(online)
with torch.no_grad(): online.w.fill_(3); online.running.fill_(.6); online.counter.fill_(7)
core.update_ema(ema,online,.5)
assert torch.equal(ema.w,torch.tensor([2.])) and torch.allclose(ema.running,torch.tensor([.4])) and ema.counter.item()==7
with torch.no_grad(): online.w.fill_(5)
core.update_ema(ema,online,.5)
assert torch.equal(ema.w,torch.tensor([3.5])) and torch.equal(online.w,torch.tensor([5.]))
root=Path(__file__).resolve().parents[1]
split=json.loads((root/'configs/split_group_fold0.json').read_text(encoding='utf-8'))
assert len(split['train'])==72 and len(split['valid'])==18 and len(set(split['test']))==10
assert not set(split['train'])&set(split['valid'])
assert not {x.split('_')[0] for x in split['train']}&{x.split('_')[0] for x in split['valid']}
assert len({x.split('_')[0] for x in split['valid']})==14
core.save_json(root/'notes_recovered/cpu_verification.json',{'passed':True,'ema_closed_form_two_updates':True,
    'EMA_does_not_mutate_online_model':True,'integer_buffers_copied':True,'A_groups_disjoint':True,
    'valid_A_groups':14,'original_split_verified':False})
print('NOTES_PROTOCOL_CPU_VERIFIED',flush=True)
