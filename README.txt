KOMAP - 2026-10-07 experiments reconstructed from recovered Sheet cell notes

ACTIVE PIPELINE
The active pipeline is src/notes_recovered.py. All five corrected experiments
are saved separately under notes_recovered/runs/. The initial runs/ results and
the canceled recorded_protocol/ plan remain available as historical records.

WHAT WAS RECOVERED
records/historical_record_notes.json retains native Google Sheet cell notes.
They confirm U-Net, tu-tf_efficientnetv2_s.in21k_ft_in1k ImageNet encoder,
22,093,908 parameters, AdamW encoder3e-5 / decoder3e-4, weight_decay0.05,
EMA0.999, seed42, batch8, 600 patches/epoch, workers2, threads4,
native384 -> input768, and validation every5 epochs. October7 final runs use
max100 epochs, patience20 epochs, and min_delta0.02 percentage points.
Centerline target: final transformed E GT -> skeletonize -> cross dilation
twice -> intersect E GT. E recall uses smooth1e-5, present-E patch averaging.
Checkpoint selection uses tile512 without TTA; final inference uses inputx2,
tile768, overlap0.5, inverse D4 probability averaging and EMA weights.
The low-LR context variant uses encoder1e-6 / decoder1e-5 and dense E recall.
Other variants start independently from the same corrected baseline best.pt.

REPRODUCTION LIMITS
Original hsson-main source, folds.csv, initial checkpoint and package versions
were not recovered. Configs/record_notes.json separates confirmed settings
from remaining assumptions: cosine per-update scheduler/min_lr1e-7,
augmentation details, inference blending/batching, and tile512 selection
inherited by fine-tunes from the earlier common protocol.
The parameter count matches the source notes; this does not establish an
identical original model initialization, software environment or file split.

DATA AND FROZEN GROUP SPLIT
data/train: supplied70 labeled pairs; data/valid: supplied20 labeled pairs;
data/test: supplied10 unlabeled images. All originals remain unchanged.
records/data_manifest.json stores hashes, sizes, colors and class counts.
configs/split_group_fold0.json freezes a reconstruction candidate before any
model evaluation: train72/valid18, validation14 A groups, no training/validation
A-group overlap. Its constraints agree with the notes but the original file
IDs are unverified. The candidate was NOT chosen by model performance.
Donor samples come only from this training split. Some train/test A-group
overlap is present in the source notes and supplied data; exact file IDs do not
overlap. Test labels are absent, so no test mIoU is reported.

MEASUREMENT
mIoU(%) = class IoU mean per image, then image mean; skip classes with empty
union. Global/pixel-pooled mIoU is a separate metric and never substitutes for
the requested score. Exported RGB masks are independently rescored; source
data hashes and split invariants are checked before results are reported.
Historical scores are reference measurements from an unverified original
split. Cross-split differences do not establish improvement or degradation.
E softmask aims for >=79%; the actual result is recorded even if below target.

PRESERVED INITIAL WORK
Initial v1 omitted recovered EMA/WD/group-split/centerline settings.
Its completed D4 results remain unchanged:
  centerline_baseline: 77.26280490514819%
  e_softmask_geometry: 77.49962168453507%
These are incomplete reconstructions, not verified original reproduction.
Initial low_lr_context_paste was stopped with last.pt epoch20 preserved;
it has no completed final D4 score. Style and illumination were canceled.
The earlier recorded_protocol two-run supplement was canceled before training
because it also lacked the recovered settings. Do not resume those old queues.

RUN / RESUME
From PowerShell in this folder:
  .\run_experiments.ps1
Equivalent corrected training command:
  .\.venv\Scripts\python.exe -X utf8 -u src\notes_recovered.py --resume
Do not launch another copy while the current GPU training process is active.
Independent corrected-result audit:
  .\.venv\Scripts\python.exe -X utf8 src\audit_results.py --runs-dir notes_recovered\runs --split-file configs\split_group_fold0.json
Status and report (CPU only):
  .\.venv\Scripts\python.exe -X utf8 src\monitor_status.py --include-legacy
  .\.venv\Scripts\python.exe -X utf8 src\build_report.py
Status describes file evidence; stale progress is never treated as a live job.

FILES
notes_recovered/common/initial_weights.pt: reconstructed immutable initialization
notes_recovered/runs/<variant>/best.pt: selected EMA inference model
notes_recovered/runs/<variant>/last.pt: raw model, EMA and training resume states
notes_recovered/runs/<variant>/history.json, progress.json, result.json: records
notes_recovered/runs/<variant>/valid_masks_d4, test_masks_d4: palette masks
notes_recovered/runs/independent_audit.json, measured_results.csv: measured audit
records/report_sheet_blocks.json: exact rows for the native experiment Sheet
실험결과.txt: regenerated local report
재실험_결과.xlsx: export of the native Sheet after result/status updates
The report builder emits TXT/JSON and does not alter the workbook. Refresh the
XLSX by exporting the verified updated native Google Sheet to preserve formats.
Original Sheet rows1:101 remain intact; new rows explicitly identify the stage.

ENVIRONMENT
requirements.lock.txt and records/environment.json describe this PC's runtime.
The .venv inherits local CUDA PyTorch; copying that venv to another PC is not
a portable installation. Data, source code, configuration, logs, checkpoints
and inference outputs are all retained in this Desktop/KOMAP project.

POWER AND OVERNIGHT INTERRUPTION
Windows sleep at 2026-10-13 18:01 KST and hibernation at20:02 paused the
existing training process until 2026-10-14 08:21. It continued without restart.
Epoch53 and total train_minutes include this wall-clock pause; these are not
active GPU time. Evidence is retained in records/power_*_20261014.json.
src/training_awake.py holds a Windows system-awake request while training runs.
It leaves display-off and screen-lock behavior enabled and releases the request
when the watched process exits. Manual Sleep/Hibernate still pauses training:
use screen lock when leaving the computer during a run.
The current request is recorded in notes_recovered/power_guard.json.

TARGET IMPROVEMENT
Complete and audit all five corrected runs first. If corrected E softmask is
below final D4 validation mIoU79, src/target_search.py provides two predeclared
LR-only experiments with the same frozen split, metric, baseline and seed.
First: e_softmask_lr_x0p1 (encoder3e-6, decoder3e-5).
Second, only if needed: e_softmask_lr_x1over30 (encoder1e-6, decoder1e-5).
Each candidate uses task-scoped automatic-sleep prevention, saves full evidence,
and requires independent scoring of all18 validation and10 test masks.
These are new optimization experiments, distinct from historical reproduction.
The original folds.csv remains unavailable; reaching79 is not proof of identical
historical conditions. Read target_search/plan.json and use --prepare-only before
execution; do not modify the fixed split or run concurrent GPU training jobs.
