# OneIE experiment setup

Upstream: https://github.com/GerlinGreen/OneIE
Pinned commit: `3560bfc68cc8cf66ea8e226a614a1a75a9a5d979` (README: v0.4.8).
Clone at `external/OneIE`. Local compatibility changes live in
`src/oneie/runtime.py`; the upstream checkout is not modified.

Environment: Python 3.11, PyTorch 2.5.1 CUDA 11.8, Transformers 4.30.2.
This is a modern compatibility environment, not upstream's Python 3.7 environment.
Install versions in `requirements.txt` into `.venv-oneie`.

## Experiment definition

- Backbone: `bert-base-multilingual-cased`, shared tokenizer for all four datasets.
- Local OneIE variant: global features disabled. These runs are not a reproduction
  of the full global-feature model or published scores.
- 10 epochs, seed 42, microbatch 1, accumulation 4, maximum 512 subwords.
- Decoder beam 5; entities and event types retain their source labels.
- Missing entity/mention types use `ENTITY`/`UNK`; no gold entity inputs at prediction.
- Source event-extraction datasets have no entity-relation supervision; relation lists
  are empty. VHE event-to-event relations are a different task and are not exported.
- Vocabulary names follow upstream's union of splits; valid label combinations
  are derived from training only. No test examples enter the optimization loop.
- Dev role F1 selects the checkpoint; test evaluation runs only after training.

## Data adaptations and audit

BKEE/GENEVA/RAMS retain original split membership. RAMS windows stay intact,
including cross-sentence arguments. Windows exceeding 510 subwords are rejected,
not truncated. Scores apply to the exported subset, not automatically the original benchmark.

VHE upstream commit `d5d7b03d016f1c25d33748424251cf346e5455bc` publishes only
`event.json`, not train/dev/test split files. The local split hashes `42:` plus
the raw text with SHA-256 into 80/10/10 buckets. This is NOT an official split.
Six malformed character spans are rejected without annotation repair.

OneIE's flat BIO representation cannot preserve overlapping gold entities or
triggers. Export chooses argument entities before other entities, longer spans
before shorter spans, then start and ID. Trigger conflicts keep source order.
Links to removed entities are removed; competing roles on one event/entity keep
the first source role. Every removed annotation is saved in `modified.json`.
`rejected.json` records whole-record failures; `report.json` records split counts.
Interpret evaluation as a baseline on this explicitly flattened task.

## Commands (repository root)

```powershell
.venv-oneie/Scripts/python.exe -m src.oneie.export
.venv-oneie/Scripts/python.exe -m unittest discover -s tests
.venv-oneie/Scripts/python.exe -m src.oneie.train --dataset BKEE --smoke
.venv-oneie/Scripts/python.exe -m src.oneie.train --dataset BKEE
# Resume at the last completed epoch:
.venv-oneie/Scripts/python.exe -m src.oneie.train --dataset BKEE --resume
```

Repeat smoke/training for GENEVA, RAMS and VHE. A smoke test loads all splits,
numberizes them with the actual upstream loader, and runs forward, backward,
optimizer step and decoding. Artifacts are under `runs/oneie/<dataset>`.
MAVEN is excluded throughout.

To smoke-test all datasets and then train them sequentially for 10 epochs:

```powershell
./configs/oneie/run_all.ps1
# CPU smoke tests only:
./configs/oneie/run_all.ps1 -Cpu -SmokeOnly
```

## Validation completed

All four full exported splits pass upstream IEDataset loading/numberization.
Each dataset passes one real forward/backward/optimizer/decode batch on CPU.
CUDA 11.8 recognizes the GTX 1070 and supports its sm_61 architecture.
No full training run has been started: the GPU reports 93 C and 0% fan at idle,
so cooling must be checked before sustained GPU training.

| Dataset | Train | Dev | Test |
|---|---:|---:|---:|
| BKEE | 10959 | 4301 | 3736 |
| GENEVA | 1968 | 783 | 933 |
| RAMS | 7311 | 924 | 868 |
| VHE | 3266 | 451 | 388 |

RAMS excludes 21 overlength records; VHE excludes six malformed spans.
GENEVA has 3292 records modified by flattening/role conflict resolution;
3341 argument links and three overlapping events are removed. Therefore its
scores must not be reported as scores on unmodified GENEVA.
