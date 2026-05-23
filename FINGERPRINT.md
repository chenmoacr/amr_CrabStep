# amr_CrabStep — fingerprint

This file fixes the SHA-256 of every published artefact (trained LoRA
adapters, training tensors, datasets) along with the exact Gemma 4 E2B-it
host-model snapshot they were produced against, so any reader can verify
they are reproducing the same setup.

None of the artefacts in this repository contain redistributed Gemma model
weights. The trained LoRA adapter files and `training_grads.pt` files are
all derivative modules that **must be combined with a separately downloaded
Gemma 4 E2B-it checkpoint** in order to run. Gemma 4 E2B-it is governed by
Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms); you must
accept that license to use the host model.

---

## 1. Trained LoRA adapters (rounds 1-3 of the Crab Step loop)

### 1.1 Round 1 — `weights/mode_off/` (23 inventory neurons suppressed)

| File | Size | SHA-256 |
|------|------|---------|
| `adapter_model.safetensors` | 48,376,416 B | `311180997FB02A4A08D55593FD85D63DDA55938AAA8747A8624D2637D42BB033` |
| `adapter_config.json`       | 1,152 B      | `19C91F3E2A85BC323BB239DE149397975E3BBFD5DAA72D04152438AF2B40FDA9` |
| `training_grads.pt`         | 1,367,318 B  | `34A1686070734EC30A0C04DAAF39347CCD2654438DA3C297B0CD0FE9FE2411ED` |

`training_grads.pt` carries: `suppress_dict`, `cum_neuron_down` (per-layer
gradient absorption tensor used to pick recruits for the next round),
`history` (per-step loss), and `config`. The adapter is a standard PEFT
LoRA r=8 α=16 over `q,k,v,o,gate,up,down`-proj.

### 1.2 Round 2 — `weights/mode_C/` (23 + 30 = 53 neurons)

| File | Size | SHA-256 |
|------|------|---------|
| `adapter_model.safetensors` | 48,376,416 B | `97E02DF5918B016B49B83297F8BFB1EA442B82268EB5E0DA52CC891212AD18FE` |
| `adapter_config.json`       | 1,152 B      | `509613B3338D6F872B34B9A636F252791B953D2A16652DE8D53C8932E1D20DCF` |
| `training_grads.pt`         | 1,368,022 B  | `3CC4FA58065F097079A320A10594C3E771C16B214E79AEDF0D292E0CC5A44C49` |

### 1.3 Round 3 — `weights/mode_D/` (23 + 30 + 30 = 83 neurons)

| File | Size | SHA-256 |
|------|------|---------|
| `adapter_model.safetensors` | 48,376,416 B | `BDEE495BD6D163C98D8A5E09817A8D8C59B57C6E27CDA01695E3475895C8C819` |
| `adapter_config.json`       | 1,152 B      | `7242B1C103849D7D6A5B22D935B8AC5AFC9348911190D0A7BD04C1DA28542DCE` |
| `training_grads.pt`         | 1,373,334 B  | `8B42836FCFABBA8B45F252708AF177A25EBA2F61A889806851923969F2FEE25E` |
| `config_resolved.yaml`      | 1,116 B      | `EFC9C4C30D3A3B4BDCEDCD80699947FA7B40C44A64241B6E48ED1D4C93E556F8` |

`config_resolved.yaml` is the exact YAML used at training time. It was
emitted by `crutch_pipeline.py` after merging defaults; the source YAML
is `core/configs/mode_D.yaml`.

---

## 2. Datasets

| File | Size | SHA-256 |
|------|------|---------|
| `data/claudeopusQA01.json` | 28,411 B | `31029F7F05EB2BF241EDE8080ADA7F99DE490627A3D5BC4780727BE0368C44E5` |
| `data/claudeopusQA02.json` | 10,474 B | `BC8B2B35DBDA034205DEEEC0F2B2C3CCF2906FE9A3BADAA4F584165CD5E7E390` |
| `data/claudeopusQA03.json` | 12,072 B | `9A20F061BCA050A7DDD3941C4786AE717283DA9A87AE5BB966E8EE99CDD9E79E` |
| `data/claudeopusQA04.json` | 10,028 B | `6999BD99CC17727A06B5EDAE9729197A08C47708FA99DCA3C2B65884AB186EC9` |
| `data/claudeopusQA05.json` | 27,729 B | `E95FC2AD8DD48DA4160A320BAD46F76323D376DE26F8D686BF8BCBD87F0FAAE9` |

Each JSON contains: `input` (story prompt), `output` (Opus-style critique),
`conclusion_analysis` (depth-tiered intent sentences), `glue_sentences`
(boilerplate to mask out at low weight). These five samples are the entire
training set; the loop's strength is exactly that 5 samples + redundancy
suffices.

---

## 3. Base neuron inventory

| File | Size | SHA-256 |
|------|------|---------|
| `core/neurons.json` | 18,445 B | `0DA64AB0C3F523B4DC67C6077A60E243EE0961CE823C1587F7EE0755C99DFEFA` |

The "round-1 base set" of ~74 known neurons, with tier (`verified`,
`general`, `lit`, `guided_diff`, etc.) and region (`thought`, `answer`,
`always`) annotations. Round 1 suppresses 23 of these (tiers ∈
{verified, general, lit, guided_diff} ∩ regions ∈ {answer, always}).

This file is heritage from the predecessor project (chenmoacr/amr_wtf,
"GHOST"); it is included verbatim because the round-1 result depends on
the exact selection.

---

## 4. Host model (must be downloaded separately)

All three LoRA adapters were trained against this exact Gemma 4 E2B-it
snapshot. The weights themselves are NOT in this repository.

| File | Size (bytes) | SHA-256 |
|------|--------------|---------|
| `model.safetensors`      | 10,246,621,918 | `2DB5482B20D746879BB3EF79B5203E9075A2E2B98F54EC7C2F281C1477DDC550` |
| `config.json`            | 4,954          | `1B28F3D2C3100F6C594754B81107428BD7B822A7F48272CA681DAE9D2EC38330` |
| `tokenizer.json`         | 32,169,626     | `CC8D3A0CE36466CCC1278BF987DF5F71DB1719B9CA6B4118264F45CB627BFE0F` |
| `tokenizer_config.json`  | 2,180          | `61876DB12AEDF4B5A7FC2605AE2A3BEA200E748C9FC52221AFBCA90323467930` |
| `chat_template.jinja`    | 16,317         | `781D10940FBC44BE40064B5D43A056FC486C84CEAA55538226368B57314132BF` |

These are from the Hugging Face `google/gemma-4-E2B-it` repository as of
the local copy timestamp **2026-04-20**. Newer or older snapshots may
produce slightly different per-layer activations and per-neuron indices;
when in doubt, retrain with `python core/mode_B_crutch_off.py` and check
that `training_grads.pt`'s top recruits match those listed in the published
`outputs/training/summary_mode_off.txt`.

How to verify your local Gemma copy:

```powershell
# Windows / PowerShell
Get-FileHash $env:GEMMA_PATH\model.safetensors -Algorithm SHA256
```

```bash
# Linux / macOS
sha256sum "$GEMMA_PATH/model.safetensors"
```

---

## 5. Reproducibility caveats

Reproducing the published numbers exactly requires:

1. Same Gemma 4 E2B-it snapshot (verified by §4 hashes).
2. Same Python / PyTorch / transformers / peft versions (see `requirements.txt`).
3. `eager` attention implementation (Gemma 4 PLE breaks on some shapes
   with `sdpa`; all scripts load with `attn_implementation="eager"`).
4. BF16 precision; FP16 may give different rounding on long sequences.
5. CUDA + a GPU with ≥ 12 GB VRAM (BF16 training peaks ~10 GB on Gemma 4 E2B).

Small numerical drift between runs (loss in the 4th decimal, recruit
ordering swaps within ±2) is expected. Large behavioural divergence
(QA loss off by > 0.3, completely different top-10 recruits) means one
of the above is misaligned.
