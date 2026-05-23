---
base_model: google/gemma-4-E2B-it
library_name: peft
tags:
- lora
- transformers
- gemma-4
- crab-step
- amr-CrabStep
---

# Mode D / `mode_D` — Crab Step round 3 (83 neurons suppressed)

LoRA r=8 α=16 adapter trained on the same 5 samples, with
**inventory 23 + Mode B top-30 + Mode C top-30 = 83 neurons** all zeroed
during forward.

This is round **3** of the Crab Step loop and the first one trained
through the YAML-driven `crutch_pipeline.py`. The source YAML
(`core/configs/mode_D.yaml`) and the resolved YAML emitted at train
time (`config_resolved.yaml` next to this file) are both included for
full provenance.

| field | value |
|---|---|
| host model            | `google/gemma-4-E2B-it` |
| LoRA target modules   | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA rank / alpha     | 8 / 16 |
| epochs × samples      | 4 × 5 = 20 steps |
| precision             | BF16 |
| neurons suppressed    | 83 (23 + 30 + 30 across 22 layers) |

To load:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained(os.environ["GEMMA_PATH"],
                                            torch_dtype="bfloat16",
                                            attn_implementation="eager")
model = PeftModel.from_pretrained(base, "weights/mode_D")
```

This is the **recommended adapter** if you just want to try the Crab Step
output style — it's the deepest LoRA in the published series.
