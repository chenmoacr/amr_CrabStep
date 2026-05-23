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

# Mode C / `mode_C` — Crab Step round 2 (53 neurons suppressed)

LoRA r=8 α=16 adapter trained on the same 5 samples as Mode B, but with
**inventory 23 + top-30 recruits from Mode B = 53 neurons** all zeroed
during forward.

This is round **2** of the Crab Step loop. The 30 recruits were chosen
as the highest `cum_neuron_down` entries from Mode B's
`training_grads.pt` that were NOT already in the inventory.

| field | value |
|---|---|
| host model            | `google/gemma-4-E2B-it` |
| LoRA target modules   | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA rank / alpha     | 8 / 16 |
| epochs × samples      | 4 × 5 = 20 steps |
| precision             | BF16 |
| neurons suppressed    | 53 (23 inventory + 30 round-1 recruits, see `training_grads.pt`) |

To load:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import os
base = AutoModelForCausalLM.from_pretrained(os.environ["GEMMA_PATH"],
                                            torch_dtype="bfloat16",
                                            attn_implementation="eager")
model = PeftModel.from_pretrained(base, "weights/mode_C")
```
