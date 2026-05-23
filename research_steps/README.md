# research_steps/ — the journey, not the destination

This folder contains the exploratory scripts that **led to** the Crab Step
loop in `core/`. They are intentionally kept un-polished:

- They still contain the original hard-coded paths
  (`J:/amr/amr_wtf/...`, `J:/amr/models/gemma-4-E2B-it`).
- They are **not guaranteed to run** without adapting paths.
- They have no input-validation, no CLI, no documentation beyond the
  module docstring at the top of each file.

They are published because:

1. Some readers want the polished pipeline → use `core/`.
2. Some readers want to see what was tried before landing on the
   "iterative SFT + neuron suppression" idea — they should read the
   scripts in here. Reading order:

```
qa01_single_sample/01_qa01_grad_probe.py   ← starts here:
                                              per-token grad on a single QA
qa01_single_sample/02_qa01_intent_sft.py   ← 1-shot SFT with token weights
                                              (precursor of mode_A_intent_sft)
qa01_single_sample/03_qa01_intent_compare.py
qa01_single_sample/q01_logit_lens.py       ← layer-wise probes that
qa01_single_sample/q01_cloze.py              eventually identified the
qa01_single_sample/q01_mlp_diff.py           neurons.json inventory
qa01_single_sample/q01_format_diff.py
qa01_single_sample/q01_guided_diff.py
qa01_single_sample/q01_v3_guided.py
qa01_single_sample/q01_old_plus_new.py
qa01_single_sample/q01_steer.py            ← clamping/steering attempts

probes_and_variants/multi_qa_compare.py    ← how to compare modes A/B/C/D
probes_and_variants/crutch_attn_anchor_probe.py
                                            ← attempt to extend suppression
                                              to attention paths (not used
                                              in the final loop)
probes_and_variants/crutch_full_input_baseline.py
probes_and_variants/crutch_full_input_infer.py
probes_and_variants/crutch_qa05_infer.py
```

If you want to run any of these, expect to do path surgery first. The
scripts under `core/` are the supported entry points.
