# amr_CrabStep

**Crab Step — a general iterative SFT + neuron-suppression loop for
probing how a learnable behaviour is encoded across an LLM's MLP
neurons.**

> "Crab Step" (螃蟹步) is the metaphor: like a crab moving sideways,
> every round we *close the road in front of us* (forcibly zero the
> top-recruited neurons from the previous SFT pass) and re-train from
> scratch. If the model still reaches the target, the capability was
> distributed; if it stalls, the capability was single-point. Either
> way, the sideways trajectory itself is a map of the redundant
> machinery.

---

- [English](#english)
- [中文](#中文)

---

## English

### What Crab Step is

Crab Step is a four-step loop that you wrap around **any** task with a
finite training set and a frozen LLM:

1. **Train** a low-rank adapter (LoRA, soft-prompt, ROME-style edit —
   anything trainable) on your samples.
2. **Measure absorption** — for each candidate neuron, sum the
   gradient flowing through it across the run. The biggest absorbers
   are the "recruits" that did most of the learning this round.
3. **Suppress** the top-K recruits with a forward-pre-hook (zero out
   the corresponding `down_proj` input dimension, attention head, or
   whatever your suppression granularity is).
4. **Reset and retrain** from the unmodified base model. Same data,
   same hyperparameters, but those neurons can no longer participate.
   Repeat.

What the trajectory tells you:

| Trajectory across rounds                       | Conclusion |
|------------------------------------------------|------------|
| Loss keeps converging round after round        | Capability is **distributed** — there is a deep bench of backup circuits |
| Loss plateaus much higher in round 2 / 3       | Capability is **single-point** — the suppressed circuit was *the* circuit |
| Loss converges, but to a qualitatively different output | Capability has **multiple solutions** — Crab Step is surfacing them |
| Top-1 absorbed gradient halves each round      | The redundant pool is being exhausted; stop in ~2 more rounds |

### What it is NOT

- It is **not** a fine-tuning recipe that produces the "best" adapter.
  Each round is a *probe*, not a destination — the adapters from
  rounds 1, 2, 3 are all valid artefacts, and ensembling them is one
  legitimate use, but the loop's purpose is **introspection**.
- It is **not** specific to LoRA. The same loop works with full SFT,
  soft prompts, attention-head ablations, etc. — anything where you
  can attribute gradient to a discrete unit and zero that unit out.
- It is **not** restricted to MLP neurons. The reference implementation
  here zeros `down_proj` inputs because that's the cleanest unit on a
  Gemma 4 MLP, but the suppression hook can target attention heads,
  experts in an MoE, or arbitrary dimensions of any intermediate
  representation.

### Things people might use Crab Step for

The reference task in this repo is "make a small model write in Opus
4.7's structured-critique style," because that's what we had data for.
But the loop applies anywhere the question "is this capability
redundantly encoded?" is interesting:

- **Distributed-vs-localised audit.** Pick any behaviour (refusal, code
  formatting, language ID, jailbreak resistance) and run Crab Step
  to count how many neurons it actually needs.
- **Surfacing dormant circuits.** The "recruits" surfaced in round 2
  and beyond are circuits the model normally doesn't use — they are
  candidates for further interpretability work or for direct
  steering.
- **Robust style / behaviour transfer.** If you want a downstream
  adapter that survives ablation of one or two known neurons, train
  it under suppression of those neurons (Mode B onward) — the
  resulting LoRA is provably independent of the suppressed set.
- **Hardening interventions.** Combine Crab Step with single-neuron
  rewrites: if a one-neuron edit changes a behaviour, run Crab Step
  with that neuron suppressed to find out whether the edit will
  survive future training.
- **Auditing which neurons are load-bearing for a fact / format / rule.**
  After N rounds, neurons that were never recruited and never
  suppressed are demonstrably idle for this task — useful for
  pruning, for sandboxing edits, or for safety analysis.

### What's in the box

The reference implementation runs on a frozen **Gemma 4 E2B-it** with
five paired (story, Opus-4.7-style critique) samples. The framework is
deliberately written so the **data file and the suppression criteria
are the only things you need to change** to repurpose it.

```
amr_CrabStep/
├── README.md                       this file
├── LICENSE                         MIT + Gemma TOU + Opus fair-use notice
├── FINGERPRINT.md                  SHA-256 of every published artefact
├── requirements.txt
│
├── core/                           the loop — clean, polished
│   ├── crutch_pipeline.py          YAML-driven entry: one command = one round
│   ├── mode_A_intent_sft.py        round 0 (no suppression, baseline)
│   ├── mode_B_crutch_off.py        round 1 (suppress base inventory)
│   ├── mode_C_crutch_C.py          round 2 (inventory + round-1 recruits)
│   ├── infer_one_mode.py           load an adapter ± suppression
│   ├── compare_modes.py            trajectory analysis across rounds
│   ├── ensemble_infer.py           weighted-LoRA fusion across rounds
│   ├── ensemble_depth_sweep.py     hyperparameter sweep around an ensemble
│   ├── configs/
│   │   └── mode_D.yaml             round 3 — 23+30+30 = 83 neurons suppressed
│   └── neurons.json                base inventory for round 1
│
├── data/                           the reference task's training set
│   └── claudeopusQA0{1..5}.json    (one of many possible tasks)
│
├── weights/                        artefacts from each round (Gemma + Opus task)
│   ├── mode_off/                   round 1 — 23-neuron suppression
│   ├── mode_C/                     round 2 — 53-neuron suppression
│   └── mode_D/                     round 3 — 83-neuron suppression
│        each contains: adapter_model.safetensors (46 MB) + adapter_config.json
│        + training_grads.pt (recruits / loss curves) + per-mode README
│
├── outputs/                        reference outputs (proof of life)
│   ├── training/                   per-round summary.txt and run.log
│   ├── inference/                  6 generated samples (mode × on/off)
│   └── ensemble/                   depth-sweep + ensemble demos
│
└── research_steps/                 the journey — un-polished, retained for context
    ├── qa01_single_sample/         single-QA exploration that preceded multi-QA
    └── probes_and_variants/        attention-anchor probe, other variants
```

### Run the included demo (no training required)

Drop a Gemma 4 E2B-it checkpoint anywhere, point `GEMMA_PATH` at it,
and load the deepest published adapter (round 3):

```powershell
# PowerShell
$env:GEMMA_PATH = "C:\models\gemma-4-E2B-it"
$env:CRABSTEP_ADAPTER = "weights\mode_D"
python core\infer_one_mode.py
# → outputs\inference\... contains a generated critique
```

```bash
# bash
export GEMMA_PATH=/models/gemma-4-E2B-it
export CRABSTEP_ADAPTER=weights/mode_D
python core/infer_one_mode.py
```

### Reproduce the rounds from scratch

```bash
export GEMMA_PATH=/models/gemma-4-E2B-it

# Round 1 — suppress base inventory (23 neurons)
python core/mode_B_crutch_off.py

# Round 2 — inventory + round-1 top-30 recruits = 53
python core/mode_C_crutch_C.py

# Round 3 — inventory + round-1 + round-2 top-30 each = 83
python core/crutch_pipeline.py core/configs/mode_D.yaml

# Round 4 (yours) — copy mode_D.yaml to mode_E.yaml,
# append weights/mode_D/training_grads.pt to add_recruits, rerun:
python core/crutch_pipeline.py core/configs/mode_E.yaml
```

From round 3 onward the loop is purely YAML-driven; rounds 1 and 2
exist as hand-coded reference implementations of the same logic.

### Apply Crab Step to your own task

Three edits and you're off:

1. **Swap in your training data** — replace `data/claudeopusQA0{1..5}.json`
   (or add new ones and bump `QA_IDS` in `mode_A_intent_sft.py` /
   `qa_ids` in `mode_D.yaml`). The pipeline assumes each file carries
   `input` (user prompt) and `output` (target). The per-token weighting
   fields (`conclusion_analysis` / `glue_sentences`) are optional —
   omit them or set all weights to 1.0 if you don't need them.

2. **Define what counts as the "base inventory"** — edit
   `core/neurons.json` to list the neurons you want suppressed in
   round 1. If you have no prior inventory, start with an empty list
   and round 1 becomes an unconstrained SFT; rounds 2+ will still find
   recruits.

3. **Pick your suppression granularity** — the included hook zeros
   columns of MLP `down_proj` input. To suppress an attention head or
   an MoE expert, edit `install_suppression_hooks()` in
   `crutch_pipeline.py` (~10 lines).

The loop and the recruit-discovery logic are otherwise task-agnostic.

### Reference-task result

For our specific demo (Gemma 4 E2B + 5 Opus-style critiques),
three rounds of Crab Step give the following trajectory:

| metric                  | round 1 (23) | round 2 (53) | round 3 (83) |
|-------------------------|-------------:|-------------:|-------------:|
| avg Δloss across 5 QAs  |        +1.91 |        +2.54 |        +2.99 |
| top-1 recruit cum_grad  |         1.19 |         0.93 |         0.86 |
| layers with recruits    |            9 |           21 |           22 |

This is the **"loss keeps converging"** branch — the structured-critique
behaviour is distributed across at least 83 neurons in 22 layers. The
included adapters and inference outputs prove every round still
generates a complete four-section critique, even with 83 neurons
forcibly zeroed.

We expect the trajectory to look different on *your* task, and that's
the point: **the trajectory is the result, not the adapter.**

### Limitations of the method

1. **Diminishing returns** — top-1 recruit gradient roughly halves
   every 2 rounds. The redundant pool is finite. Plan for 3–5 rounds,
   not 50.
2. **Sensitive to attribution choice** — "biggest absorber" can be
   defined as cumulative grad, max grad, integrated gradients, etc.;
   different choices give different recruit orderings. We use
   per-step `grad.norm(dim=0)` cumulated over the run.
3. **Suppression collateral** — zeroing a neuron does collateral
   damage to whatever else it does. Round 3 / 4 outputs may exhibit
   token-level degeneracy (e.g. trailing repetition) that has nothing
   to do with the target behaviour. Read the outputs critically.
4. **Granularity matters** — MLP-neuron suppression doesn't touch
   attention paths. If your behaviour is attention-mediated, swap the
   hook target or you'll get a false "single-point" conclusion.

### Reproducibility

| concern             | how addressed                                          |
|---------------------|--------------------------------------------------------|
| host-model drift    | SHA-256 in `FINGERPRINT.md` for Gemma snapshot         |
| attention backend   | `eager`-attention is required (Gemma 4 PLE)            |
| precision           | BF16 throughout; FP16 may drift on long sequences      |
| determinism         | `do_sample=False` greedy; minor variance still possible|

### Acknowledgments

- Built on Google DeepMind's **Gemma 4 E2B-it**.
- The five demo critiques (`data/claudeopusQA0{1..5}.json`) were
  generated with **Claude Opus 4.6** during the predecessor project
  ([chenmoacr/amr_wtf](https://github.com/chenmoacr/amr_wtf), "GHOST");
  used here as one concrete task to demonstrate the loop. Fair-use
  educational reference.
- Sibling project:
  [chenmoacr/AMR_ReplaceNeuron](https://github.com/chenmoacr/AMR_ReplaceNeuron)
  — single-neuron rewriting on the same host. Crab Step is the
  *dynamic* counterpart: where AMR_ReplaceNeuron pins behaviour to a
  hand-picked neuron, Crab Step asks "what happens if you take that
  neuron away?"

### Contact

Author: IndexGuc · indexguc@gmail.com · https://github.com/chenmoacr

---

## 中文

### 螃蟹步是什么

Crab Step（螃蟹步）是一个**通用四步循环**，可以套在任何「冻结底模 +
有限训练集」的任务上：

1. **训练**一个低秩适配器（LoRA、soft prompt、ROME 编辑——任何可训练
   的东西）在你的样本上。
2. **测量吸收**——对每个候选神经元，把这一轮全程的梯度累加起来。吸
   收最多的就是这一轮的"招募"神经元，做了最多学习工作。
3. **抑制**——用 forward-pre-hook 把 top-K 招募神经元强制清零（清零
   对应的 `down_proj` 输入维度，或注意力头，或你选定的任何抑制粒
   度）。
4. **重置后重训**——从未经修改的底模开始，用同样的数据、同样的超
   参重训一遍，但被抑制的神经元这次不能参与。回到第 1 步。

跨轮次的轨迹告诉你什么：

| 跨轮次轨迹                              | 结论 |
|----------------------------------------|------|
| Loss 每轮都还在收敛                    | 能力是**分布式编码**——有一条很长的备份板凳 |
| 第 2/3 轮 loss 卡死在很高的位置        | 能力是**单点电路**——被抑制的那条就是唯一通路 |
| Loss 收敛了但输出明显变成另一个形态    | 能力存在**多解**——Crab Step 把它们一个个挖出来了 |
| 每轮 top-1 招募梯度折半                | 冗余池正在耗尽，再 2 轮可以停了 |

### 不是什么

- **不是**一个"产出最好 adapter"的微调配方。每一轮都是一次**探针**而
  非终点——第 1、2、3 轮的 adapter 全都是合法的工件，把它们集成是一
  种合理用法，但循环本身的目的是**内省**。
- **不绑死 LoRA**。同一循环对全模型 SFT、soft prompt、注意力头消融都
  适用——只要你能把梯度归因到离散单元，并把那个单元清零，就能跑。
- **不限于 MLP 神经元**。本仓库参考实现清零 `down_proj` 输入，是因为
  那是 Gemma 4 MLP 上最干净的单元。抑制钩子可以挂在注意力头、MoE
  expert、或任意中间表征的任意维度上。

### 大家可以用螃蟹步做什么

本仓库的参考任务是「让小模型写 Opus 4.7 结构化批评风格」，那只是因
为我们手上有这份数据。但只要你想问「这个能力是不是冗余编码的？」，
循环就能用：

- **分布式 vs. 局部审计**。挑任何一种行为（refusal、代码格式、语言
  检测、jailbreak 抵抗），跑一遍 Crab Step，数一数它到底需要多少
  神经元。
- **挖出沉睡电路**。第 2 轮起被招募的神经元就是模型平时不用的电路
  ——它们是后续可解释性工作 / 直接 steering 的候选。
- **稳健的风格 / 行为迁移**。如果你想要一个能在「某些已知神经元被
  消融」时仍然成立的 adapter，那就在抑制它们的条件下训练（Mode B
  之后）——这样产出的 LoRA 在数学意义上独立于被抑制的集合。
- **加固干预**。把 Crab Step 跟单神经元改写组合：如果一次单点编辑
  改变了某个行为，把那个神经元抑制后跑 Crab Step，看这次编辑能不
  能扛得住后续训练。
- **审计「这个事实 / 这个格式 / 这条规则到底压在哪些神经元上」**。
  跑 N 轮后，从未被招募、从未被抑制的神经元就**实证地**对这个任务
  是闲置的——可以拿去剪枝、沙箱编辑、做安全分析。

### 仓库地图

参考实现跑在冻结的 **Gemma 4 E2B-it** 上，用 5 对（故事，Opus-4.7
风格批评）样本。框架的写法**故意**让数据文件和抑制条件成为唯一需要
改的东西，方便迁移到别的任务。

```
amr_CrabStep/
├── README.md                       本文件
├── LICENSE                         MIT + Gemma TOU + Opus 引用合理使用
├── FINGERPRINT.md                  所有发布文件的 SHA-256
├── requirements.txt
│
├── core/                           循环本体（精修过）
│   ├── crutch_pipeline.py          YAML 驱动主入口，一条命令跑完一轮
│   ├── mode_A_intent_sft.py        第 0 轮（无抑制，基线）
│   ├── mode_B_crutch_off.py        第 1 轮（抑制基础 inventory）
│   ├── mode_C_crutch_C.py          第 2 轮（inventory + 第 1 轮招募）
│   ├── infer_one_mode.py           加载某 adapter ± 抑制做推理
│   ├── compare_modes.py            跨轮次轨迹分析
│   ├── ensemble_infer.py           多轮 LoRA 加权融合
│   ├── ensemble_depth_sweep.py     集成的超参 sweep
│   ├── configs/
│   │   └── mode_D.yaml             第 3 轮配置（83 个神经元）
│   └── neurons.json                第 1 轮用的基础 inventory
│
├── data/                           参考任务的训练集
│   └── claudeopusQA0{1..5}.json    （只是众多可能任务之一）
│
├── weights/                        各轮训练产物（Gemma + Opus 任务）
│   ├── mode_off/                   第 1 轮 —— 23 个神经元抑制
│   ├── mode_C/                     第 2 轮 —— 53 个抑制
│   └── mode_D/                     第 3 轮 —— 83 个抑制
│        每个目录含：adapter_model.safetensors (46 MB) + adapter_config.json
│        + training_grads.pt（招募集 / loss 曲线）+ per-mode README
│
├── outputs/                        参考输出（证明跑得起来）
│   ├── training/                   每轮的 summary.txt + run.log
│   ├── inference/                  6 篇生成样本（mode × on/off）
│   └── ensemble/                   depth-sweep + ensemble demos
│
└── research_steps/                 研发过程（未精修，留作上下文）
    ├── qa01_single_sample/         多 QA 之前的单 QA 探索
    └── probes_and_variants/        注意力锚定 probe、其他 variant
```

### 一键跑 demo（不用训练）

把 Gemma 4 E2B-it 检查点放在硬盘任意位置，用 `GEMMA_PATH` 指过去，
加载最深的第 3 轮 adapter：

```powershell
# PowerShell
$env:GEMMA_PATH = "C:\models\gemma-4-E2B-it"
$env:CRABSTEP_ADAPTER = "weights\mode_D"
python core\infer_one_mode.py
```

```bash
# bash
export GEMMA_PATH=/models/gemma-4-E2B-it
export CRABSTEP_ADAPTER=weights/mode_D
python core/infer_one_mode.py
```

### 从零复现各轮

```bash
export GEMMA_PATH=/models/gemma-4-E2B-it

# 第 1 轮：抑制基础 inventory（23 个）
python core/mode_B_crutch_off.py

# 第 2 轮：inventory + 第 1 轮 top-30 = 53 个
python core/mode_C_crutch_C.py

# 第 3 轮：inventory + 前两轮各 top-30 = 83 个
python core/crutch_pipeline.py core/configs/mode_D.yaml

# 第 4 轮（你的）—— 复制 mode_D.yaml 为 mode_E.yaml，
# 在 add_recruits 里追加 weights/mode_D/training_grads.pt，再跑：
python core/crutch_pipeline.py core/configs/mode_E.yaml
```

第 3 轮起循环完全 YAML 化；第 1、2 轮的硬编码脚本是同一逻辑的参考
实现。

### 把螃蟹步套到你自己的任务上

改三处就够：

1. **换掉训练数据**——把 `data/claudeopusQA0{1..5}.json` 替换成你的
   样本（或新增几个，把 `mode_A_intent_sft.py` 里的 `QA_IDS` /
   `mode_D.yaml` 里的 `qa_ids` 同步改了）。流水线假定每个文件含
   `input`（用户 prompt）和 `output`（目标）。按 token 加权的
   `conclusion_analysis` / `glue_sentences` 字段是可选的——不需要
   就忽略，或把所有权重设成 1.0。

2. **定义什么算「基础 inventory」**——改 `core/neurons.json`，列出
   你想在第 1 轮就抑制的神经元。如果没有先验 inventory，留空列表
   ，第 1 轮就退化成无约束 SFT；从第 2 轮起仍能挖到招募集。

3. **挑你的抑制粒度**——内置钩子清零的是 MLP `down_proj` 输入列。
   要抑制注意力头或 MoE expert，改 `crutch_pipeline.py` 里的
   `install_suppression_hooks()`（约 10 行）。

循环和招募发现逻辑本身和任务无关。

### 参考任务的结果

具体到我们这个 demo（Gemma 4 E2B + 5 段 Opus 风格批评），三轮 Crab
Step 的轨迹是：

| 指标                     | 第 1 轮 (23) | 第 2 轮 (53) | 第 3 轮 (83) |
|--------------------------|-------------:|-------------:|-------------:|
| 5 个 QA 平均 Δloss       |        +1.91 |        +2.54 |        +2.99 |
| top-1 招募 cum_grad      |         1.19 |         0.93 |         0.86 |
| 出现招募的层数           |            9 |           21 |           22 |

这是「**loss 一直在收敛**」分支——结构化批评行为分布在至少 83 个神
经元、22 层里。仓库附带的 adapter 和推理输出可以验证：即使强制清零
83 个神经元，每一轮仍能生成完整的四段批评。

**你的任务上的轨迹会不一样，那才是重点：轨迹本身就是结果，不是
adapter。**

### 方法本身的局限

1. **边际收益快速衰减**——top-1 招募梯度每 2 轮折半。冗余池是有限
   的，规划 3-5 轮就够，不要指望 50 轮。
2. **对归因方式敏感**——「最大吸收者」可以定义成累计梯度、最大梯
   度、积分梯度等等；不同定义给出不同招募排序。我们用按步
   `grad.norm(dim=0)` 累加。
3. **抑制的副作用**——清零一个神经元会牵连它在做的其他事。第 3、4
   轮的输出可能出现跟目标行为无关的 token 级退化（比如尾部复读）。
   读输出时要带着批判。
4. **粒度很关键**——MLP 神经元抑制碰不到注意力路径。如果你的行为
   是注意力中介的，要换钩子目标，否则会得出假的「单点电路」结论。

### 复现性说明

| 关注点         | 处理方式                                                  |
|----------------|-----------------------------------------------------------|
| 底模漂移       | `FINGERPRINT.md` 里有 Gemma 快照的 SHA-256                |
| 注意力实现     | 必须 `eager`-attention（Gemma 4 PLE 在 sdpa 上有问题）    |
| 精度           | 全程 BF16；FP16 在长序列上可能有偏移                      |
| 确定性         | `do_sample=False` 贪婪解码；仍有少量数值非确定性          |

### 致谢

- 基于 Google DeepMind 的 **Gemma 4 E2B-it**。
- 五段 demo 批评（`data/claudeopusQA0{1..5}.json`）是在前置项目
  [chenmoacr/amr_wtf](https://github.com/chenmoacr/amr_wtf)（"GHOST"）
  里用 **Claude Opus 4.6** 生成的；这里作为一个具体任务来演示循环。
  教育性引用合理使用。
- 姊妹仓库：
  [chenmoacr/AMR_ReplaceNeuron](https://github.com/chenmoacr/AMR_ReplaceNeuron)
  —— 同底模上的单神经元改写。Crab Step 是它的**动态**对应物：
  AMR_ReplaceNeuron 把行为钉在一个手工挑的神经元上，Crab Step 问
  「如果把那个神经元拿掉会发生什么？」。

### 联系方式

作者：IndexGuc · indexguc@gmail.com · https://github.com/chenmoacr
