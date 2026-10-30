<p align="center">
  <img src="assets/readme-banner.svg" width="100%" alt="Slogan Remix — Familiar words. A new brand voice." />
</p>

### Recontextualizing Famous Quotes for Brand Slogan Generation

**Ziao Yang**, Zizhang Chen, Lei Zhang, and Hongfu Liu

**EMNLP 2026 · Main Conference**

**[PDF](https://arxiv.org/pdf/2602.06049)** · **[Paper](https://arxiv.org/abs/2602.06049)** · **[Explore code](#explore-the-code)**

---

Turn familiar quotes into brand slogans with a structured language-model workflow: **quote matching → structure breakdown → vocabulary replacement → remix generation**.

## Explore the code

| Workflow | Entry point |
| :--- | :--- |
| Single-pass prompting with multiple samples | [`1stage_multi_pass.py`](1stage_multi_pass.py) |
| Quote selection followed by slogan remixing | [`2stage.py`](2stage.py) · [`run_2stage.sh`](run_2stage.sh) |
| Explicit four-stage generation | [`scripts/4stage.py`](scripts/4stage.py) · [`run_4stage.sh`](run_4stage.sh) |
| Slogan datasets | [`dataset_slogans.csv`](dataset_slogans.csv) · [`dataset_slogans-v2.csv`](dataset_slogans-v2.csv) |
| Fine-tuning scripts | [`scripts/finetune_lora.py`](scripts/finetune_lora.py) · [`scripts/deepspeed_finetune_lora.py`](scripts/deepspeed_finetune_lora.py) |

## Getting started

```bash
git clone https://github.com/yangziao56/qwen-lora-finetune.git
cd qwen-lora-finetune
pip install -r requirements.txt
```

Choose a generation workflow above, then configure the model, brand, persona, and output settings in its entry point or launcher.

## Human evaluation

The companion [slogan-survey-web](https://github.com/yangziao56/slogan-survey-web) repository contains the web interface for slogan preference surveys.

---

<details>
<summary><strong>Citation</strong></summary>

```bibtex
@misc{yang2026sloganremix,
  title = {Recontextualizing Famous Quotes for Brand Slogan Generation},
  author = {Yang, Ziao and Chen, Zizhang and Zhang, Lei and Liu, Hongfu},
  year = {2026},
  eprint = {2602.06049},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2602.06049}
}
```

</details>
