# RapidFire AI Winter Competition — SFT Dialogue Summarization (SAMSum)

This project runs an SFT (supervised fine-tuning) sweep for **dialogue summarization** (dialogue → short third-person summary) using **GPT-2 + LoRA**.



## Repository structure
- `rf_colab_sft_competition.ipynb` — main Colab notebook (data → sweep → eval → logging)
- `Fine-Tuning-Design-Summary.md` — experiment report / write-up
- `auxiliary/logs/` — TensorBoard event logs (`events.out.tfevents.*`) for loss/eval_loss/ROUGE/gen_len/etc.
- `auxiliary/screenshots/` — optional screenshots of key plots
- `README.md` — this file

**Tip - How to view metrics plots through TensorBoard without running experiments:** Run the "Directly View Results" section at the end of the Colab notebook.
