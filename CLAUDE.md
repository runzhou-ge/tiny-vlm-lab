# CLAUDE.md

Coding rules for this repo:

1. **Keep it minimal.** No new abstractions, registries, factories, or config systems.
2. **No over-engineering.** If a plain function or a few lines does the job, do that.
3. **Use uv.** Manage dependencies with `uv sync` and run scripts with `uv run`.
4. **Update README.md** when behavior, commands, or data formats change.
5. **Do not add files** unless they are clearly necessary for the MVP to work.
6. **No distributed training, LoRA, Gradio, evaluation harnesses, or complex configs.**
7. **Prefer readable, linear code** over clever or highly abstracted code.
