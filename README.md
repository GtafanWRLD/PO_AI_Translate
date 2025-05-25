# MirTankov AI Powered PO Translator

A Python tool to batch-translate .po (Portable Object) localization files for games (Used for MirTankov localization project), optimized for mixed English/Russian/Chinese projects.

- ⚡ Only translates lines that are NOT already in English—saving time and reduces API requests !
- 🏷️ Preserves all in-game tags and formatting (like %(tank)s, {player}, etc.)
- 🪖 Uses World of Tanks terminology by default, can be changed by the user.
- 📁 Processes every .po file in your `input/` folder, outputs to `translated/`.

---

## Features

- Batch translates `.po` files from Russian or Chinese to English using OpenAI GPT-3.5-turbo (model can be changed, currently this is the cheapest one)
- Automatically skips already-English lines
- Maintains line breaks and multi-line formatting
- Auto-creates `input/` and `translated/` folders on first run

---
## Requirements

- Python 3.8+
- pip packages: `openai polib tqdm send2trash langdetect`
- IMPORTANT: OpenAI API key (set as env variable `OPENAI_API_KEY` or in a `key.txt` file)

Install requirements:
```sh
pip install openai polib tqdm send2trash langdetect
```

---

## Usage

1. Place your `.po` files in the `input/` folder (auto-created if missing).
2. Add your OpenAI API key to an environment variable `OPENAI_API_KEY` or create a `key.txt` in the script's directory.
3. Run the script:

```sh
python your_script_name.py
```

4. All translated files will appear in the `translated/` folder, with the same filenames.

**Already-English lines will not be re-translated.**

---

## Configuration

Edit these at the top of the script:

```python
OPENAI_MODEL = "gpt-3.5-turbo"  # (recommended: do not change unless you want to pay more)
BATCH_SIZE = 50                  # Lower if you see output mismatch; raise for speed if no issues, values up to 80 are safe
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "translated"
```

---

## FAQ / Troubleshooting

**Q: Why are some placeholders like %(tank)s not working?**  
> This tool  should mask and restore all such placeholders, so if you spot issues, create an issue report.

**Q: My translated file is missing lines or seems misaligned!**  
> Lower `BATCH_SIZE` to 10 or even 1 for stubborn files. GPT sometimes skips or merges lines in big batches (not the case for MirTankov).

**Q: Does this tool handle big .po files?**  
> Yes, but watch your OpenAI API usage. Large files = higher costs. Already-English lines are skipped automatically. The limit is around 200k characters per file.

---

## Credits
- Script & README: [Gtafan]
- Powered by [OpenAI GPT-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5)
