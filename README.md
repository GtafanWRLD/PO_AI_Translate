# GPT PO Translator 

This is a translation automation tool that batch-translates `.po` files using OpenAI's GPT-3.5 or Claude APIs. It is optimized for localizing game UI strings, particularly from Russian to English.

---

## 🔧 Features

- Translates `.po` files in bulk with GPT-3.5
- Skips English and empty strings automatically
- Multi-threaded batch processing (configurable)
- Auto-detects Cyrillic and avoids unnecessary API calls
- Detects strings that likely don't need translation using heuristics

---

## 🗂 Folder Structure

```
project/
│
├── po_translator_gpt3.5_claude.py
├── input/             ← Put your `.po` files here
└── output/            ← Translated files will be saved here
```

---

## ⚙️ Requirements

- Python 3.7+
- `openai`, `tqdm`, `langdetect`

Install dependencies:

```bash
pip install openai tqdm langdetect
```

Also make sure you have `key.txt` in the same folder containing your OpenAI API key.

---

## 🚀 How To Use

1. Run the script:
   ```bash
   python po_translator_gpt3.5.py
   ```

2. If `input/` folder doesn't exist, it will be created.

3. Drop `.po` files into the `input/` folder.

4. Script will process and generate output in `output/`.

---

## 📈 Stats Tracked

- Total blocks processed
- Skipped English strings
- Skipped empty strings
- Translated Russian strings
- API calls saved

---

## 📄 License

MIT License
