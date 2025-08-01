# 📝 PO AI Translator

A powerful localization automation tool for translating `.po` files from **Russian into over 30 languages** using **OpenAI GPT-3.5**. Designed to preserve formatting, optimize performance, and reduce redundant translations.

---


## ⚠️ WARNING

- I am not actively supporting development of this version — check out the cheaper Deepseek variant here: [PoDeepSeekTranslate](https://github.com/GtafanWRLD/PoDeepSeekTranslate).

---


## ✅ Features

1. **🌍 Multi-language support**
   - Translate `.po` files from Russian into any language, selected dynamically at runtime.
   - Supports 30+ languages with customizable language code and name.

2. **🚀 Optimized Translation Pipeline**
   - Uses OpenAI GPT-3.5 for batch translations.
   - Batching, multithreading, and retry logic for efficiency.

3. **📦 Intelligent Skipping**
   - Automatically skips strings already in English or empty.

4. **⚙️ User Configuration**
   - First-time setup wizard to configure batch size, max threads, and more.
   - All settings saved to `config.json` (except language, chosen each run).

5. **🛠 Format-Preserving Output**
   - Handles multiline `msgstr`, variables like `%d`, `{0}`, `%(goal)s`, and escape characters.
   - Maintains exact formatting required by using gettext .

---

## 📁 Usage

1. Place your `.po` files into the `input/` folder.
2. Make sure `key.txt` contains your OpenAI API key (one line, starts with `sk-`).
3. Run the .exe file
4. Translated `.po` files will appear in the `output/` folder.

---

## 🛠 Requirements

- Python 3.7+
- Required packages:
  ```
  pip install openai tqdm langdetect
  ```

---

## 📄 Notes

- API key must be stored in a `key.txt` file.
- Language selection is interactive at every launch.
- Designed for gaming localization: uses terminology like battle, tank, armor, etc.


---




## 📃 License

MIT License
