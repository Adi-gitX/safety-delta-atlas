# Colab Run Guide (T4/L4)

## 1) Clone + env
```bash
git clone https://github.com/Adi-gitX/safety-delta-atlas.git
cd safety-delta-atlas
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Authenticate Hugging Face (required for Gemma)
```bash
hf auth login
```

## 3) Set Gemini key
```python
import os
os.environ["GEMINI_API_KEY"] = "<YOUR_KEY>"
```

## 4) Run the project
```bash
bash scripts/run_all.sh --mode full
```

## 5) Build PDFs
```bash
bash scripts/make_pdfs.sh
```

## 6) Push results
```bash
git add .
git commit -m "Safety Delta Atlas: outputs + writeup"
git push
```
