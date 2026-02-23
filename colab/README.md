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

## 2) Set Gemini key
```python
import os
os.environ["GEMINI_API_KEY"] = "<YOUR_KEY>"
```

## 3) Run the project
```bash
bash scripts/run_all.sh --mode full
```

## 4) Build PDFs
```bash
bash scripts/make_pdfs.sh
```

## 5) Push results
```bash
git add .
git commit -m "Safety Delta Atlas: outputs + writeup"
git push
```
