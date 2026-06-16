# Installation Guide

## Option A — Google Colab (recommended)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Clone this repo:
```bash
!git clone https://github.com/k2406/compose-axion-ax-hackathon-2026-full-solution-template.git
%cd compose-axion-ax-hackathon-2026-full-solution-template/src
```
3. Install dependencies:
```bash
!pip install -r requirements.txt -q
```
4. Run the app:
```python
import app
ui = app.build_ui()
ui.launch(share=True)
```

## Option B — Local (Mac/Linux)

```bash
git clone https://github.com/k2406/compose-axion-ax-hackathon-2026-full-solution-template.git
cd compose-axion-ax-hackathon-2026-full-solution-template/src
python -m venv env && source env/bin/activate
pip install -r requirements.txt
python app.py
```

## Running the Benchmark

```bash
cd src
python evaluate.py
# Outputs KPIs to stdout and saves results.csv
```
