# 🎯 GSEA Plotting Tool

This is a Python command-line tool for visualizing [GSEAPy](https://github.com/zqfang/GSEApy) prerank results from `.pkl` files using `lets-plot`. The tool creates an enrichment plot and saves it as an image file.

---

## 📦 Features

- 📈 Line plot of RES (Running Enrichment Score)
- 🔦 Hit indicator lines and colored gradient background
- 🛠 Easy CLI with [Fire](https://github.com/google/python-fire)
- 🎨 High-quality output using `lets-plot`

---

## 🚀 Installation

### 🔧 Clone the repo:

```bash
git clone https://github.com/xpf10/gseapyvis.git
cd gseapyvis

python .\gseapyvis\cli.py .\data\test.pkl --output_file gsea_result.pdf
```
windows:
```
poetry  install
Invoke-Expression (poetry env activate)
gseapyvis.cmd .\data\test.pkl 
```
Linux:
```
poetry install
poetry env use python3
gseapyvis .\data\test.pkl
```