# gseapyvis

<p align="center">
  <img src="assets/logo.png" alt="gseapyvis Logo" width="200" height="200">
</p>

GSEA enrichment plot visualization for [GSEAPy](https://github.com/zqfang/GSEApy) prerank results, built with [lets-plot](https://lets-plot.org/).

## Features

- Running Enrichment Score line plot with hit markers
- Colored gradient heatmap strip
- Gene name tooltips on hit points
- Jupyter inline display & CLI export

## Installation

```bash
git clone https://github.com/xpf10/gseapyvis.git
cd gseapyvis
uv sync
```

## Usage

### Jupyter Notebook

```python
import gseapy as gp
from lets_plot import *
LetsPlot.setup_html()

from gseapyvis import gsea_plot

# Run GSEA
pre_res = gp.prerank(
    rnk=rnk,
    gene_sets="KEGG_2019_Mouse",
    permutation_num=100,
    outdir=None,
    no_plot=True,
)

# Plot — displays inline automatically
term = pre_res.res2d.Term.iloc[0]
gsea_plot(pre_res.results, term)

# With gene score tooltips (optional)
rnk_df = pd.DataFrame({"gene": rnk.index, "score": rnk.values})
gsea_plot(pre_res.results, term, rnk=rnk_df)

# Save to file
p = gsea_plot(pre_res.results, term)
ggsave(p, "gsea_plot.html")
```

### CLI

```bash
# Default: HTML output, first term
python ./gseapyvis/cli.py plot ./data/test.pkl

# Specify output format (.html, .svg, .png, .pdf)
python ./gseapyvis/cli.py plot ./data/test.pkl -o gsea_result.svg

# Select term by index
python ./gseapyvis/cli.py plot ./data/test.pkl -t 1 -o gsea_result.html
```

Or after `uv sync` / `poetry install`:

```bash
gseapyvis plot ./data/test.pkl -o gsea_result.html
```
