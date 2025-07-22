#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")

import os
import joblib
import pandas as pd
import numpy as np
from plotnine import *
import patchworklib as pw
import fire
from rich.console import Console
from rich import print

console = Console()

def plot_gsea_result(pkl_file, output_file="gsea_plot.png", term_index=1):
    """
    Plot GSEA result from a .pkl file.

    Args:
        pkl_file (str): Path to the .pkl file (GSEAPy prerank result)
        output_file (str): Output file path (.png, .pdf, .jpg)
        term_index (int): Index of the term to plot (default: 1)
    """
    if not os.path.exists(pkl_file):
        console.print(f"[bold red]❌ File not found:[/bold red] {pkl_file}")
        return

    valid_exts = {".png", ".pdf", ".jpg", ".jpeg"}
    ext = os.path.splitext(output_file)[-1].lower()
    if ext not in valid_exts:
        console.print(f"[bold red]❌ Invalid output format:[/bold red] {ext}")
        console.print("Supported formats: [bold green].png, .pdf, .jpg[/bold green]")
        return

    console.print(f"[cyan]📂 Loading GSEA result from:[/cyan] {pkl_file}")
    pre_res = joblib.load(pkl_file)

    try:
        terms = pre_res.res2d.Term
        term = terms[term_index]
        hits = pre_res.results[term]["hits"]
        resdata = pd.DataFrame({
            "res": pre_res.results[term]["RES"],
            "index": range(len(pre_res.results[term]["RES"]))
        })
        nes = pre_res.results[term]["nes"]
        pval = pre_res.results[term]["pval"]
        fdr = pre_res.results[term]["fdr"]
    except Exception as e:
        console.print(f"[bold red]❌ Failed to extract term at index {term_index}[/bold red]")
        console.print(f"[red]Details:[/red] {e}")
        return

    position_left = len(resdata) * 0.70
    position_top = max(resdata["res"]) * 0.80

    p1 = (ggplot(resdata, aes(x='index', y='res')) +
          geom_line() +
          theme_bw() +
          theme(axis_text_x=element_blank(),
                legend_position="none",
                axis_ticks=element_blank(),
                panel_background=element_blank()) +
          annotate("text", x=position_left, y=position_top, label=f"nes = {nes:.4f}", color="blue", size=15,ha='left',) +
          annotate("text", x=position_left, y=position_top-0.05, label=f"pval = {pval:.4f}", color="blue", size=15,ha='left',) +
          annotate("text", x=position_left, y=position_top-0.1, label=f"Fdr = {fdr:.4f}", color="blue", size=15,ha='left',) +
          xlab('') + ylab('') +
          ggtitle(term) +
          scale_x_continuous(expand=(0, 0)) +
          scale_y_continuous(expand=(0, 0)))

    x_vals = np.linspace(0, len(resdata), len(resdata) * 10)
    bg_df = pd.DataFrame({'x': x_vals})
    bg_df['y'] = 0
    bg_df['fill'] = bg_df['x']

    p2 = (ggplot() +
          geom_vline(xintercept=hits, color='black') +
          xlim(0, len(resdata)-1) +
          scale_x_continuous(expand=(0, 0)) +
          geom_tile(data=bg_df, mapping=aes(x='x', y='y', fill='fill'),
                    width=0.1, height=10, alpha=0.4) +
          scale_fill_gradient(low="lightblue", high="purple") +
          theme(legend_position="none",
                axis_text=element_blank(),
                axis_ticks=element_blank(),
                axis_title=element_blank(),
                panel_background=element_blank()) +
          scale_y_continuous(expand=(0, 0)) +
          xlab('') + ylab('')+
          guides(color=False, fill=False, shape=False, size=False))

    f1 = pw.load_ggplot(p1, figsize=(5, 3))
    f2 = pw.load_ggplot(p2, figsize=(5, 0.5))
    pw.vstack(f2, f1, margin=0).savefig(output_file)

    console.print(f"[bold green]✅ Plot saved to:[/bold green] [underline]{output_file}[/underline]")

def main():
    fire.Fire(plot_gsea_result)

if __name__ == "__main__":
    main()
