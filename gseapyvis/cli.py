#!/usr/bin/env python3
import matplotlib
from matplotlib import colors
matplotlib.use("Agg")

import os
import joblib
import pandas as pd
import numpy as np
from plotnine import *
import patchworklib as pw
import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion_option=False)

def compute_heat_blocks(gsdata, htCol=("red", "blue"), htHeight=1.0):
    all_blocks = []

    for setid in gsdata["Description"].unique():
        tmp = gsdata[gsdata["Description"] == setid].copy()

        rev_pos = tmp["position"].values[::-1]
        rev_cumsum = np.cumsum(rev_pos)

        v = np.linspace(1, rev_pos.sum(), 9)
        inv = np.searchsorted(v, rev_cumsum, side="right")

        if inv.min() == 0:
            inv += 1

        tmp = tmp.reset_index(drop=True)
        tmp["inv"] = inv

        tmp["group"] = (tmp["inv"] != tmp["inv"].shift()).cumsum()

        blocks = []
        for _, g in tmp.groupby("group"):
            xmin = g.index.min()
            xmax = g.index.max() + 1
            color_idx = g["inv"].iloc[0]
            blocks.append({
                "xmin": xmin,
                "xmax": xmax,
                "ymin": 0,
                "ymax": htHeight,
                "col": color_idx,
                "Description": setid
            })

        cmap = colors.LinearSegmentedColormap.from_list("custom_gradient", [htCol[0], "white", htCol[1]])
        color_list = [colors.to_hex(cmap(i / 10)) for i in range(10)]

        block_df = pd.DataFrame(blocks)
        block_df["col"] = block_df["col"] - 1
        block_df["col"] = block_df["col"].apply(lambda i: color_list[i])

    return block_df

@app.command()
def plot_gsea_result(
    pkl_file: str = typer.Argument(..., help="GSEAPy prerank result .pkl file path"),
    output_file: str = typer.Option("gsea_plot.png", "-o", "--output", help="Output image file path"),
    term_index: int = typer.Option(1, "-t", "--term-index", help="Index of the term to plot (starting from 0)"),
    to_buffer: bool = typer.Option(False, help="Return plot as BytesIO buffer instead of saving to file")
):
    """
    Plot GSEA result from a .pkl file.
    """
    if not os.path.exists(pkl_file):
        console.print(f"[bold red]❌ File not found:[/bold red] {pkl_file}")
        raise typer.Exit(code=1)

    valid_exts = {".png", ".pdf", ".jpg", ".jpeg"}
    ext = os.path.splitext(output_file)[-1].lower()
    if ext not in valid_exts:
        console.print(f"[bold red]❌ Invalid output format:[/bold red] {ext}")
        console.print("Supported formats: [bold green].png, .pdf, .jpg, .jpeg[/bold green]")
        raise typer.Exit(code=1)

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
        raise typer.Exit(code=1)

    if nes > 0:
        position_top = max(resdata["res"]) * 0.70
    else:
        position_top = (max(resdata["res"]) + min(resdata["res"])) / 2 * 0.8
    position_left = len(resdata) * 0.70

    p1 = (ggplot(resdata, aes(x='index', y='res')) +
          geom_line(color="green") +
          geom_hline(yintercept=0, color="grey", size=0.5, linetype="dashed") +
          theme_bw() +
          theme(axis_text_x=element_blank(),
                legend_position="none",
                axis_ticks=element_blank(),
                panel_background=element_blank(),
                panel_grid_major=element_blank()) +
          annotate("text", x=position_left, y=position_top, label=f"nes = {nes:.4f}", color="black", size=10, ha='left') +
          annotate("text", x=position_left, y=position_top-0.03, label=f"pval = {pval:.4f}", color="black", size=10, ha='left') +
          annotate("text", x=position_left, y=position_top-0.06, label=f"Fdr = {fdr:.4f}", color="black", size=10, ha='left') +
          xlab('') + ylab('Running Enrichment Score ') +
          ggtitle(term) +
          scale_x_continuous(expand=(0, 0)) +
          scale_y_continuous(expand=(0, 0)))

    gsdata = pd.DataFrame({
        "position": [0] * len(resdata),
    })
    gsdata["Description"] = term
    gsdata.loc[pre_res.results[term]["hits"], "position"] = 1
    heatmap_data = compute_heat_blocks(gsdata)

    p2 = (ggplot() +
          geom_rect(aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="col"), data=heatmap_data) +
          geom_vline(xintercept=hits, color='black') +
          scale_fill_identity() +
          scale_y_continuous(expand=(0, 0)) +
          scale_x_continuous(expand=(0, 0)) +
          theme(legend_position="none",
                axis_text=element_blank(),
                axis_ticks=element_blank(),
                axis_title=element_blank(),
                panel_background=element_blank()) +
          xlab("Rank in Ordered Dataset"))

    f1 = pw.load_ggplot(p1, figsize=(5, 3))
    f2 = pw.load_ggplot(p2, figsize=(5, 0.5))
    pw_plot = pw.vstack(f2, f1, margin=0.05)

    if to_buffer:
        import io
        buf = io.BytesIO()
        pw_plot.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        console.print("[bold green]✅ Plot generated in memory (BytesIO)[/bold green]")
        return buf
    else:
        pw_plot.savefig(output_file)
        console.print(f"[bold green]✅ Plot saved to:[/bold green] [underline]{output_file}[/underline]")

if __name__ == "__main__":
    app()

