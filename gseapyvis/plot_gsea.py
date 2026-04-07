#!/usr/bin/env python3
from matplotlib import colors

import os
import joblib
import pandas as pd
import numpy as np
from lets_plot import *
import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)


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


@app.command(name="plot")
def plot_gsea_result(
    pkl_file: str = typer.Argument(..., help="GSEAPy prerank result .pkl file path"),
    output_file: str = typer.Option("gsea_plot.html", "-o", "--output", help="Output file path (.html, .svg, .png, .pdf)"),
    term_index: int = typer.Option(1, "-t", "--term-index", help="Index of the term to plot (starting from 0)"),
    to_buffer: bool = typer.Option(False, help="Return plot as BytesIO buffer instead of saving to file"),
):
    """
    Plot GSEA result from a .pkl file.
    """
    if not os.path.exists(pkl_file):
        console.print(f"[bold red]❌ File not found:[/bold red] {pkl_file}")
        raise typer.Exit(code=1)

    valid_exts = {".html", ".htm", ".svg", ".png", ".pdf", ".jpg", ".jpeg"}
    ext = os.path.splitext(output_file)[-1].lower()
    if ext not in valid_exts:
        console.print(f"[bold red]❌ Invalid output format:[/bold red] {ext}")
        console.print("Supported formats: [bold green].html, .svg, .png, .pdf, .jpg, .jpeg[/bold green]")
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
        nes = round(pre_res.results[term]["nes"], 4)
        pval = round(pre_res.results[term]["pval"], 4)
        fdr = round(pre_res.results[term]["fdr"], 4)
        matched_genes = pre_res.results[term]["matched_genes"].split(";")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to extract term at index {term_index}[/bold red]")
        console.print(f"[red]Details:[/red] {e}")
        raise typer.Exit(code=1)

    hits_data = pd.DataFrame({
        "hits": hits,
        "gene": matched_genes
    })

    x_position = 0.75 * max(resdata["index"])
    if nes > 0:
        y_position = 0.75 * max(resdata["res"])
    else:
        y_position = -0.05

    p1 = (ggplot(resdata, aes(x='index', y='res')) +
          geom_line(color="green", show_legend=False, tooltips='none') +
          geom_hline(yintercept=0, color="grey", size=0.5, linetype="dashed") +
          geom_point(data=hits_data,
                     mapping=aes(x='hits', y=[resdata['res'][i] for i in hits_data['hits']]),
                     color="red", size=0.5,
                     tooltips=layer_tooltips().title('@gene')) +
          geom_text(x=x_position, y=y_position,
                    label=f"nes:{nes}\npval:{pval}\nfdr:{fdr}",
                    size=8, hjust=0, vjust=1, fontface="italic", lineheight=1.3) +
          theme_bw() +
          theme(axis_text_x=element_blank(),
                legend_position="none",
                axis_ticks=element_blank(),
                panel_grid_major=element_blank()) +
          xlab('') + ylab('Running Enrichment Score ') +
          ggtitle(term) +
          scale_x_continuous(expand=(0, 0)) +
          scale_y_continuous(expand=(0, 0)))

    gsdata = pd.DataFrame({
        "position": [0] * len(resdata),
    })
    gsdata["Description"] = term
    gsdata.loc[hits, "position"] = 1
    heatmap_data = compute_heat_blocks(gsdata)

    p2 = (ggplot() +
          geom_rect(aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="col"), data=heatmap_data) +
          geom_vline(aes(xintercept="hits"), color='black', data=hits_data) +
          scale_fill_identity() +
          scale_y_continuous(expand=(0, 0)) +
          scale_x_continuous(expand=(0, 0)) +
          theme(legend_position="none",
                axis_text=element_blank(),
                axis_ticks=element_blank(),
                axis_title=element_blank(),
                panel_background=element_blank()) +
          xlab("Rank in Ordered Dataset"))

    combined = gggrid([p1, p2], ncol=1, align=True, heights=[0.7, 0.1], vspace=-20) + ggsize(700, 500)

    if to_buffer:
        import io
        import tempfile
        fd, tmppath = tempfile.mkstemp(suffix='.svg')
        os.close(fd)
        ggsave(combined, tmppath)
        with open(tmppath, 'rb') as f:
            buf = io.BytesIO(f.read())
        os.unlink(tmppath)
        buf.seek(0)
        console.print("[bold green]✅ Plot generated in memory (BytesIO)[/bold green]")
        return buf
    else:
        output_dir = os.path.dirname(output_file) or '.'
        output_name = os.path.basename(output_file)
        ggsave(combined, output_name, path=output_dir)
        console.print(f"[bold green]✅ Plot saved to:[/bold green] [underline]{output_file}[/underline]")


if __name__ == "__main__":
    app()
