#!/usr/bin/env python

import argparse
import joblib
import pandas as pd
import numpy as np
from plotnine import *
import patchworklib as pw

def plot_gsea_result(pkl_file, output_file, term_index=1):
    pre_res = joblib.load(pkl_file)
    terms = pre_res.res2d.Term
    term = terms[term_index]

    hits = pre_res.results[term]["hits"]
    resdata = pd.DataFrame({
        "res": pre_res.results[term]["RES"],
        "index": range(len(pre_res.results[term]["RES"]))
    })

    # Line plot
    p1 = (ggplot(resdata, aes(x='index', y='res')) +
          geom_line() +
          theme_bw() +
          theme(axis_text_x=element_blank(),
                legend_position="none",
                axis_ticks=element_blank(),
                panel_background=element_blank()) +
          xlab('') + ylab('') +
          ggtitle(term) +
          scale_x_continuous(expand=(0, 0)) +
          scale_y_continuous(expand=(0, 0)))

    # Gradient background + vertical lines
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
          xlab('') + ylab('') +
          guides(color=False, fill=False, shape=False, size=False))

    # Combine
    f1 = pw.load_ggplot(p1, figsize=(5, 3))
    f2 = pw.load_ggplot(p2, figsize=(5, 0.5))
    pw.vstack(f2, f1, margin=0).savefig(output_file)
    print(f"✅ Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="GSEA result plot tool using plotnine and patchworklib.")
    parser.add_argument("pkl_file", help="Path to the .pkl file (GSEAPy prerank result)")
    parser.add_argument("-o", "--output", default="gsea_plot.png", help="Output file path (e.g. gsea_plot.png)")
    parser.add_argument("-i", "--index", type=int, default=1, help="Index of term to plot (default: 1)")

    args = parser.parse_args()
    plot_gsea_result(args.pkl_file, args.output, args.index)

if __name__ == "__main__":
    main()
