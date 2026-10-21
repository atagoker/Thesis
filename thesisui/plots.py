"""Matplotlib plots embedded directly inside native Qt widgets."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy

from thesisui import theme as t


class Plot(FigureCanvasQTAgg):
    def __init__(self, title="", xlabel="", ylabel="", minimum_height=210):
        self.figure = Figure(figsize=(6, 2.5), dpi=100, facecolor=t.PANEL, layout="constrained")
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self.setMinimumHeight(minimum_height)
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.reset()

    def reset(self):
        self.ax.clear()
        self.ax.set_facecolor(t.PANEL)
        self.ax.tick_params(colors=t.MUTED, labelsize=8, length=3)
        for side, spine in self.ax.spines.items():
            spine.set_color(t.BORDER)
            spine.set_visible(side in ("bottom", "left"))
        self.ax.set_title(self.title, loc="left", color=t.TEXT, fontsize=10, pad=12, weight="normal")
        self.ax.set_xlabel(self.xlabel, color=t.MUTED, fontsize=8)
        self.ax.set_ylabel(self.ylabel, color=t.MUTED, fontsize=8)
        self.ax.grid(axis="y", color=t.BORDER, alpha=.65, linewidth=.5)

    def placeholder(self, text):
        self.reset()
        self.ax.text(.5, .5, text, color=t.MUTED, ha="center", va="center", fontsize=10, transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_idle()

    def series(self, lines, ylim=None, percent=False):
        self.reset()
        for label, x, y, color in lines:
            self.ax.plot(x, y, color=color, lw=2.1, label=label)
        if ylim:
            self.ax.set_ylim(*ylim)
        if percent:
            from matplotlib.ticker import PercentFormatter
            self.ax.yaxis.set_major_formatter(PercentFormatter(1))
        if "epoch" in self.xlabel.lower():
            from matplotlib.ticker import MaxNLocator
            self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
        if len(lines) > 1:
            legend = self.ax.legend(loc="upper right", fontsize=8, facecolor=t.PANEL, edgecolor=t.BORDER, labelcolor=t.TEXT)
            legend.get_frame().set_alpha(.95)
        self.draw_idle()

    def waveform(self, info, cursor=None):
        self.reset()
        self.ax.plot(info["wave_t"], info["wave_y"], color=t.CYAN, lw=.7, alpha=.95, rasterized=True)
        self.ax.set_xlim(0, info["duration"])
        if cursor is not None:
            self.ax.axvline(cursor, color=t.AMBER, lw=1.4)
        self.draw_idle()

    def scores(self, values, binary=True):
        self.reset()
        x = np.arange(1, len(values) + 1)
        self.ax.plot(x, values, color=t.PURPLE, lw=2.1, marker="o", ms=3.5)
        self.ax.fill_between(x, values, color=t.PURPLE, alpha=.13)
        if binary:
            self.ax.axhline(.5, color=t.AMBER, lw=.8, ls="--")
        self.ax.set_ylim(-.03, 1.03)
        self.ax.set_xlim(.5, max(2, len(values) + .5))
        self.draw_idle()

    def comparison(self, summary, target=None):
        from thesisui.data import METHOD_NAMES
        self.reset()
        methods = list(METHOD_NAMES)
        colors = [t.BLUE, t.AMBER, t.GREEN, t.PURPLE]
        if target is None:
            values = summary.get("overall_target_macro_f1", {})
        else:
            values = {r["method"]: r["macro_f1_mean"] for r in summary.get("summary", [])
                      if r["scope"] == "target" and r["adaptation_target"] == target}
        if not values:
            self.placeholder("Published experiment metrics are not available yet.")
            return
        heights = [100 * values.get(m, 0) for m in methods]
        self.ax.bar(range(4), heights, color=colors, width=.5, alpha=1.)
        for i, value in enumerate(heights):
            self.ax.text(i, value + 3, f"{value:.2f}%", ha="center", color=t.TEXT, fontsize=10)
        self.ax.set_xticks(range(4), [METHOD_NAMES[m] for m in methods], color=t.MUTED, fontsize=9)
        self.ax.set_ylim(0, 114)
        self.ax.set_yticks([0, 25, 50, 75, 100])
        self.draw_idle()
