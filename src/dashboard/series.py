# src/dashboard/series.py

import random
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from typing import Dict, List
import pandas as pd

class SeriesDash:
    def __init__(self, years: List[str]):
        self.years: List[str] = years
        self.country_colors: Dict[str, str] = {}
        self.fig = self._build_figure()

    def _build_figure(self):
        p = figure(
            title="Trend for Top 10 Countries",
            toolbar_location="right",
            width=950, height=430,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        p.xaxis.major_label_orientation = 45
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "Value"
        p.xgrid.grid_line_color = "#dddddd"
        p.ygrid.grid_line_color = "#dddddd"
        return p

    def _get_country_color(self, country: str, idx: int) -> str:
        if country in self.country_colors:
            return self.country_colors[country]
        palette = Category20[20] if idx < 20 else None
        color = (
            palette[idx]
            if palette
            else f"#{random.randint(50,230):02x}"
                 f"{random.randint(50,230):02x}"
                 f"{random.randint(50,230):02x}"
        )
        self.country_colors[country] = color
        return color

    def update(
        self,
        selected_table: pd.DataFrame,
        all_precomputed: Dict[str, Dict[str, pd.DataFrame]],
        years: List[str],
        current_type: str
    ):
        """
        Plot the trend for the top 10 countries (from selected_table)
        over all given years, pulling each year's own 'table' from all_precomputed.
        """
        # Clear old renderers
        self.fig.renderers = []

        # Pick top 10 countries in the selected year
        top_countries = selected_table.head(10)["country"].tolist()

        # Prepare data containers
        plot_data = {
            "x": [], "y": [], "color": [],
            "country": [], "year": [], "value": []
        }

        # Build a line (and collect points) for each country
        for i, country in enumerate(top_countries):
            yrs, vals = [], []
            for yr in years:
                tbl = all_precomputed.get(yr, {}).get("table")
                if tbl is None:
                    continue
                row = tbl[tbl["country"] == country]
                if not row.empty:
                    yrs.append(yr)
                    vals.append(float(row["value"].iloc[0]))
            if len(yrs) > 1:
                color = self._get_country_color(country, i)
                # draw the line
                self.fig.line(
                    x=yrs, y=vals,
                    line_width=1.5,
                    color=color,
                    legend_label=country
                )
                # accumulate scatter data
                for y, v in zip(yrs, vals):
                    plot_data["x"].append(y)
                    plot_data["y"].append(v)
                    plot_data["color"].append(color)
                    plot_data["country"].append(country)
                    plot_data["year"].append(y)
                    plot_data["value"].append(v)

        # draw scatter
        src = ColumnDataSource(plot_data)
        scatter = self.fig.scatter(
            x="x", y="y",
            size=8, fill_color="color",
            line_color="black", line_width=1,
            source=src, alpha=0.8
        )

        # hover for the points
        hover = HoverTool(
            tooltips=[
                ("Country", "@country"),
                ("Year", "@year"),
                ("Value", "@value{0,0.00}")
            ],
            renderers=[scatter]
        )
        # replace any old HoverTools
        self.fig.tools = [
            t for t in self.fig.tools if not isinstance(t, HoverTool)
        ]
        self.fig.add_tools(hover)

        # only set legend props if items exist
        if self.fig.legend and self.fig.legend.items:
            self.fig.legend.click_policy = "hide"
            self.fig.legend.location = "top_right"

        # update title
        self.fig.title.text = f"Trend for Top 10 Countries ({current_type})"

    def clear(self):
        """Remove any existing renderers and reset title."""
        self.fig.renderers = []
        self.fig.title.text = "Trend for Top 10 Countries"
