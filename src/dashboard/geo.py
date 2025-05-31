# src/dashboard/geo.py

import numpy as np
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar,
    HoverTool, Range1d, FixedTicker
)
from shapely.geometry import Polygon, MultiPolygon
from typing import List, Dict

class GeoDash:
    def __init__(self, world_gdf, palette: List[str]):
        self.world    = world_gdf
        self.palette  = palette
        self._built   = False
        self._build_figure()

    def _build_figure(self):
        p = figure(
            toolbar_location=None, tools="hover",
            width=950, height=430, match_aspect=True
        )
        hover = p.select_one(HoverTool)
        hover.tooltips = [("Country","@NAME"),("Value","@current{0.00}")]
        p.axis.visible          = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.background_fill_color = None
        p.outline_line_color    = None
        p.y_range = Range1d(-95, 90)

        self.color_mapper = LinearColorMapper(
            palette=self.palette, low=0, high=1, nan_color="lightgray"
        )
        self.color_bar = ColorBar(
            color_mapper=self.color_mapper,
            location=(0,10), orientation='horizontal',
            width=600, height=20, ticker=FixedTicker(ticks=[]),
            major_label_text_font_style="normal",
            title_text_font_size="12pt"
        )
        p.add_layout(self.color_bar, 'below')
        self.fig = p

    def load_indicator(self,
                       years: List[str],
                       all_values: Dict[str, List[float]]):
        """
        Build a ColumnDataSource with one patch per Polygon (or per part of a MultiPolygon).
        all_values[yr] is a list of normalized_value *by row index* in world_gdf.
        """
        xs: List[List[float]] = []
        ys: List[List[float]] = []
        # for each year we’ll build a flat list, one entry per polygon patch:
        year_cols: Dict[str, List[float]] = {yr: [] for yr in years}

        for idx, row in self.world.iterrows():
            geom = row.geometry
            # grab the value arrays for this country (row)
            vals_for_row = { yr: all_values[yr][idx] for yr in years }

            # helper to append one Polygon’s exterior ring
            def _append_poly(poly: Polygon):
                x, y = poly.exterior.coords.xy
                xs.append(list(x))
                ys.append(list(y))
                # replicate this country’s value for every year
                for yr in years:
                    year_cols[yr].append(vals_for_row[yr])

            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    _append_poly(poly)
            elif isinstance(geom, Polygon):
                _append_poly(geom)
            else:
                # sometimes there are GeometryCollections—skip them or handle as needed
                continue

        # now build the CDS
        data = {
            "xs": xs,
            "ys": ys,
            **{yr: year_cols[yr] for yr in years},
            "current": year_cols[years[0]]
        }

        self.source = ColumnDataSource(data)
        # draw all patches once
        self.fig.patches(
            "xs", "ys", source=self.source,
            fill_color={"field":"current", "transform": self.color_mapper},
            line_color="black", line_width=0.5
        )
        self._built = True

    def set_year(self, year: str):
        """Instantly swap in the chosen year’s values."""
        if not self._built:
            return
        self.source.patch({
            'current': [(slice(None), self.source.data[year])]
        })

    def clear(self):
        """Reset everything to NaN."""
        if not self._built:
            return
        n = len(self.source.data["xs"])
        self.source.patch({
            'current': [(slice(None), [np.nan]*n)]
        })
