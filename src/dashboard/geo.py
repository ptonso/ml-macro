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
        self.world = world_gdf
        self.palette = palette.copy()
        self._built = False
        self.source = None
        self._build_figure()

    def _build_figure(self):
        p = figure(
            toolbar_location=None, 
            tools="",
            width=1100, height=530, match_aspect=True
        )
        
        p.axis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.background_fill_color = None
        p.outline_line_color = None
        p.y_range = Range1d(-95, 90)

        self.color_mapper = LinearColorMapper(
            palette=self.palette, 
            low=0, 
            high=1, 
            nan_color="lightgray"
        )
        
        center_x = 1100 / 2
        bar_width = 700
        x_offset = center_x - (bar_width / 2)
        
        self.color_bar = ColorBar(
            color_mapper=self.color_mapper,
            location=(x_offset, 10), 
            orientation='horizontal',
            width=bar_width, 
            height=20, 
            ticker=FixedTicker(ticks=[]),
            major_label_text_font_style="normal",
            title_text_font_size="12pt",
            major_tick_line_color="black", 
            bar_line_color="black",         
            border_line_color=None,
            major_label_text_color="black",
            title_text_font_style="bold",
            title_text_color="black",
            padding=5,
            label_standoff=12,
            title_standoff=10
        )
        
        p.add_layout(self.color_bar, 'below')
        self.fig = p

    def load_indicator(self,
                       years: List[str],
                       all_values: Dict[str, Dict[str, List[float]]]):
        if self._built:
            self.fig.renderers = []
            self.source = None
            
        xs: List[List[float]] = []
        ys: List[List[float]] = []
        year_cols: Dict[str, List[float]] = {yr: [] for yr in years}
        raw_cols: Dict[str, List[float]] = {yr: [] for yr in years}
        display_name_cols: Dict[str, List[str]] = {yr: [] for yr in years}

        for idx, row in self.world.iterrows():
            geom = row.geometry
            name = row["NAME"]
            vals_for_row = {yr: all_values[yr]["values"][idx] for yr in years}
            raw_vals_for_row = {yr: all_values[yr]["raw_values"][idx] for yr in years}
            names_for_row = {yr: all_values[yr]["display_names"][idx] 
                            if idx < len(all_values[yr]["display_names"]) else name 
                            for yr in years}

            def _append_poly(poly: Polygon):
                x, y = poly.exterior.coords.xy
                xs.append(list(x))
                ys.append(list(y))
                for yr in years:
                    year_cols[yr].append(vals_for_row[yr])
                    raw_cols[yr].append(raw_vals_for_row[yr])
                    display_name_cols[yr].append(names_for_row[yr])

            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    _append_poly(poly)
            elif isinstance(geom, Polygon):
                _append_poly(geom)
            else:
                continue

        data = {
            "xs": xs,
            "ys": ys,
            **{yr: year_cols[yr] for yr in years},
            **{f"raw_{yr}": raw_cols[yr] for yr in years},
            **{f"display_{yr}": display_name_cols[yr] for yr in years},
            "current": year_cols[years[0]],
            "raw_value": raw_cols[years[0]],
            "display_name": display_name_cols[years[0]]
        }

        self.source = ColumnDataSource(data)
        
        patches = self.fig.patches(
            "xs", "ys", source=self.source,
            fill_color={"field":"current", "transform": self.color_mapper},
            line_color="black", line_width=0.5
        )
        
        hover = HoverTool(
            tooltips=[
                ("Country", "@display_name"),
                ("Value", "@raw_value{0,0.00}")
            ],
            formatters={"@raw_value": "numeral"},
            renderers=[patches]
        )
        self.fig.add_tools(hover)
        
        self._built = True

    def set_year(self, year: str):
        if not self._built or self.source is None:
            return
            
        self.source.patch({
            'current': [(slice(None), self.source.data[year])],
            'raw_value': [(slice(None), self.source.data[f"raw_{year}"])],
            'display_name': [(slice(None), self.source.data[f"display_{year}"])]
        })

    def clear(self):
        if not self._built or self.source is None:
            return
            
        n = len(self.source.data["xs"])
        self.source.patch({
            'current': [(slice(None), [np.nan]*n)],
            'raw_value': [(slice(None), [np.nan]*n)],
            'display_name': [(slice(None), ["Unknown"]*n)]
        })