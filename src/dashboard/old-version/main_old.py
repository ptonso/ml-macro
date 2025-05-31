import pandas as pd
import geopandas as gpd
from pathlib import Path
import pycountry
import numpy as np
import zipfile
import requests
import tempfile
import random
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from bokeh.io import curdoc
from bokeh.models import (
    GeoJSONDataSource, LinearColorMapper, ColorBar, Slider,
    Tabs, TabPanel, Div, HoverTool, Range1d, FixedTicker, Select,
    ColumnDataSource
)
from bokeh.palettes import Category20
from bokeh.plotting import figure
from bokeh.layouts import column, row

from src.logger import setup_logger, setup_logging
from src.preprocess.dataset import Dataset
from src.preprocess.result import ResultData

@dataclass
class WorldTimeConfig:
    shapefile_dir: Path
    year: str
    title: str = "Dashboard"
    log_file: str = "api.log"

def format_option_name(name: str) -> str:
    if name.endswith("_world_bank"):
        name = name[:-11]
    return name.replace("_", " ").replace("-", " ").title()

def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def make_blue_red_transparent_palette(
    blue_hex: str = "#324376",
    red_hex: str = "#6E0D25",
    steps: int = 10
) -> List[str]:
    half = steps // 2
    br, bg, bb = _hex_to_rgb(blue_hex)
    rr, rg, rb = _hex_to_rgb(red_hex)
    palette = []
    for i in range(half):
        alpha = 1.0 - (i / (half - 1)) * 0.8 if half > 1 else 1.0
        palette.append(f"rgba({br},{bg},{bb},{alpha:.2f})")
    for i in range(half):
        alpha = 0.2 + (i / (half - 1)) * 0.8 if half > 1 else 0.2
        palette.append(f"rgba({rr},{rg},{rb},{alpha:.2f})")
    return palette

def safe_log_transform(value: float) -> float:
    if pd.isna(value):
        return np.nan
    if value > 0:
        return np.log1p(value)
    if value < 0:
        return -np.log1p(abs(value))
    return 0.0

class MultiDashboard:
    def __init__(self, config: WorldTimeConfig):
        """Initialize dashboard and load configuration."""
        self.config = config
        setup_logging()
        self.logger = setup_logger(self.config.log_file)
        self.dataset = Dataset()
        res = self.dataset.get(
            ResultData(datadict=True, metadata=True)
        )

        self.data_dict: Dict[str, pd.DataFrame] = res.datadict or {}
        self.metadata: Optional[Dict[str, Any]] = res.metadata
        
        name_to_path: Dict[str, Path] = {v:k for k, v in res.path_names_dict.items()}

        self.sector_map: Dict[str, List[str]] = {}
        for indicator_name, sector in res.metadata.category_dict.items():
            p = name_to_path.get(indicator_name)
            if p:
                self.sector_map.setdefault(sector, []).append(p)

        self.world: Optional[gpd.GeoDataFrame] = None
        self.geosource: Optional[GeoJSONDataSource] = None
        self.df: Optional[pd.DataFrame] = None

        self.precomputed: Dict[str, Any] = {}
        self.years: List[str] = []
        self.current_year: str = self.config.year
        self.current_type: str = ""
        self.country_colors: Dict[str, str] = {}

        self.palette = make_blue_red_transparent_palette()
        self.color_mapper = LinearColorMapper(
            palette=self.palette, low=0, high=1, nan_color="lightgray"
        )

        self.title_div = Div(width=950, height=50)
        self.geo_plot = None
        self.individual_plot = None
        self.slider = None
        self.sector_select = None
        self.type_select = None
        self.color_bar = None

    def run(self) -> None:
        """Build and serve the dashboard layout."""
        self.logger.info("Starting dashboard")
        self._load_shapefile()
        self.title_div = Div(text="", width=950, height=50,
                             styles={"font-size": "24pt", "font-weight": "bold", "text-align": "center"})
        self.geo_plot = self._build_geo_figure()
        self.individual_plot = self._build_individual_components()
        sidebar = self._build_sidebar()
        self.color_bar = self._build_color_bar()
        self.geo_plot.add_layout(self.color_bar, 'below')

        tabs = Tabs(tabs=[
            TabPanel(child=column(self.geo_plot),      title="Geospatial"),
            TabPanel(child=column(self.individual_plot), title="Individual"),
        ])
        layout_ = column(row(sidebar, column(self.title_div, tabs)))
        self._set_map_to_grey()
        curdoc().add_root(layout_)
        curdoc().title = self.config.title

    def _load_shapefile(self) -> None:
        """Load or download and simplify the world shapefile."""
        shapefile_dir = self.config.shapefile_dir / "ne_110m_admin_0_countries"
        shp_path = shapefile_dir / "ne_110m_admin_0_countries.shp"
        if not shp_path.exists():
            self.logger.info("Downloading shapefile")
            url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            shapefile_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp.write(resp.content)
                with zipfile.ZipFile(tmp.name, 'r') as z:
                    z.extractall(shapefile_dir)
            Path(tmp.name).unlink()
        self.world = gpd.read_file(shp_path)
        self.world["geometry"] = self.world["geometry"].simplify(tolerance=0.01)

    def _build_sidebar(self):
        """Create sidebar widgets for sector, type, and year."""
        sector_opts = [("", "Select Sector")] + [
            (sector, format_option_name(sector))
            for sector in sorted(self.sector_map)
        ]
        self.sector_select = Select(options=sector_opts, width=230)
        self.sector_select.on_change("value", self._on_sector_change)

        self.type_select = Select(options=[("", "")], width=230)
        self.type_select.on_change("value", self._on_type_change)

        self.slider = Slider(start=2020, end=2023,
                             value=int(self.current_year),
                             step=1, width=230)
        self.slider.on_change("value", self._on_year_change)

        return column(
            Div(text="<b>Sector</b>"),
            self.sector_select,
            Div(text="<b>Type</b>"),
            self.type_select,
            Div(text="<b>Year</b>"),
            self.slider,
            width=250, background="#f5f5f5"
        )

    def _on_sector_change(self, attr, old, new) -> None:
        """Update types when sector changes."""
        if not new:
            self.type_select.options = [("", "Select Indicator")]
            self._reset_dashboard()
            return
        opts: List[tuple[str, str]] = [("", "Select Indicator")]
        for path in sorted(self.sector_map[new], key=lambda p: p.name):
            key = self.dataset.path_names_dict[path]
            label = format_option_name(key)
            opts.append((key, label))

        self.type_select.options = opts
        self._reset_dashboard()

    def _on_type_change(self, attr, old, new) -> None:
        """Load and prepare data when type changes."""
        if not new:
            self._reset_dashboard()
            return
        
        path = Path(new)
        self.current_type = format_option_name(path.stem)
        self._load_dataset(new)
        if self.df is None:
            self._reset_dashboard()
            return
        self._prepare_data()
        self._update_year_slider()
        self._update_data_for_year(self.current_year)
        self.title_div.text = f"{self.current_type.upper()} ({self.current_year})"

    def _on_year_change(self, attr, old, new) -> None:
        """Update plots when year slider moves."""
        self.current_year = str(new)
        if self.df is None:
            return
        self._update_data_for_year(self.current_year)
        self.title_div.text = f"{self.current_type.upper()} ({self.current_year})"

    def _reset_dashboard(self) -> None:
        """Clear data and reset views."""
        self.df = None
        self.precomputed.clear()
        self.country_colors.clear()
        self.geosource = None
        self._set_map_to_grey()
        self._clear_individual_dashboard()
        self.title_div.text = ""

    def _load_dataset(self, dataset_name: str) -> None:
        """Fetch the dataset via Dataset.get."""
        self.logger.info(f"Fetching data for: {dataset_name}")
        req = ResultData(datadict=True)
        res = self.dataset.get(req)
        data_dict = res.datadict or {}
        self.df = data_dict.get(dataset_name)
        if self.df is None:
            self.logger.error(f"No data for: {dataset_name}")

    def _prepare_data(self) -> None:
        """Compute log values, deciles and merge by year."""
        self.df.reset_index(inplace=True)
        df_melt = (
            self.df.melt(id_vars="date", var_name="country", value_name="value")
                .assign(iso_a3=lambda d: d["country"].map(self._get_iso3))
        )
        df_melt["log_value"] = df_melt["value"].apply(safe_log_transform)
        vals = df_melt["log_value"].dropna()
        deciles = np.nanpercentile(vals, np.arange(0, 101, 10)) if not vals.empty else [0]
        def assign_decile(x):
            if pd.isna(x):
                return np.nan
            idx = np.searchsorted(deciles, x, side='right') - 1
            return min(max(idx, 0), 9) / 9.0
        df_melt["normalized_value"] = df_melt["log_value"].apply(assign_decile)

        self.precomputed.clear()
        year_ints = sorted({d.year for d in self.df["date"]})
        self.years = [str(y) for y in year_ints]
        for yr in self.years:
            yr_int = int(yr)
            df_year = df_melt[df_melt["date"].dt.year == yr_int]
            df_geo = df_year[["iso_a3", "value", "normalized_value"]]
            merged = self.world.merge(df_geo, left_on="ADM0_A3",
                                     right_on="iso_a3", how="left")
            
            table = df_year.sort_values("value", ascending=False)
            self.precomputed[yr] = {
                "geojson": merged.to_json(),
                "table": table
            }

    def _update_year_slider(self) -> None:
        """Adjust slider range based on available years."""
        if not self.years:
            return
        start, end = int(self.years[0]), int(self.years[-1])
        self.slider.start, self.slider.end = start, end
        if start <= int(self.current_year) <= end:
            self.slider.value = int(self.current_year)
        else:
            self.slider.value = start

    def _update_data_for_year(self, year: str) -> None:
        """Refresh map and individual plots for the given year."""
        info = self.precomputed.get(year)
        if not info:
            self._set_map_to_grey()
            self._clear_individual_dashboard()
            return
        if not self.geosource:
            self.geosource = GeoJSONDataSource(geojson=info["geojson"])
            self.geo_plot.patches(
                "xs", "ys", source=self.geosource,
                fill_color={"field": "normalized_value", "transform": self.color_mapper},
                line_color="black", line_width=0.5
            )
        else:
            self.geosource.geojson = info["geojson"]
        self._update_individual_dashboard(info["table"])

    def _set_map_to_grey(self) -> None:
        """Clear map coloring."""
        empty = self.world.copy()
        for col in ["value", "log_value", "normalized_value"]:
            empty[col] = np.nan
        geojson = empty.to_json()
        if not self.geosource:
            self.geosource = GeoJSONDataSource(geojson=geojson)
            self.geo_plot.patches(
                "xs", "ys", source=self.geosource,
                fill_color={"field": "normalized_value", "transform": self.color_mapper},
                line_color="black", line_width=0.5
            )
        else:
            self.geosource.geojson = geojson

    def _build_geo_figure(self) -> figure:
        """Create the base map figure."""
        p = figure(toolbar_location=None, tools="hover",
                   width=950, height=430, match_aspect=True)
        hover = p.select_one(HoverTool)
        hover.tooltips = [("Country", "@NAME"), ("Value", "@value{0,0.00}")]
        p.axis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.background_fill_color = None
        p.outline_line_color = None
        p.y_range = Range1d(-95, 90)
        return p

    def _build_individual_components(self) -> figure:
        """Create the trend plot for top countries."""
        p = figure(title="Trend for Top 10 Countries",
                   toolbar_location="right", width=950, height=430,
                   tools="pan,wheel_zoom,box_zoom,reset,save")
        p.xaxis.major_label_orientation = 45
        p.xgrid.grid_line_color = "#dddddd"
        p.ygrid.grid_line_color = "#dddddd"
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "Value"
        return p

    def _get_country_color(self, country: str, idx: int) -> str:
        """Assign or reuse a color for each country."""
        if country in self.country_colors:
            return self.country_colors[country]
        palette = Category20[20] if idx < 20 else None
        color = palette[idx] if palette else f"#{random.randint(50,230):02x}{random.randint(50,230):02x}{random.randint(50,230):02x}"
        self.country_colors[country] = color
        return color

    def _update_individual_dashboard(self, table_data: pd.DataFrame) -> None:
        """Plot lines and scatter for top countries over years."""
        self.individual_plot.renderers = []
        top = table_data.head(10)["country"].tolist()
        all_data: Dict[str, Dict[str, Any]] = {}
        for i, country in enumerate(top):
            years, vals = [], []
            for yr in self.years:
                df = self.precomputed[yr]["table"]
                row = df[df["country"] == country]
                if not row.empty:
                    years.append(yr)
                    vals.append(float(row["value"].iloc[0]))
            if len(years) > 1:
                color = self._get_country_color(country, i)
                self.individual_plot.line(x=years, y=vals, line_width=1.5, color=color, legend_label=country)
                all_data[country] = {"years": years, "values": vals, "color": color}

        plot_data = {"x":[], "y":[], "color":[], "country":[], "year":[], "value":[]}
        for country, d in all_data.items():
            for yr, val in zip(d["years"], d["values"]):
                plot_data["x"].append(yr)
                plot_data["y"].append(val)
                plot_data["color"].append(d["color"])
                plot_data["country"].append(country)
                plot_data["year"].append(yr)
                plot_data["value"].append(val)

        src = ColumnDataSource(data=plot_data)
        scatter = self.individual_plot.scatter(
            x="x", y="y", size=8, fill_color="color",
            line_color="black", line_width=1, source=src, alpha=0.8
        )
        hover = HoverTool(tooltips=[("Country","@country"),("Year","@year"),("Value","@value{0,0.00}")],
                           renderers=[scatter])
        self.individual_plot.tools = [t for t in self.individual_plot.tools if not isinstance(t, HoverTool)]
        self.individual_plot.add_tools(hover)
        self.individual_plot.title.text = f"Trend for Top 10 Countries ({self.current_type})"
        self.individual_plot.legend.click_policy = "hide"
        self.individual_plot.legend.location = "top_right"

    def _clear_individual_dashboard(self) -> None:
        """Remove existing renderers."""
        self.individual_plot.renderers = []
        self.individual_plot.title.text = "Trend for Top 10 Countries"

    def _build_color_bar(self) -> ColorBar:
        """Create a horizontal color bar."""
        bar = ColorBar(color_mapper=self.color_mapper,
                       location=(0,10), orientation='horizontal',
                       width=600, height=20, ticker=FixedTicker(ticks=[]),
                       major_label_text_font_style="normal",
                       title_text_font_size="12pt")
        return bar

    def _get_iso3(self, name: str) -> Optional[str]:
        """Convert country name to ISO3 code."""
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            return None

# Initialize and run
config = WorldTimeConfig(
    shapefile_dir=Path("data/shapefiles"),
    year="2023",
    title="World Data Dashboard"
)
dashboard = MultiDashboard(config)
dashboard.run()


# python3 -m bokeh serve src/dashboard/main.py --show