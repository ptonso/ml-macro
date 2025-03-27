import pandas as pd
import geopandas as gpd
from pathlib import Path
import pycountry
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import zipfile
import requests
import tempfile

from bokeh.io import curdoc
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, Slider
from bokeh.plotting import figure
from bokeh.palettes import Viridis256
from bokeh.layouts import column

from src.logger import setup_logger


@dataclass
class WorldTimeConfig:
    dataset_path: Path
    year: str
    title: str = "GDP per Capita Map"
    log_file: str = "api.log"
    shapefile_dir: Path = Path("data/shapefiles")


class GdpDashboard:
    def __init__(self, config: WorldTimeConfig) -> None:
        self.config: WorldTimeConfig = config
        self.logger = setup_logger(self.config.log_file)
        self.world: Optional[gpd.GeoDataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.geosource: Optional[GeoJSONDataSource] = None
        self.precomputed: Dict[str, Tuple[str, float, float]] = {}
        self.years: List[str] = []
        self.current_year: str = ""
        self.color_mapper: Optional[LinearColorMapper] = None
        self.plot = None
        self.slider = None

    def run(self) -> None:
        """Run the interactive dashboard."""
        self.logger.info(f"Starting dashboard...")
        self._load_data()
        self._prepare_data()

        self.years = sorted(self.precomputed.keys())
        self.current_year = self.config.year

        self.color_mapper = LinearColorMapper(palette=Viridis256)
        self._update_data_for_year(self.current_year)

        self.plot = self._build_figure()

        self.slider = Slider(
            start=int(self.years[0]),
            end=int(self.years[-1]),
            value=int(self.current_year),
            step=1,
            title="Year"
        )
        self.slider.on_change("value", self._on_year_change)

        layout = column(self.slider, self.plot)
        curdoc().add_root(layout)

    def _load_data(self) -> None:
        """Load CSV data and world geometries."""
        self.logger.info(f"Loading data...")
        try:
            self.df = pd.read_csv(self.config.dataset_path)
            shapefile_path: Path = self._get_or_download_shapefile()
            self.world = gpd.read_file(shapefile_path)
            # Simplify geometries to boost performance
            self.world["geometry"] = self.world["geometry"].simplify(tolerance=0.01)
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing error: {e}")
            raise

    def _prepare_data(self) -> None:
        """Precompute merged GeoJSON data for each year."""
        self.logger.info(f"Precomputing merged data for each year...")
        self.precomputed = {}
        df = self.df.copy()
        df_melt = df.melt(
            id_vars="date",
            var_name="country",
            value_name="gdp_per_capita"
        )
        df_melt["iso_a3"] = df_melt["country"].apply(self._get_iso3)
        years = sorted(df["date"].str[:4].unique())
        for year in years:
            df_year = df_melt[df_melt["date"].str.startswith(year)]
            merged = self.world.merge(
                df_year,
                left_on="ADM0_A3",
                right_on="iso_a3",
                how="left"
            )
            gdp_min: float = df_year["gdp_per_capita"].min()
            gdp_max: float = df_year["gdp_per_capita"].max()
            self.precomputed[year] = (merged.to_json(), gdp_min, gdp_max)

        init_data, _, _ = self.precomputed[self.config.year]
        self.geosource = GeoJSONDataSource(geojson=init_data)

    def _update_data_for_year(self, year: str) -> None:
        """Update the GeoJSON data source with precomputed data for the year."""
        self.logger.info(f"Updating data for year: {year}")
        if year in self.precomputed:
            geojson, gdp_min, gdp_max = self.precomputed[year]
            self.geosource.geojson = geojson
            self.color_mapper.low = gdp_min
            self.color_mapper.high = gdp_max
        else:
            self.logger.error(f"No precomputed data for year: {year}")

    def _on_year_change(self, attr: str, old: int, new: int) -> None:
        """Handle slider value changes to update the displayed year."""
        self.current_year = str(new)
        self._update_data_for_year(self.current_year)
        self.plot.title.text = f"{self.config.title} ({self.current_year})"

    def _build_figure(self) -> figure:
        """Build the Bokeh choropleth map."""
        p = figure(
            title=f"{self.config.title} ({self.current_year})",
            toolbar_location=None,
            tools="",
            width=1000,
            height=600,
            match_aspect=True
        )
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        p.patches(
            "xs", "ys",
            source=self.geosource,
            fill_color={"field": "gdp_per_capita", "transform": self.color_mapper},
            line_color="gray",
            line_width=0.25,
            fill_alpha=1
        )

        color_bar = ColorBar(
            color_mapper=self.color_mapper,
            location=(0, 0),
            label_standoff=12
        )
        p.add_layout(color_bar, "right")
        return p

    def _get_iso3(self, name: str) -> Optional[str]:
        """Lookup ISO3 code for the given country name."""
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            return None

    def _get_or_download_shapefile(self) -> Path:
        """Retrieve or download the Natural Earth shapefile."""
        shapefile_dir: Path = self.config.shapefile_dir / "ne_110m_admin_0_countries"
        shapefile_path: Path = shapefile_dir / "ne_110m_admin_0_countries.shp"

        if shapefile_path.exists():
            self.logger.info(f"Using cached shapefile: {shapefile_path}")
            return shapefile_path

        self.logger.info(f"Downloading Natural Earth shapefile...")
        url: str = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download shapefile: HTTP {response.status_code}")

        shapefile_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_zip_path: Path = Path(tmp_file.name)

        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(shapefile_dir)
        tmp_zip_path.unlink()
        self.logger.info(f"Shapefile extracted to: {shapefile_dir}")
        return shapefile_path


from src.utils import get_data_dir

data_dir: Path = get_data_dir() / "00--raw" / "macro"
category: str = "macroeconomic"
csv_name: str = "gdp_current_usd_world_bank.csv"

dataset_path: Path = data_dir / category / csv_name
title: str = "".join(csv_name.split(".")[:-1]).replace("_", " ").capitalize()

config = WorldTimeConfig(
    dataset_path=dataset_path,
    year="2023",
    title=title
)

dashboard = GdpDashboard(config)
dashboard.run()



# python3 -m bokeh serve src/dashboard/world_year.py --show
