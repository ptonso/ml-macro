import pandas as pd
import geopandas as gpd
from pathlib import Path
import pycountry
import numpy as np
import zipfile
import requests
import tempfile
import random
from collections import defaultdict

from bokeh.io import curdoc
from bokeh.models import (
    GeoJSONDataSource, LinearColorMapper, ColorBar, Slider,
    Tabs, Div, HoverTool, Range1d, FixedTicker, Select,
    Title, RadioButtonGroup, ColumnDataSource, DataTable, 
    TableColumn, NumberFormatter, StringFormatter, TabPanel,
    MultiLine, Legend
)
from bokeh.plotting import figure
from bokeh.layouts import column, row, layout, Spacer
from bokeh.palettes import Category10, Category20

from src.logger import setup_logger
from src.utils import get_data_dir

class WorldTimeConfig:
    def __init__(self, shapefile_dir, year, title="Dashboard", log_file="api.log"):
        self.shapefile_dir = shapefile_dir
        self.year = year
        self.title = title
        self.log_file = log_file

SECTOR_CSV_MAP = {
    "demography": [
        "economic_activity_world_bank",
        "gdp_per_person_employed_constant_2011_ppp_usd_world_bank",
        "gini_income_inequality_world_bank",
        "life_expectancy_at_birth_total_years_world_bank",
        "population_size_world_bank",
        "poverty_headcount_ratio_at_1.90_a_day_2011_ppp_percent_of_population_world_bank",
        "total_population_world_bank",
    ],
    "education": [
        "education_expenditures_world_bank",
        "education_years_world_bank",
    ],
    "energy": [
        "coal_energy_production_world_bank",
        "energy_use_kg_of_oil_equivalent_per_capita_world_bank",
        "gas_energy_production_world_bank",
        "hydro_electric_energy_production_world_bank",
        "petroleum_energy_production_world_bank",
    ],
    "geographic": [
        "area_world_bank",
    ],
    "global-economic-linkages": [
        "net_official_development_assistance_received_current_usd_world_bank",
    ],
    "infrastructure-and-technology": [
        "individuals_using_the_internet_percent_of_population_world_bank",
        "research_and_development_expenditure_percent_of_gdp_world_bank",
    ],
    "macroeconomic": [
        "consumer_price_index_change_world_bank",
        "gdp_current_usd_world_bank",
        "unemployment_rate_percent_of_total_labor_force_world_bank",
    ],
    "political-and-policy-environment": [
        "ease_of_doing_business_rank_world_bank",
        "political_stability_and_absence_of_violence_terrorism_percentile_rank_world_bank",
    ],
    "sectoral-performance": [
        "manufacturing_value_added_percent_of_gdp_world_bank",
        "services_value_added_percent_of_gdp_world_bank",
    ],
    "trade-and-commerce": [
        "fdi_net_inflows_current_usd_world_bank",
        "net_trade_in_goods_and_services_current_usd_world_bank",
    ]
}

def format_option_name(name):
    if name.endswith("_world_bank"):
        name = name[:-11]
    
    formatted = name.replace("_", " ").replace("-", " ")
    formatted = formatted.title()
    
    return formatted

def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def make_blue_red_transparent_palette(
    blue_hex = "#324376",
    red_hex = "#6E0D25",
    steps = 10
):
    half = steps // 2
    br, bg, bb = _hex_to_rgb(blue_hex)
    rr, rg, rb = _hex_to_rgb(red_hex)

    palette = []
    for i in range(half):
        alpha = 1.0 - (i / (half - 1)) * 0.8 if (half - 1) else 1.0
        palette.append(f"rgba({br},{bg},{bb},{alpha:.2f})")

    for i in range(half):
        alpha = 0.2 + (i / (half - 1)) * 0.8 if (half - 1) else 0.2
        palette.append(f"rgba({rr},{rg},{rb},{alpha:.2f})")

    return palette

def safe_log_transform(value):
    if pd.isna(value):
        return np.nan
    elif value > 0:
        return np.log1p(value)
    elif value < 0:
        return -np.log1p(abs(value))
    else:
        return 0.0

class MultiDashboard:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(self.config.log_file)

        self.df = None
        self.world = None
        self.geosource = None
        
        self.title_div = None

        self.precomputed = {}

        self.years = []
        self.current_year = self.config.year
        
        self.current_type = ""

        self.palette = make_blue_red_transparent_palette(
            blue_hex="#324376",
            red_hex="#6E0D25",
            steps=10
        )
        
        self.color_mapper = LinearColorMapper(
            palette=self.palette,
            low=0,
            high=1,
            nan_color="lightgray"
        )

        self.geo_plot = None
        self.individual_plot = None
        
        self.slider = None
        self.sector_select = None
        self.type_select = None
        self.dashboard_select = None

        self.sidebar_width = 250
        self.main_content_width = 1200
        self.color_bar = None
        
        self.current_view = 0
        
        self.country_colors = {}

    def run(self):
        self.logger.info("Starting Multi Dashboard...")

        self._load_shapefile()

        self.title_div = Div(
            text="",
            width=self.main_content_width,
            height=50,
            styles={"font-size": "24pt", "font-weight": "bold", "text-align": "center"}
        )

        self.geo_plot = self._build_geo_figure()
        self.individual_plot = self._build_individual_components()
        sidebar = self._build_sidebar()
        
        self.color_bar = self._build_color_bar()
        self.geo_plot.add_layout(self.color_bar, 'below')
        
        geo_view = column(self.geo_plot)
        individual_view = column(self.individual_plot)
        
        self.dashboard_tabs = Tabs(
            tabs=[
                TabPanel(child=geo_view, title="Geospatial Dashboard"),
                TabPanel(child=individual_view, title="Individual Dashboard")
            ]
        )
        
        main_content = column(self.title_div, self.dashboard_tabs)
        main_row = row(sidebar, main_content)
        dashboard_layout = column(main_row)

        self._set_map_to_grey()

        curdoc().add_root(dashboard_layout)
        curdoc().title = self.config.title

    def _load_shapefile(self):
        shapefile_path = self._get_or_download_shapefile()
        self.world = gpd.read_file(shapefile_path)
        self.world["geometry"] = self.world["geometry"].simplify(tolerance=0.01)

    def _build_sidebar(self):
        sector_title = Div(
            text="""<div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">Sector</div>""",
            width=self.sidebar_width - 20
        )
        
        type_title = Div(
            text="""<div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">Type</div>""",
            width=self.sidebar_width - 20
        )
        
        year_title = Div(
            text="""<div style="font-size: 18px; font-weight: bold; margin-bottom: 5px; margin-top: 5px;">Year</div>""",
            width=self.sidebar_width - 20
        )

        sector_options = [("", "")]
        for sector in SECTOR_CSV_MAP.keys():
            formatted_name = format_option_name(sector)
            sector_options.append((sector, formatted_name))
        
        self.sector_select = Select(
            title="",
            value="",
            options=sector_options,
            width=self.sidebar_width - 20
        )
        self.sector_select.on_change("value", self._on_sector_change)

        self.type_select = Select(
            title="",
            value="",
            options=[("", "")],
            width=self.sidebar_width - 20
        )
        self.type_select.on_change("value", self._on_type_change)

        self.slider = Slider(
            start=2020,
            end=2023,
            value=int(self.current_year),
            step=1,
            title="",
            width=self.sidebar_width - 20
        )
        self.slider.on_change("value", self._on_year_change)

        sidebar = column(
            sector_title,
            self.sector_select,
            type_title,
            self.type_select,
            year_title,
            self.slider,
            width=self.sidebar_width,
            background="#f5f5f5",
            css_classes=["sidebar"]
        )
        return sidebar

    def _on_sector_change(self, attr, old, new):
        sector = new
        if sector == "":
            self.type_select.options = [("", "")]
            self.type_select.value = ""
            self._set_map_to_grey()
            self._clear_individual_dashboard()
            self.update_title("")
            self.current_type = ""
        else:
            csv_list = SECTOR_CSV_MAP[sector]
            type_options = [("", "")]
            for csv in csv_list:
                formatted_name = format_option_name(csv)
                type_options.append((csv, formatted_name))
            
            self.type_select.options = type_options
            self.type_select.value = ""
            self._set_map_to_grey()
            self._clear_individual_dashboard()

    def _on_type_change(self, attr, old, new):
        csv_name = new
        
        current_csv_name = csv_name
        
        self._set_map_to_grey()
        self._clear_individual_dashboard()
        self.update_title("")
        
        if current_csv_name == "":
            self.current_type = ""
            self.country_colors = {}
            return

        self.current_type = format_option_name(current_csv_name)
        
        self.country_colors = {}
        
        def load_data():
            print(f"Loading data for: {current_csv_name}")
            
            self.precomputed = {}
            self.years = []
            
            self._load_csv_data(current_csv_name)
            
            if self.df is None:
                self._set_map_to_grey()
                self._clear_individual_dashboard()
                self.update_title("")
                self.current_type = ""
                return
                
            self._prepare_data()
            self._update_year_slider()

            self.current_year = str(self.slider.value)
            self._update_data_for_year(self.current_year)
            
            title_text = f"{self.current_type.upper()} ({self.current_year})"
            print(f"Setting title: {title_text}")
            self.update_title(title_text)
        
        curdoc().add_next_tick_callback(load_data)

    def _on_year_change(self, attr, old, new):
        self.current_year = str(new)
        
        if not self.current_type:
            return
            
        self._update_data_for_year(self.current_year)
        
        if self.current_type:
            title_text = f"{self.current_type.upper()} ({self.current_year})"
            print(f"Setting title after year change: {title_text}")
            self.update_title(title_text)
        else:
            self.update_title("")

    def update_title(self, title_text):
        self.title_div.text = title_text
        print(f"Title set to: '{title_text}'")

    def _load_csv_data(self, csv_name):
        self.logger.info(f"Loading CSV: {csv_name}")
        data_dir = get_data_dir() / "00--raw"
        possible_paths = list(data_dir.rglob(f"{csv_name}.csv"))
        if not possible_paths:
            self.logger.error(f"CSV not found for: {csv_name}")
            self.df = None
            return
        csv_path = possible_paths[0]
        self.df = pd.read_csv(csv_path)

    def _prepare_data(self):
        if self.df is None:
            return
        self.logger.info("Precomputing merged data by year...")
        self.precomputed = {}

        df = self.df.copy()
        df_melt = df.melt(
            id_vars="date",
            var_name="country",
            value_name="value"
        )
        df_melt["iso_a3"] = df_melt["country"].apply(self._get_iso3)
        
        years = sorted(df["date"].str[:4].unique())
        self.years = years
        
        print("Applying log transformation and calculating deciles...")
        df_melt["log_value"] = df_melt["value"].apply(safe_log_transform)
        
        non_nan_values = df_melt["log_value"].dropna()
        
        if len(non_nan_values) > 0:
            deciles = np.nanpercentile(non_nan_values, np.arange(0, 101, 10))
            print(f"Decile values: {deciles}")
            
            def assign_decile(x):
                if pd.isna(x):
                    return np.nan
                decile_index = np.searchsorted(deciles, x, side='right') - 1
                return min(max(decile_index, 0), 9) / 9.0
            
            df_melt["normalized_value"] = df_melt["log_value"].apply(assign_decile)
        else:
            print("No valid values in the data, using 0.5 as default")
            df_melt["normalized_value"] = 0.5
        
        for year in years:
            print(f"Processing year: {year}")
            df_year = df_melt[df_melt["date"].str.startswith(year)]
            
            merged = self.world.merge(
                df_year,
                left_on="ADM0_A3",
                right_on="iso_a3",
                how="left"
            )
            
            table_data = df_year.sort_values(by="value", ascending=False)
            
            val_min = df_year["value"].min()
            val_max = df_year["value"].max()
            if pd.isna(val_min) or pd.isna(val_max):
                val_min, val_max = 0, 1
                
            self.precomputed[year] = (merged.to_json(), val_min, val_max, table_data)
            print(f"Data processed for {year}. Min: {val_min}, Max: {val_max}")

    def _update_year_slider(self):
        if not self.years:
            self.slider.start = 2020
            self.slider.end = 2023
            self.slider.value = 2021
        else:
            self.slider.start = int(self.years[0])
            self.slider.end = int(self.years[-1])
            if self.slider.start <= int(self.config.year) <= self.slider.end:
                self.slider.value = int(self.config.year)
            else:
                self.slider.value = self.slider.start

    def _update_data_for_year(self, year):
        self.logger.info(f"Updating data for year: {year}")
        if year in self.precomputed:
            geojson, val_min, val_max, table_data = self.precomputed[year]
            
            self.geosource.geojson = geojson
            self.color_mapper.low = 0
            self.color_mapper.high = 1
            
            self._update_individual_dashboard(table_data)
        else:
            self._set_map_to_grey()
            self._clear_individual_dashboard()

    def _set_map_to_grey(self):
        empty_json = self.world.copy()
        empty_json["value"] = np.nan
        empty_json["log_value"] = np.nan
        empty_json["normalized_value"] = np.nan
        
        if self.geosource is None:
            self.geosource = GeoJSONDataSource(geojson=empty_json.to_json())
            
            self.geo_plot.patches(
                "xs", "ys",
                source=self.geosource,
                fill_color={"field": "normalized_value", "transform": self.color_mapper},
                line_color="black",
                line_width=0.5,
                fill_alpha=1
            )
        else:
            self.geosource.geojson = empty_json.to_json()
            
        self.color_mapper.low = 0
        self.color_mapper.high = 1

    def _build_geo_figure(self):
        p = figure(
            title="",
            toolbar_location=None,
            tools="hover",
            width=self.main_content_width,
            height=700,
            match_aspect=True
        )
        hover = p.select_one(HoverTool)
        hover.tooltips = [
            ("Country", "@NAME"),
            ("Value", "@value{0,0.00}")
        ]
        hover.point_policy = "follow_mouse"

        p.axis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.outline_line_color = None
        p.background_fill_color = None
        p.border_fill_color = None
        p.y_range = Range1d(-95, 90)

        return p

    def _build_individual_components(self):
        ind_plot = figure(
            title="Trend for Top 10 Countries",
            toolbar_location="right",
            width=self.main_content_width,
            height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        ind_plot.xaxis.major_label_orientation = 45
        ind_plot.xgrid.grid_line_color = "#dddddd"
        ind_plot.ygrid.grid_line_color = "#dddddd"
        ind_plot.xaxis.axis_label = "Year"
        ind_plot.yaxis.axis_label = "Value"
        
        return ind_plot

    def _get_country_color(self, country, index=None):
        if country in self.country_colors:
            return self.country_colors[country]
        
        if index is not None and index < 20:
            palette = Category20[20]
            color = palette[index % 20]
        else:
            r = random.randint(50, 230)
            g = random.randint(50, 230)
            b = random.randint(50, 230)
            color = f"#{r:02x}{g:02x}{b:02x}"
        
        self.country_colors[country] = color
        return color

    def _update_individual_dashboard(self, current_year_data):
        self.individual_plot.renderers = []
        
        top_countries_df = current_year_data.head(10)
        top_countries = top_countries_df['country'].tolist()
        
        all_countries_data = {}
        
        for year in self.years:
            _, _, _, year_data = self.precomputed[year]
            for i, country in enumerate(top_countries):
                if country not in all_countries_data:
                    all_countries_data[country] = {'years': [], 'values': [], 'color': self._get_country_color(country, i)}
                
                country_data = year_data[year_data['country'] == country]
                if not country_data.empty:
                    all_countries_data[country]['years'].append(year)
                    all_countries_data[country]['values'].append(float(country_data['value'].iloc[0]))
        
        for country, data in all_countries_data.items():
            if len(data['years']) > 1:
                self.individual_plot.line(
                    x=data['years'],
                    y=data['values'],
                    line_width=1.5,
                    color=data['color'],
                    legend_label=country
                )
        
        plot_data = {
            'x': [],
            'y': [],
            'color': [],
            'country': [],
            'year': [],
            'value': []
        }
        
        for country, data in all_countries_data.items():
            for i, year in enumerate(data['years']):
                plot_data['x'].append(year)
                plot_data['y'].append(data['values'][i])
                plot_data['color'].append(data['color'])
                plot_data['country'].append(country)
                plot_data['year'].append(year)
                plot_data['value'].append(data['values'][i])
        
        source = ColumnDataSource(data=plot_data)
        
        scatter = self.individual_plot.scatter(
            x='x',
            y='y',
            size=8,
            fill_color='color',
            line_color='black',
            line_width=1,
            source=source,
            alpha=0.8
        )
        
        hover = HoverTool(
            tooltips=[
                ("Country", "@country"),
                ("Year", "@year"),
                ("Value", "@value{0,0.00}")
            ],
            renderers=[scatter]
        )
        
        self.individual_plot.tools = [tool for tool in self.individual_plot.tools if not isinstance(tool, HoverTool)]
        self.individual_plot.add_tools(hover)
        
        self.individual_plot.title.text = f"Trend for Top 10 Countries ({self.current_type})"
        self.individual_plot.xaxis.axis_label = "Year"
        self.individual_plot.yaxis.axis_label = self.current_type
        
        self.individual_plot.legend.click_policy = "hide"
        self.individual_plot.legend.location = "top_right"
        self.individual_plot.legend.title = "Countries"

    def _clear_individual_dashboard(self):
        self.individual_plot.renderers = []
        self.individual_plot.title.text = "Trend for Top 10 Countries"

    def _build_color_bar(self):
        custom_ticks = []

        center_x = self.main_content_width / 2
        bar_width = 600
        x_offset = center_x - (bar_width / 2)

        color_bar = ColorBar(
            color_mapper=self.color_mapper,
            location=(x_offset, 10),
            orientation='horizontal',
            width=bar_width,
            height=20,
            padding=5,
            label_standoff=12,
            title="",
            title_standoff=10,
            ticker=FixedTicker(ticks=custom_ticks),
            major_tick_line_color="black",
            bar_line_color="black",
            border_line_color=None,
            major_label_text_color="black",
            major_label_text_font_style="normal",
            title_text_font_style="bold",
            title_text_color="black",
            title_text_font_size="12pt",
        )
        
        return color_bar

    def _get_iso3(self, name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            return None

    def _get_or_download_shapefile(self):
        shapefile_dir = self.config.shapefile_dir / "ne_110m_admin_0_countries"
        shapefile_path = shapefile_dir / "ne_110m_admin_0_countries.shp"

        if shapefile_path.exists():
            self.logger.info(f"Using cached shapefile: {shapefile_path}")
            return shapefile_path

        self.logger.info("Downloading Natural Earth shapefile...")
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download shapefile: HTTP {response.status_code}")

        shapefile_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_zip_path = Path(tmp_file.name)

        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(shapefile_dir)
        tmp_zip_path.unlink()
        self.logger.info(f"Shapefile extracted to: {shapefile_dir}")
        return shapefile_path

custom_css = """
<style>
.sidebar {
    padding: 15px;
    border-right: 1px solid #ddd;
}
.color-bar-container {
    margin-top: 10px;
    text-align: center;
}
.bk-ColorBar-title {
    font-weight: bold !important;
    font-style: normal !important;
    color: black !important;
}
.bk-input {
    font-size: 14px !important;
    font-weight: bold !important;
}
select.bk-input option {
    font-weight: bold !important;
}
</style>
"""

curdoc().template_variables["custom_css"] = custom_css

config = WorldTimeConfig(
    shapefile_dir=Path("data/shapefiles"),
    year="2023",
    title="World Data Dashboard"
)

dashboard = MultiDashboard(config)
dashboard.run()