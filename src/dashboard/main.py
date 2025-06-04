# src/dashboard/main.py

from pathlib import Path

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Tabs, TabPanel, Div, Select, Slider

from src.dashboard.config       import WorldTimeConfig
from src.dashboard.utils        import format_option_name, make_blue_red_transparent_palette
from src.dashboard.data_manager import DataManager
from src.dashboard.geo          import GeoDash
from src.dashboard.series       import SeriesDash
from src.logger                 import setup_logger, setup_logging

class MultiDashboard:
    def __init__(self, cfg: WorldTimeConfig):
        setup_logging()
        self.logger = setup_logger(cfg.log_file)
        self.cfg    = cfg

        # ── light data load ───────────────────────────────────────────────
        self.dm = DataManager(cfg)

        # ── panels ────────────────────────────────────────────────────────
        palette = make_blue_red_transparent_palette()
        self.geo_panel    = GeoDash(self.dm.world, palette=palette)
        self.series_panel = SeriesDash(self.dm.years)

        # ── title & controls ──────────────────────────────────────────────
        self.title_div = Div(
            width=950, height=50,
            styles={"font-size":"24pt","font-weight":"bold","text-align":"center"}
        )
        self._make_sidebar()
        self._populate_sectors()

    def _make_sidebar(self):
        self.sector_select = Select(options=[("", "Select Sector")], width=230)
        self.type_select   = Select(options=[("", "Select Indicator")], width=230)
        self.slider        = Slider(start=2020, end=2023,
                                    value=int(self.cfg.year),
                                    step=1, width=230)

        self.sector_select.on_change("value", self._on_sector)
        self.type_select.on_change("value",   self._on_type)
        self.slider.on_change("value",        self._on_year)

        self.sidebar = column(
            Div(text="<b>Sector</b>"),     self.sector_select,
            Div(text="<b>Indicator</b>"), self.type_select,
            Div(text="<b>Year</b>"),      self.slider,
            width=250, background="#f5f5f5"
        )

    def _populate_sectors(self):
        opts = [("", "Select Sector")] + [
            (s, format_option_name(s))
            for s in sorted(set(self.dm.category_dict.values()))
        ]
        self.sector_select.options = opts

    def run(self):
        self.logger.info("Starting dashboard")
        tabs = Tabs(tabs=[
            TabPanel(child=column(self.geo_panel.fig),    title="Geospatial"),
            TabPanel(child=column(self.series_panel.fig), title="Individual"),
        ])
        layout = column(row(self.sidebar, column(self.title_div, tabs)))
        curdoc().add_root(layout)
        curdoc().title = self.cfg.title

    def _on_sector(self, attr, old, new):
        # clear both panels
        self.series_panel.clear()
        self.geo_panel.clear()
        self.title_div.text = ""

        if not new:
            self.type_select.options = [("", "Select Indicator")]
            return

        inds = [ind for ind, sec in self.dm.category_dict.items() if sec == new]
        opts = [("", "Select Indicator")] + [
            (i, format_option_name(i)) for i in sorted(inds)
        ]
        self.type_select.options = opts

    def _on_type(self, attr, old, new):
        if not new:
            return self._on_sector(attr, old, "")

        # 1) heavy preprocess of the **entire timeline** for this indicator
        self.dm.fetch_and_prepare(new)

        # 2) reset the year slider bounds & pick a start value
        yrs_int = [int(y) for y in self.dm.years]
        self.slider.start = min(yrs_int)
        self.slider.end   = max(yrs_int)
        default = int(self.cfg.year)
        self.slider.value = default if default in yrs_int else min(yrs_int)

        # 3) build the map's ColumnDataSource ONCE for **all** years
        all_vals = {
            yr: self.dm.precomputed[yr]["values"]
            for yr in self.dm.years
        }
        self.geo_panel.load_indicator(self.dm.years, all_vals)

        # 4) do **one** series redraw for the initial year
        initial = str(self.slider.value)
        self.series_panel.update(
            selected_table  = self.dm.precomputed[initial]["table"],
            all_precomputed = self.dm.precomputed,
            years           = self.dm.years,
            current_type    = format_option_name(new).upper()
        )

        # 5) patch the map to that initial year
        self.geo_panel.set_year(initial)

    def _on_year(self, attr, old, new):
        """Only patch the map—**no** series redraw here."""
        yr = str(new)
        label = format_option_name(self.type_select.value).upper()
        self.title_div.text = f"{label} ({yr})"
        self.geo_panel.set_year(yr)


# entrypoint
cfg = WorldTimeConfig(
    shapefile_dir=Path("data/shapefiles"),
    year="2023",
    title="World Data Dashboard"
)
app = MultiDashboard(cfg)
app.run()
print("run with: python3 -m bokeh serve src/dashboard --show --port 5007")



# python3 -m bokeh serve src/dashboard --show --port 5007
