from pathlib import Path
import sys
import os

# Add parent directories to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent
sys.path.insert(0, str(grandparent_dir))
sys.path.insert(0, str(parent_dir))

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Tabs, TabPanel, Div, Select, Slider, CheckboxGroup

# Now import from the correct paths
try:
    # Try absolute imports first
    from src.dashboard.config import WorldTimeConfig
    from src.dashboard.utils import format_option_name, make_blue_red_transparent_palette
    from src.dashboard.data_manager import DataManager
    from src.dashboard.geo import GeoDash
    from src.dashboard.series import SeriesDash
    from src.dashboard.forecast import ForecastDash
    from src.logger import setup_logger, setup_logging
except ImportError:
    # Fall back to relative imports
    from config import WorldTimeConfig
    from utils import format_option_name, make_blue_red_transparent_palette
    from data_manager import DataManager
    from geo import GeoDash
    from series import SeriesDash
    from forecast import ForecastDash
    # For logger, we need to go up one level
    sys.path.insert(0, str(parent_dir))
    from logger import setup_logger, setup_logging

class MultiDashboard:
    def __init__(self, cfg: WorldTimeConfig):
        setup_logging()
        self.logger = setup_logger(cfg.log_file)
        self.cfg = cfg
        self.current_indicator = ""
        self.dm = DataManager(cfg)
        palette = make_blue_red_transparent_palette(
            blue_hex="#324376",
            red_hex="#6E0D25",
            steps=10
        )
        self.geo_panel = GeoDash(self.dm.world, palette=palette)
        self.series_panel = SeriesDash(self.dm.years)
        self.forecast_panel = ForecastDash(self.dm.years)
        self.title_div = Div(
            width=1500, height=50,
            styles={"font-size":"24pt","font-weight":"bold","text-align":"center"}
        )
        self._make_sidebar()
        self._populate_sectors()

    def _make_sidebar(self):
        self.sector_select = Select(
            options=[("", "Select Sector")], 
            width=230,
            name="sector_select"
        )
        
        self.type_select = Select(
            options=[("", "Select Indicator")], 
            width=230,
            name="type_select"
        )
        
        self.slider = Slider(
            start=2020, 
            end=2023,
            value=int(self.cfg.year),
            step=1, 
            width=230,
            name="year_slider"
        )
        
        self.country_select = Select(
            options=[("", "Select Country")],
            value="",
            width=230
        )
        
        self.g20_checkbox = CheckboxGroup(
            labels=["Compare with G-20"], 
            active=[0],
            width=230
        )

        self.sector_select.on_change("value", self._on_sector)
        self.type_select.on_change("value", self._on_type)
        self.slider.on_change("value", self._on_year)
        self.country_select.on_change('value', self._on_country_change)
        self.g20_checkbox.on_change('active', self._on_g20_filter_change)

        self.sidebar = column(
            Div(text="<b>Sector</b>"),      self.sector_select,
            Div(text="<b>Indicator</b>"),   self.type_select,
            Div(text="<b>Year</b>"),        self.slider,
            Div(text="<b>Country</b>"),     self.country_select,
            self.g20_checkbox,
            width=250, background="#f5f5f5"
        )
    
    def _on_country_change(self, attr, old, new):
        if not self.current_indicator:
            return
            
        if not new and 0 not in self.g20_checkbox.active:
            self.series_panel.selected_country = None
            self.series_panel._update_visible_countries()
            return
            
        if new:
            self.series_panel.add_country(new)
    
    def _on_g20_filter_change(self, attr, old, new):
        if not self.current_indicator:
            return
            
        show_g20 = 0 in self.g20_checkbox.active
        
        self._update_country_options(False)
        
        self.series_panel.set_g20_only(show_g20)
    
    def _update_country_options(self, show_only_g20=True):
        countries = self.dm.get_filtered_countries(show_only_g20)
        
        options = [("", "Select Country")]
        options.extend([(c, c.title() if isinstance(c, str) else c) for c in countries])
        self.country_select.options = options
        
        if self.country_select.value not in [opt[0] for opt in self.country_select.options]:
            self.country_select.value = ""

    def _populate_sectors(self):
        opts = [("", "Select Sector")] + [
            (s, format_option_name(s))
            for s in sorted(set(self.dm.category_dict.values()))
        ]
        self.sector_select.options = opts

    def run(self):
        self.logger.info("Starting dashboard")
        
        # Get the forecast panel layout directly
        forecast_layout = self.forecast_panel.get_layout()
        
        tabs = Tabs(tabs=[
            TabPanel(child=column(self.geo_panel.fig), title="Geospatial"),
            TabPanel(child=column(self.series_panel.fig), title="Individual"),
            TabPanel(child=forecast_layout, title="Forecast")
        ])
        
        # Clean overall layout
        layout = column(
            row(
                self.sidebar, 
                column(
                    self.title_div, 
                    tabs,
                    width=1450
                )
            )
        )
        curdoc().add_root(layout)
        curdoc().title = self.cfg.title

    def _on_sector(self, attr, old, new):
        self.series_panel.clear()
        self.geo_panel.clear()
        self.forecast_panel.clear()
        self.title_div.text = ""
        self.current_indicator = ""
        self.country_select.options = [("", "Select Country")]
        self.country_select.value = ""

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
            
        self.current_indicator = new
        self.dm.fetch_and_prepare(new)

        yrs_int = [int(y) for y in self.dm.years]
        self.slider.start = min(yrs_int)
        self.slider.end = max(yrs_int)
        default = int(self.cfg.year)
        self.slider.value = default if default in yrs_int else min(yrs_int)

        all_vals = {
            yr: {
                "values": self.dm.precomputed[yr]["values"],
                "raw_values": self.dm.precomputed[yr]["raw_values"],
                "display_names": self.dm.precomputed[yr]["display_names"]
            }
            for yr in self.dm.years
        }
        self.geo_panel.load_indicator(self.dm.years, all_vals)

        initial = str(self.slider.value)
        self.series_panel.update(
            data_manager=self.dm,
            selected_table=self.dm.precomputed[initial]["table"],
            all_precomputed=self.dm.precomputed,
            years=self.dm.years,
            current_type=format_option_name(new).upper(),
            indicator=new,
            filter_year=initial
        )
        
        # Update forecast panel
        self.forecast_panel.update(self.dm, new)
        
        self._update_country_options(False)
        
        self.country_select.value = ""

        self.geo_panel.set_year(initial)
        self.title_div.text = f"{self._format_indicator_title(new)} ({initial})"

    def _format_indicator_title(self, indicator):
        """Format indicator name for display"""
        # Convert underscores to spaces and capitalize properly
        formatted = indicator.replace('_', ' ')
        # Split by spaces and capitalize each word
        words = formatted.split()
        capitalized_words = []
        for word in words:
            if len(word) > 2:  # Capitalize longer words
                capitalized_words.append(word.capitalize())
            else:  # Keep short words (like 'of', 'in') lowercase unless first word
                if len(capitalized_words) == 0:
                    capitalized_words.append(word.capitalize())
                else:
                    capitalized_words.append(word.lower())
        return ' '.join(capitalized_words)

    def _on_year(self, attr, old, new):
        if not self.current_indicator:
            return
            
        yr = str(new)
        label = self._format_indicator_title(self.current_indicator)
        self.title_div.text = f"{label} ({yr})"
        self.geo_panel.set_year(yr)
        self.series_panel.update_filter_year(yr)


cfg = WorldTimeConfig(
    shapefile_dir=Path("data/shapefiles"),
    year="2023",
    title="World Data Dashboard"
)
app = MultiDashboard(cfg)
app.run()


# python3 -m bokeh serve src/dashboard --show --port 5007