import random
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, HoverTool, Span, Legend, LegendItem
)
from bokeh.palettes import Category20
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

class SeriesDash:
    def __init__(self, years: List[str]):
        self.years: List[str] = years
        self.country_colors: Dict[str, str] = {}
        self.all_countries: List[str] = []
        self.all_precomputed = {}
        self.current_type = ""
        self.filter_year = years[0] if years else ""
        self.visible_countries: Set[str] = set()
        self.selected_country: Optional[str] = None
        self.is_data_loaded = False
        self.data_manager = None
        self.current_indicator = ""
        self.g20_only = True
        self.g20_countries_in_data: Set[str] = set()
        self.G20_COUNTRIES = [
            "Argentina", "Australia", "Brazil", "Canada", "China", 
            "France", "Germany", "India", "Indonesia", "Italy", 
            "Japan", "Mexico", "Russia", "Saudi Arabia", "South Africa", 
            "South Korea", "Turkey", "United Kingdom", "United States", "European Union"
        ]
        self.fig = self._build_figure()
        self.source = ColumnDataSource(data=dict(
            x=[], y=[], country=[], color=[], year=[], value=[], alpha=[]
        ))
        self.scatter_renderer = self.fig.scatter(
            x="x", y="y",
            size=8,
            fill_color="color",
            line_color="black",
            line_width=1,
            fill_alpha=0.8,
            line_alpha=1.0,
            source=self.source,
            level="overlay"
        )
        hover = HoverTool(
            tooltips=[
                ("Country", "@country"),
                ("Year", "@year"),
                ("Value", "@value{0,0.00}")
            ],
            renderers=[self.scatter_renderer],
            formatters={"@value": "numeral"}
        )
        self.fig.add_tools(hover)
        self.year_marker = Span(
            location=0,
            dimension='height',
            line_color='red',
            line_width=2,
            line_dash='dashed'
        )
        self.fig.add_layout(self.year_marker)
        self.legend = Legend(
            click_policy="hide",
            location="top_right",
            items=[]
        )
        self.fig.add_layout(self.legend)
        self.line_renderers = {}
        
    def _build_figure(self):
        p = figure(
            title="Country Trend",
            toolbar_location="right",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=1100,
            height=430,
            sizing_mode="fixed"
        )
        p.xaxis.major_label_orientation = 45
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "Value"
        p.xaxis.axis_label_text_font_style = "bold"
        p.yaxis.axis_label_text_font_style = "bold"
        p.xaxis.axis_label_text_color = "black"
        p.yaxis.axis_label_text_color = "black"
        p.title.text_color = "black"
        p.xgrid.grid_line_color = "#dddddd"
        p.ygrid.grid_line_color = "#dddddd"
        p.output_backend = "webgl"
        return p

    def _get_country_color(self, country: str, idx: int) -> str:
        if country in self.country_colors:
            return self.country_colors[country]
        if idx < 20:
            palette = Category20[20]
            color = palette[idx]
        else:
            color = f"#{random.randint(50,230):02x}{random.randint(50,230):02x}{random.randint(50,230):02x}"
        self.country_colors[country] = color
        return color
    
    def add_country(self, country: str):
        if not self.is_data_loaded:
            return
            
        if self.selected_country == country:
            return
            
        self.selected_country = country if country else None
        
        self._update_visible_countries()
    
    def set_g20_only(self, g20_only: bool):
        self.g20_only = g20_only
        
        self._update_visible_countries()
    
    def _update_visible_countries(self):
        old_visible = set(self.visible_countries)
        
        self.visible_countries = set()
        
        if self.g20_only and self.is_data_loaded:
            self.visible_countries.update(self.g20_countries_in_data)
        
        if self.selected_country:
            self.visible_countries.add(self.selected_country)
        
        if old_visible != self.visible_countries:
            self._update_plot()
    
    def _update_plot(self):
        if not self.is_data_loaded:
            return
            
        filter_year_idx = self.years.index(self.filter_year) if self.filter_year in self.years else 0
        filtered_years = self.years[filter_year_idx:]
        if filtered_years:
            self.year_marker.location = filtered_years[0]
        
        current_countries = set(self.visible_countries)
        countries_to_remove = set(self.line_renderers.keys()) - current_countries
        
        for country in countries_to_remove:
            if country in self.line_renderers:
                renderer = self.line_renderers[country]
                if renderer in self.fig.renderers:
                    self.fig.renderers.remove(renderer)
                del self.line_renderers[country]
        
        plot_data = {
            "x": [], "y": [], "color": [],
            "country": [], "year": [], "value": [],
            "alpha": []
        }
        legend_items = []
        
        for idx, country in enumerate(self.visible_countries):
            yrs, vals = self.data_manager.get_country_data(country, self.current_indicator)
            filter_indices = [i for i, yr in enumerate(yrs) if yr in filtered_years]
            yrs = [yrs[i] for i in filter_indices]
            vals = [vals[i] for i in filter_indices]
            
            if len(yrs) > 1:
                color = self._get_country_color(country, idx)
                display_country = country.title() if isinstance(country, str) else country
                
                opacity = 1.0
                if self.selected_country and country != self.selected_country:
                    opacity = 0.15  # Increased transparency (0.3 -> 0.15)
                
                if country in self.line_renderers:
                    line = self.line_renderers[country]
                    line.data_source.data.update({'x': yrs, 'y': vals})
                    line.glyph.line_alpha = opacity
                else:
                    line = self.fig.line(
                        x=yrs, y=vals,
                        line_width=2.5,
                        color=color,
                        line_alpha=opacity,
                        name=country
                    )
                    self.line_renderers[country] = line
                
                legend_items.append(LegendItem(label=display_country, renderers=[line]))
                
                for y, v in zip(yrs, vals):
                    plot_data["x"].append(y)
                    plot_data["y"].append(v)
                    plot_data["color"].append(color)
                    plot_data["country"].append(display_country)
                    plot_data["year"].append(y)
                    plot_data["value"].append(v)
                    plot_data["alpha"].append(opacity)
        
        self.source.data = plot_data
        
        if self.scatter_renderer:
            if plot_data["alpha"]:
                self.scatter_renderer.glyph.fill_alpha = 'alpha'
                self.scatter_renderer.glyph.line_alpha = 'alpha'
            else:
                self.scatter_renderer.glyph.fill_alpha = 0.8
                self.scatter_renderer.glyph.line_alpha = 1.0
        
        self.legend.items = legend_items
        
        if plot_data["y"]:
            min_val = min(v for v in plot_data["y"] if v is not None)
            max_val = max(v for v in plot_data["y"] if v is not None)
            padding = (max_val - min_val) * 0.1 if max_val > min_val else max_val * 0.1
            self.fig.y_range.start = max(0, min_val - padding)
            self.fig.y_range.end = max_val + padding
        
        self.fig.title.text = f"Country Trend ({self.current_type}) - From {self.filter_year} onwards"
        self.fig.title.text_color = "black"

    def update(
        self,
        data_manager,
        selected_table: pd.DataFrame,
        all_precomputed: Dict[str, Dict[str, pd.DataFrame]],
        years: List[str],
        current_type: str,
        indicator: str,
        filter_year: Optional[str] = None
    ):
        old_indicator = self.current_indicator
        
        self.data_manager = data_manager
        self.all_precomputed = all_precomputed
        self.years = years
        self.current_type = current_type
        self.current_indicator = indicator
        
        if filter_year and filter_year in years:
            self.filter_year = filter_year
        
        if old_indicator != indicator:
            for country, renderer in self.line_renderers.items():
                if renderer in self.fig.renderers:
                    self.fig.renderers.remove(renderer)
            
            self.line_renderers = {}
            self.legend.items = []
            self.country_colors = {}
            self.visible_countries = set()
            self.selected_country = None
            
            if self.scatter_renderer:
                self.scatter_renderer.glyph.fill_alpha = 0.8
                self.scatter_renderer.glyph.line_alpha = 1.0
            
            self.all_countries = data_manager.get_filtered_countries(False)
            
            self.g20_countries_in_data = set()
            all_countries_lower = [c.lower() if isinstance(c, str) else c for c in self.all_countries]
            for g20_country in self.G20_COUNTRIES:
                g20_lower = g20_country.lower()
                for i, country_lower in enumerate(all_countries_lower):
                    if country_lower == g20_lower:
                        actual_country = self.all_countries[i]
                        self.g20_countries_in_data.add(actual_country)
        
        self.is_data_loaded = True
        
        self._update_visible_countries()
        
    def update_filter_year(self, year: str):
        if year not in self.years or not self.is_data_loaded:
            return
        self.filter_year = year
        self._update_plot()

    def clear(self):
        for country, renderer in self.line_renderers.items():
            if renderer in self.fig.renderers:
                self.fig.renderers.remove(renderer)
        self.line_renderers = {}
        self.source.data = dict(x=[], y=[], country=[], color=[], year=[], value=[], alpha=[])
        
        if self.scatter_renderer:
            self.scatter_renderer.glyph.fill_alpha = 0.8
            self.scatter_renderer.glyph.line_alpha = 1.0
        
        self.legend.items = []
        self.fig.title.text = "Country Trend"
        self.fig.title.text_color = "black"
        self.visible_countries = set()
        self.selected_country = None
        self.all_countries = []
        self.all_precomputed = {}
        self.is_data_loaded = False