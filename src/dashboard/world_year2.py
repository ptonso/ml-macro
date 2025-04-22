import pandas as pd
import geopandas as gpd
from pathlib import Path
import pycountry
import numpy as np
import zipfile
import requests
import tempfile

from bokeh.io import curdoc
from bokeh.models import (
    GeoJSONDataSource, LinearColorMapper, ColorBar, Slider,
    Panel, Tabs, Div, HoverTool, Range1d, FixedTicker, Select,
    Title
)
from bokeh.plotting import figure
from bokeh.layouts import column, row, layout, Spacer

from src.logger import setup_logger, setup_logging
from src.utils import get_data_dir

class WorldTimeConfig:
    def __init__(self, shapefile_dir, year, title="Generic Map", log_file="api.log"):
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

# ---------------------------------------------------------
# FUNÇÕES PARA FORMATAÇÃO DOS NOMES DOS SETORES E TIPOS
# ---------------------------------------------------------
def format_option_name(name):
    """
    Formata o nome para exibição nas opções:
    1. Substitui underscores por espaços
    2. Capitaliza as palavras
    3. Remove '_world_bank' do final
    """
    # Remove '_world_bank' do final se existir
    if name.endswith("_world_bank"):
        name = name[:-11]
    
    # Substitui underscores por espaços e capitaliza
    formatted = name.replace("_", " ").replace("-", " ")
    formatted = formatted.title()
    
    return formatted

# ---------------------------------------------------------
# FUNÇÕES PARA GERAÇÃO DA PALETA (com transparência ajustada)
# ---------------------------------------------------------
def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def make_blue_red_transparent_palette(
    blue_hex = "#324376",
    red_hex = "#6E0D25",
    steps = 8
):
    half = steps // 2
    br, bg, bb = _hex_to_rgb(blue_hex)
    rr, rg, rb = _hex_to_rgb(red_hex)

    palette = []
    # azul
    for i in range(half):
        alpha = 1.0 - (i / (half - 1)) * 0.75 if (half - 1) else 1.0
        palette.append(f"rgba({br},{bg},{bb},{alpha:.2f})")

    # vermelho
    for i in range(half):
        alpha = 0.25 + (i / (half - 1)) * 0.75 if (half - 1) else 0.25
        palette.append(f"rgba({rr},{rg},{rb},{alpha:.2f})")

    return palette

# ---------------------------------------------------------

class MultiCsvDashboard:
    def __init__(self, config):
        self.config = config
        setup_logging()
        self.logger = setup_logger(self.config.log_file)

        # DataFrame atual (carregado de acordo com "Setor" e "Type")
        self.df = None
        self.world = None
        self.geosource = None
        
        # Título
        self.title_div = None

        # Armazenará dados precomputados por ano (como no seu código original)
        self.precomputed = {}

        self.years = []
        self.current_year = self.config.year
        
        # Rastrear o tipo selecionado atual para atualizar o título
        self.current_type = ""

        # Paleta
        self.palette = make_blue_red_transparent_palette(
            blue_hex="#324376",
            red_hex="#6E0D25",
            steps=8
        )
        
        # ColorMapper com cor cinza para NaN
        self.color_mapper = LinearColorMapper(
            palette=self.palette,
            low=0,
            high=1,
            nan_color="lightgray"  # Isto garante que, sem dados (NaN), fique tudo cinza
        )

        self.plot = None
        self.slider = None

        # Widgets para setor e type
        self.sector_select = None
        self.type_select = None

        self.sidebar_width = 200
        self.main_content_width = 1000
        self.color_bar = None

    def run(self):
        self.logger.info("Starting Multi-CSV dashboard...")

        # Carrega shapefile apenas uma vez
        self._load_shapefile()

        # Cria um título personalizado como um Div em vez de usar o título do plot
        self.title_div = Div(
            text="",
            width=self.main_content_width,
            height=50,
            styles={"font-size": "24pt", "font-weight": "bold", "text-align": "center"}
        )

        # Constrói a figura (mapa) e a barra lateral (inclui setor, tipo, slider)
        self.plot = self._build_figure()
        sidebar = self._build_sidebar()
        
        # Criamos a barra de cores depois do plot para ter acesso às dimensões corretas
        self.color_bar = self._build_color_bar()
        
        # Adiciona a barra de cores como uma anotação no mapa
        self.plot.add_layout(self.color_bar, 'below')
        
        # Layout com o título personalizado acima do mapa
        main_content = column(self.title_div, self.plot)
        
        # Layout principal
        main_row = row(sidebar, main_content)
        dashboard_layout = column(main_row)

        # Inicializa o GeoJSONDataSource com geometria + valores vazios (NaN)
        # para aparecer tudo cinza antes de qualquer seleção:
        self._set_map_to_grey()

        curdoc().add_root(dashboard_layout)
        curdoc().title = self.config.title

    # ---------------------------------------------------------
    # Carrega shapefile (uma vez) e simplifica a geometria
    # ---------------------------------------------------------
    def _load_shapefile(self):
        shapefile_path = self._get_or_download_shapefile()
        self.world = gpd.read_file(shapefile_path)
        self.world["geometry"] = self.world["geometry"].simplify(tolerance=0.01)

    # ---------------------------------------------------------
    # Monta a sidebar: título, selects de setor/tipo, slider de ano
    # ---------------------------------------------------------
    def _build_sidebar(self):
        sidebar_title = Div(
            text="""<h2 style="text-align: center; margin-bottom: 20px;">FILTER</h2>""",
            width=self.sidebar_width
        )

        # Títulos em formato DIV para os selects
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

        # Prepara as opções formatadas para o setor
        sector_options = [("", "")]  # Opção vazia
        for sector in SECTOR_CSV_MAP.keys():
            formatted_name = format_option_name(sector)
            # Sem usar HTML para evitar problemas de renderização
            sector_options.append((sector, formatted_name))
        
        # Select para 'Sector' (sem HTML no título)
        self.sector_select = Select(
            title="",  # Título vazio, usamos o DIV acima
            value="",  # começa sem nada selecionado
            options=sector_options,
            width=self.sidebar_width - 20
        )
        self.sector_select.on_change("value", self._on_sector_change)

        # Select para 'Type' (sem HTML no título)
        self.type_select = Select(
            title="",  # Título vazio, usamos o DIV acima
            value="",
            options=[("", "")],  # vazio no começo
            width=self.sidebar_width - 20
        )
        self.type_select.on_change("value", self._on_type_change)

        # Slider de ano (sem HTML no título)
        self.slider = Slider(
            start=2020,
            end=2023,
            value=int(self.current_year),
            step=1,
            title="",  # Título vazio, usamos o DIV acima
            width=self.sidebar_width - 20
        )
        self.slider.on_change("value", self._on_year_change)

        sidebar = column(
            sidebar_title,
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

    # ---------------------------------------------------------
    # Callbacks dos widgets
    # ---------------------------------------------------------
    def _on_sector_change(self, attr, old, new):
        """Quando o usuário seleciona um setor, atualizamos as opções de 'Type'."""
        sector = new
        if sector == "":
            # Se limpou a seleção, esvazia o Type
            self.type_select.options = [("", "")]
            self.type_select.value = ""
            self._set_map_to_grey()
            # Reseta o título para vazio
            self.update_title("")
            self.current_type = ""
        else:
            # Carrega as opções de CSV para esse setor com formatação
            csv_list = SECTOR_CSV_MAP[sector]
            type_options = [("", "")]  # Opção vazia
            for csv in csv_list:
                formatted_name = format_option_name(csv)
                # Sem usar HTML para evitar problemas de renderização
                type_options.append((csv, formatted_name))
            
            self.type_select.options = type_options
            self.type_select.value = ""
            # Mantenha o mapa cinza até escolher o Type
            self._set_map_to_grey()

    def _on_type_change(self, attr, old, new):
        """Quando o usuário seleciona um 'Type', carrega o CSV correspondente e atualiza."""
        csv_name = new
        
        # FASE 1: Coloca o mapa em cinza imediatamente
        self._set_map_to_grey()
        self.update_title("")
        
        if csv_name == "":
            # Se limpou a seleção, não precisamos fazer mais nada
            self.current_type = ""
            return

        # Guarda o tipo atual formatado para atualizar o título
        self.current_type = format_option_name(csv_name)
        
        # FASE 2: Agenda o carregamento dos dados para o próximo ciclo de eventos
        # Isso permite que a interface atualize o mapa para cinza antes de começar o carregamento
        def load_data():
            # Print para debug
            print(f"Carregando dados para: {csv_name}")
            
            # Reinicializa completamente os dados
            self.precomputed = {}
            self.years = []
            
            # Carrega e processa o CSV
            self._load_csv_data(csv_name)
            
            if self.df is None:
                # Se não conseguiu carregar o CSV, mantém em cinza
                self._set_map_to_grey()
                self.update_title("")
                self.current_type = ""
                return
                
            self._prepare_data()  # Faz o melt e precompute
            self._update_year_slider()  # Ajusta o slider de acordo com as datas encontradas

            # Redesenha para o ano atual (ou o year do slider)
            self.current_year = str(self.slider.value)
            self._update_data_for_year(self.current_year)
            
            # Atualiza o título com o tipo escolhido
            title_text = f"{self.current_type.upper()} ({self.current_year})"
            # Print para debug
            print(f"Definindo título: {title_text}")
            self.update_title(title_text)
        
        # Registrar o callback para execução no próximo ciclo
        curdoc().add_next_tick_callback(load_data)

    def _on_year_change(self, attr, old, new):
        self.current_year = str(new)
        
        # Se não tiver tipo selecionado, não faz nada
        if not self.current_type:
            return
            
        self._update_data_for_year(self.current_year)
        
        # Atualiza o título mantendo o tipo atual
        if self.current_type:
            title_text = f"{self.current_type.upper()} ({self.current_year})"
            # Print para debug
            print(f"Definindo título após mudança de ano: {title_text}")
            self.update_title(title_text)
        else:
            self.update_title("")

    def update_title(self, title_text):
        """Atualiza o título usando o Div personalizado."""
        self.title_div.text = title_text
        print(f"Título definido para: '{title_text}'")

    # ---------------------------------------------------------
    # Carrega CSV específico e guarda em self.df
    # ---------------------------------------------------------
    def _load_csv_data(self, csv_name):
        """Carrega o CSV do disco (conforme estrutura do seu projeto)."""
        self.logger.info(f"Loading CSV: {csv_name}")
        data_dir = get_data_dir() / "00--raw"  # ajuste se necessário
        # Você pode ter subpastas. Ex: macro, demography etc.
        # Se quiser, pode mapear 'csv_name' -> Path direto.
        # Exemplo (supondo que ele está espalhado em subpastas):
        #   Tente localizar em qualquer subpasta
        possible_paths = list(data_dir.rglob(f"{csv_name}.csv"))
        if not possible_paths:
            self.logger.error(f"CSV not found for: {csv_name}")
            self.df = None
            return
        csv_path = possible_paths[0]  # pega o primeiro encontrado
        self.df = pd.read_csv(csv_path)

    # ---------------------------------------------------------
    # Faz o pré-processamento do CSV para cada ano (similar ao original)
    # ---------------------------------------------------------
    def _prepare_data(self):
        """Precompute merged GeoJSON data for each year."""
        if self.df is None:
            return
        self.logger.info("Precomputing merged data by year...")
        self.precomputed = {}

        # Exemplo: assumimos que a df tem a coluna 'date' e as colunas de países
        # no mesmo formato do seu código original
        df = self.df.copy()
        df_melt = df.melt(
            id_vars="date",
            var_name="country",
            value_name="value"  # agora chamamos de 'value' para ficar genérico
        )
        df_melt["iso_a3"] = df_melt["country"].apply(self._get_iso3)

        # Lista de anos
        years = sorted(df["date"].str[:4].unique())
        self.years = years

        # Prepara GeoJSON para cada ano
        for year in years:
            df_year = df_melt[df_melt["date"].str.startswith(year)]
            merged = self.world.merge(
                df_year,
                left_on="ADM0_A3",
                right_on="iso_a3",
                how="left"
            )
            val_min = df_year["value"].min()
            val_max = df_year["value"].max()
            # Se não tiver dado, evita erro definindo 0 e 1
            if pd.isna(val_min) or pd.isna(val_max):
                val_min, val_max = 0, 1
            self.precomputed[year] = (merged.to_json(), val_min, val_max)

    # ---------------------------------------------------------
    # Atualiza o range do Slider de acordo com os anos do CSV
    # ---------------------------------------------------------
    def _update_year_slider(self):
        if not self.years:
            # Se não tiver anos, define algo genérico
            self.slider.start = 2020
            self.slider.end = 2023
            self.slider.value = 2021
        else:
            self.slider.start = int(self.years[0])
            self.slider.end = int(self.years[-1])
            # Se a config.year estiver dentro do range, vamos usar
            if self.slider.start <= int(self.config.year) <= self.slider.end:
                self.slider.value = int(self.config.year)
            else:
                # caso contrário, usa o primeiro ano
                self.slider.value = self.slider.start

    # ---------------------------------------------------------
    # Atualiza o GeoJSONSource com os dados do ano escolhido
    # ---------------------------------------------------------
    def _update_data_for_year(self, year):
        self.logger.info(f"Updating data for year: {year}")
        if year in self.precomputed:
            geojson, val_min, val_max = self.precomputed[year]
            self.geosource.geojson = geojson
            self.color_mapper.low = val_min
            self.color_mapper.high = val_max
        else:
            # Se não achar (p.ex. não existe esse ano no CSV), deixar cinza
            self._set_map_to_grey()

    # ---------------------------------------------------------
    # Força o mapa todo a ficar cinza (usado quando não há Type/Setor)
    # ---------------------------------------------------------
    def _set_map_to_grey(self):
        # Basta preencher a coluna "value" com NaN
        empty_json = self.world.copy()
        empty_json["value"] = np.nan
        
        # Se ainda não existe o geosource, cria e conecta aos patches
        if self.geosource is None:
            self.geosource = GeoJSONDataSource(geojson=empty_json.to_json())
            
            # Vincula o geojson ao patches do mapa
            self.plot.patches(
                "xs", "ys",
                source=self.geosource,
                fill_color={"field": "value", "transform": self.color_mapper},
                line_color="black",
                line_width=0.5,
                fill_alpha=1
            )
        else:
            # Atualiza o geosource existente
            self.geosource.geojson = empty_json.to_json()
            
        # Ajusta limites do color_mapper para não dar erro
        self.color_mapper.low = 0
        self.color_mapper.high = 1

    # ---------------------------------------------------------
    # Constrói a figura do mapa (sem patches; patches serão adicionados depois)
    # ---------------------------------------------------------
    def _build_figure(self):
        p = figure(
            title="",  # Começamos sem título no plot
            toolbar_location=None,
            tools="hover",
            width=self.main_content_width,
            height=480,
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

    # ---------------------------------------------------------
    # Cria a barra de cores
    # ---------------------------------------------------------
    def _build_color_bar(self):
        # Sem ticks
        custom_ticks = []

        # Calcula o ponto central do gráfico no eixo X
        # A largura do conteúdo principal (mapa) é self.main_content_width
        # A largura da barra de cores é 600, então precisamos calcular o deslocamento para centralizá-la
        center_x = self.main_content_width / 2
        bar_width = 600
        x_offset = center_x - (bar_width / 2)

        color_bar = ColorBar(
            color_mapper=self.color_mapper,
            location=(x_offset, 10),  # Define a posição X para centralizar e Y para ficar na parte inferior
            orientation='horizontal',
            width=bar_width,
            height=20,
            padding=5,
            label_standoff=12,
            title="",  # Removido o título "Value"
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

    # ---------------------------------------------------------
    # Converte nome de país em ISO3
    # ---------------------------------------------------------
    def _get_iso3(self, name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            return None

    # ---------------------------------------------------------
    # Baixa o shapefile do Natural Earth (caso não exista)
    # ---------------------------------------------------------
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

# -------------------------
# Exemplo de uso:
# -------------------------
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
/* Forçar título da barra de cores em negrito */
.bk-ColorBar-title {
    font-weight: bold !important;
    font-style: normal !important;
    color: black !important;
}
/* Estilo para os selects */
.bk-input {
    font-size: 14px !important;
    font-weight: bold !important;
}
/* Customização dos selects */
select.bk-input option {
    font-weight: bold !important;
}
</style>
"""

curdoc().template_variables["custom_css"] = custom_css

config = WorldTimeConfig(
    shapefile_dir=Path("data/shapefiles"),
    year="2023",
    title="Multi CSV Dashboard"
)

dashboard = MultiCsvDashboard(config)
dashboard.run()

# python3 -m bokeh serve src/dashboard/world_year2.py --show
