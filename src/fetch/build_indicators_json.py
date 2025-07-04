import os
import json
from src.utils import get_metadata_dir, get_data_dir
from src.logger import setup_logger

def main():

    logger = setup_logger()
    data_dir = get_data_dir()
    metadata_path = get_metadata_dir()
    os.makedirs(metadata_path, exist_ok=True)

    indicator_json = metadata_path / "indicators.json"

    with open(indicator_json, 'w') as file:
        json.dump(indicators, file, indent=4)
        logger.success(f"indicators.json saved to: data/{indicator_json.parent.relative_to(data_dir)}")


indicators = {
    "macroeconomic": {
        "gdp_current_usd": {
            "title": "GDP Current USD",
            "description": "Measures total economic output.",
            "world_bank": "NY.GDP.MKTP.CD"
        },
        "unemployment_rate_percent_of_total_labor_force": {
            "title": "Unemployment Rate Percent of Total Labor Force",
            "description": "Percentage of the labor force that is unemployed.",
            "world_bank": "SL.UEM.TOTL.ZS"
        },
        "consumer_price_index_change": {
            "title": "Consumer Price Index Change",
            "description": "Measures inflation by tracking changes in the price level of a basket of goods.",
            "world_bank": "FP.CPI.TOTL"
        },
    },
    "trade-and-commerce": {
        "net_trade_in_goods_and_services_current_usd": {
            "title": "Net Trade in Goods and Services Current USD",
            "description": "Balance of exports and imports.",
            "world_bank": "NE.RSB.GNFS.CD"
        },
        "fdi_net_inflows_current_usd": {
            "title": "FDI Net Inflows Current USD",
            "description": "Investment into a country from foreign entities.",
            "world_bank": "BX.KLT.DINV.CD.WD"
        }
    },
    "financial-markets": {
        "major_stock_index": {
            "title": "Major Stock Index",
            "description": "Key indicator of stock market performance.",
            "alpha_vantage": "STOCK_INDEX_PLACEHOLDER"
        },
        "exchange_rate_local_currency_unit_per_usd_period_average": {
            "title": "Exchange Rate Local Currency Unit Per USD Period Average",
            "description": "Exchange rate of local currency to USD.",
            "open_exchange_rates": "PA.NUS.FCRF"
        }
    },
    "demography": {
        "total_population": {
            "title": "Total Population",
            "description": "Number of people residing in a country.",
            "world_bank": "SP.POP.TOTL"
        },
        "gdp_per_person_employed_constant_2011_ppp_usd": {
            "title": "GDP Per Person Employed Constant 2011 PPP USD",
            "description": "Productivity measure.",
            "world_bank": "SL.GDP.PCAP.EM.KD"
        },
        "life_expectancy_at_birth_total_years": {
            "title": "Life Expectancy at Birth Total Years",
            "description": "Average lifespan at birth.",
            "world_bank": "SP.DYN.LE00.IN"
        },
        "poverty_headcount_ratio_at_1.90_a_day_2011_ppp_percent_of_population": {
            "title": "Poverty Headcount Ratio at 1.90 a Day 2011 PPP Percent of Population",
            "description": "Percentage of population below the poverty line.",
            "world_bank": "SI.POV.DDAY"
        },
        "population_size": {
            "title": "Population Size",
            "description": "Number of residents.",
            "world_bank": "SP.POP.TOTL"
        },
        "economic_activity": {
            "title": "Economic Activity",
            "description": "Labor force participation rate.",
            "world_bank": "SL.TLF.ACTI.ZS"
        },
        "gini_income_inequality": {
            "title": "Gini Income Inequality",
            "description": "Income inequality measure.",
            "world_bank": "SI.POV.GINI"
        },
        "human_development_index": {
            "title": "Human Development Index",
            "description": "Composite index of life expectancy, education, and per capita income.",
            "undp": "HDI_INDEX_PLACEHOLDER"
        },
        "crime_rate": {
            "title": "Crime Rate",
            "description": "Incidents per 100,000 people.",
            "unodc": "VC.IHR.PSRC.P5"
        }
    },
    "sectoral-performance": {
        "manufacturing_value_added_percent_of_gdp": {
            "title": "Manufacturing Value Added Percent of GDP",
            "description": "Contribution of manufacturing to GDP.",
            "world_bank": "NV.IND.MANF.ZS"
        },
        "services_value_added_percent_of_gdp": {
            "title": "Services Value Added Percent of GDP",
            "description": "Contribution of services to GDP.",
            "world_bank": "NV.SRV.TOTL.ZS"
        }
    },
    "energy": {
        "energy_use_kg_of_oil_equivalent_per_capita": {
            "title": "Energy Use KG of Oil Equivalent Per Capita",
            "description": "Per capita energy consumption.",
            "world_bank": "EG.USE.PCAP.KG.OE"
        },
        "renewables_excluding_hydro_share": {
            "title": "Electricity from Renewables Excluding Hydroelectric (% of total)",
            "description": "Electricity from Renewables Excluding Hydroelectric (% of total)",
            "world_bank": "EG.ELC.RNWX.ZS"
        },
        "hydro_electric_energy_production": {
            "title": "Hydro Electric Energy Production",
            "description": "Amount of energy produced by hydro-electric sources.",
            "world_bank": "EG.ELC.HYRO.ZS"
        },
        "petroleum_energy_production": {
            "title": "Petroleum Energy Production",
            "description": "Amount of energy produced by petroleum sources.",
            "world_bank": "EG.ELC.PETR.ZS"
        },
        "gas_energy_production": {
            "title": "Gas Energy Production",
            "description": "Amount of energy produced by gas sources.",
            "world_bank": "EG.ELC.NGAS.ZS"
        },
        "coal_energy_production": {
            "title": "Coal Energy Production",
            "description": "Amount of energy produced by coal sources.",
            "world_bank": "EG.ELC.COAL.ZS"
        }
    },
    "infrastructure-and-technology": {
        "individuals_using_the_internet_percent_of_population": {
            "title": "Individuals Using the Internet Percent of Population",
            "description": "Internet penetration rate.",
            "world_bank": "IT.NET.USER.ZS"
        },
        "research_and_development_expenditure_percent_of_gdp": {
            "title": "Research and Development Expenditure Percent of GDP",
            "description": "Spending on R&D.",
            "world_bank": "GB.XPD.RSDV.GD.ZS"
        }
    },
    "global-economic-linkages": {
        "net_official_development_assistance_received_current_usd": {
            "title": "Net Official Development Assistance Received Current USD",
            "description": "Amount of aid received.",
            "world_bank": "DT.ODA.ODAT.CD"
        }
    },
    "commodities": {
        "potato": {
            "title": "Potato",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.POTA"
        },
        "rice": {
            "title": "Rice",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.RICE"
        },
        "soy": {
            "title": "Soy",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.SOYB"
        },
        "corn": {
            "title": "Corn",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.CORN"
        },
        "wheat": {
            "title": "Wheat",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.WHEA"
        },
        "sugar": {
            "title": "Sugar",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.SUGR"
        },
        "coffee": {
            "title": "Coffee",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.COFF"
        },
        "milk": {
            "title": "Milk",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.MILK"
        },
        "cattle": {
            "title": "Cattle",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.CATL"
        },
        "fish": {
            "title": "Fish",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.FISH"
        },
        "pig": {
            "title": "Pig",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.PIG"
        },
        "poultry_chicken": {
            "title": "Poultry (Chicken)",
            "description": "Agricultural production volume.",
            "fao": "AG.PRD.POLT"
        }
    },
    "geographic": {
        "area": {
            "title": "Area",
            "description": "Total land area.",
            "world_bank": "AG.SRF.TOTL.K2"
        },
    },
    "education": {
        "education_expenditures": {
            "title": "Education Expenditures",
            "description": "Spending on education.",
            "world_bank": "SE.XPD.TOTL.GD.ZS"
        },
        "education_years": {
            "title": "Education Years",
            "description": "Average years of schooling.",
            "world_bank": "SE.PRM.DURS"
        },
        "paper_production": {
            "title": "Paper Production",
            "description": "Academic papers published.",
            "unesco": "IP.PRD.PAPER"
        }
    }
}



if __name__ == "__main__":
    main()
