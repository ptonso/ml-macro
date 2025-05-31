import os
import json

from src.utils import get_metadata_dir, get_data_dir
from src.logger import setup_logger

def main():

    logger = setup_logger()
    data_dir = get_data_dir()
    metadata_path = get_metadata_dir()

    os.makedirs(metadata_path, exist_ok=True)

    countries_json = metadata_path / "countries.json"

    with open(countries_json, 'w') as file:
        json.dump(countries, file, indent=4)
        logger.success(f"countries.json saved to: data/{countries_json.parent.relative_to(data_dir)}")


countries = {
    "ivory_coast": {
        "title": "C\u00f4te d'Ivoire",
        "aliases": [
            "cote_d_ivoire",
            "cote_divoire",
            "c\u00f4te_d'ivoire",
            "ivory_coast"
        ],
        "categorization": []
    },
    "united_kingdom": {
        "title": "United Kingdom",
        "aliases": [
            "england",
            "great_britain",
            "uk",
            "united_kingdom"
        ],
        "categorization": []
    },
    "russia": {
        "title": "Russia",
        "aliases": [
            "russia",
            "russian_federation"
        ],
        "categorization": []
    },
    "slovakia": {
        "title": "Slovakia",
        "aliases": [
            "slovak_republic",
            "slovakia"
        ],
        "categorization": []
    },
    "syria": {
        "title": "Syria",
        "aliases": [
            "syria",
            "syrian_arab_republic"
        ],
        "categorization": []
    },
    "vietnam": {
        "title": "Vietnam",
        "aliases": [
            "viet_nam",
            "vietnam"
        ],
        "categorization": []
    },
    "united_states": {
        "title": "United States",
        "aliases": [
            "united_states",
            "united_states_of_america",
            "us",
            "usa"
        ],
        "categorization": []
    },
    "regions": {
        "title": "Regions and Aggregates",
        "aliases": [
            "africa_eastern_and_southern",
            "africa_western_and_central",
            "arab_world",
            "caribbean_small_states",
            "central_europe_and_the_baltics",
            "early_demographic_dividend",
            "east_asia_&_pacific",
            "east_asia_&_pacific_(excluding_high_income)",
            "east_asia_&_pacific_(ida_&_ibrd_countries)",
            "euro_area",
            "europe_&_central_asia",
            "europe_&_central_asia_(excluding_high_income)",
            "europe_&_central_asia_(ida_&_ibrd_countries)",
            "european_union",
            "fragile_and_conflict_affected_situations",
            "heavily_indebted_poor_countries_(hipc)",
            "high_income",
            "ibrd_only",
            "ida_&_ibrd_total",
            "ida_blend",
            "ida_only",
            "ida_total",
            "late_demographic_dividend",
            "latin_america_&_caribbean",
            "latin_america_&_caribbean_(excluding_high_income)",
            "latin_america_&_the_caribbean_(ida_&_ibrd_countries)",
            "least_developed_countries:_un_classification",
            "low_&_middle_income",
            "low_income",
            "lower_middle_income",
            "middle_east_&_north_africa",
            "middle_east_&_north_africa_(excluding_high_income)",
            "middle_east_&_north_africa_(ida_&_ibrd_countries)",
            "middle_income",
            "north_america",
            "oecd_members",
            "other_small_states",
            "pacific_island_small_states",
            "post_demographic_dividend",
            "pre_demographic_dividend",
            "small_states",
            "south_asia",
            "south_asia_(ida_&_ibrd)",
            "sub_saharan_africa",
            "sub_saharan_africa_(excluding_high_income)",
            "sub_saharan_africa_(ida_&_ibrd_countries)",
            "world"
        ],
        "categorization": []
    },
    "ecuador": {
        "title": "Ecuador",
        "aliases": [
            "ecuador"
        ],
        "categorization": []
    },
    "egypt": {
        "title": "Egypt",
        "aliases": [
            "egypt",
            "egypt_arab_rep"
        ],
        "categorization": []
    },
    "el_salvador": {
        "title": "El Salvador",
        "aliases": [
            "el_salvador"
        ],
        "categorization": []
    },
    "equatorial_guinea": {
        "title": "Equatorial Guinea",
        "aliases": [
            "equatorial_guinea"
        ],
        "categorization": []
    },
    "eritrea": {
        "title": "Eritrea",
        "aliases": [
            "eritrea"
        ],
        "categorization": []
    },
    "estonia": {
        "title": "Estonia",
        "aliases": [
            "estonia"
        ],
        "categorization": []
    },
    "eswatini": {
        "title": "Eswatini",
        "aliases": [
            "eswatini",
            "swaziland"
        ],
        "categorization": []
    },
    "ethiopia": {
        "title": "Ethiopia",
        "aliases": [
            "ethiopia"
        ],
        "categorization": []
    },
    "faroe_islands": {
        "title": "Faroe Islands",
        "aliases": [
            "faroe_islands"
        ],
        "categorization": []
    },
    "fiji": {
        "title": "Fiji",
        "aliases": [
            "fiji"
        ],
        "categorization": []
    },
    "finland": {
        "title": "Finland",
        "aliases": [
            "finland"
        ],
        "categorization": []
    },
    "france": {
        "title": "France",
        "aliases": [
            "france"
        ],
        "categorization": []
    },
    "french_polynesia": {
        "title": "French Polynesia",
        "aliases": [
            "french_polynesia"
        ],
        "categorization": []
    },
    "gabon": {
        "title": "Gabon",
        "aliases": [
            "gabon"
        ],
        "categorization": []
    },
    "gambia": {
        "title": "Gambia",
        "aliases": [
            "gambia",
            "gambia_the",
            "the_gambia"
        ],
        "categorization": []
    },
    "georgia": {
        "title": "Georgia",
        "aliases": [
            "georgia"
        ],
        "categorization": []
    },
    "germany": {
        "title": "Germany",
        "aliases": [
            "germany"
        ],
        "categorization": []
    },
    "ghana": {
        "title": "Ghana",
        "aliases": [
            "ghana"
        ],
        "categorization": []
    },
    "gibraltar": {
        "title": "Gibraltar",
        "aliases": [
            "gibraltar"
        ],
        "categorization": []
    },
    "greece": {
        "title": "Greece",
        "aliases": [
            "greece"
        ],
        "categorization": []
    },
    "greenland": {
        "title": "Greenland",
        "aliases": [
            "greenland"
        ],
        "categorization": []
    },
    "grenada": {
        "title": "Grenada",
        "aliases": [
            "grenada"
        ],
        "categorization": []
    },
    "guam": {
        "title": "Guam",
        "aliases": [
            "guam"
        ],
        "categorization": []
    },
    "guatemala": {
        "title": "Guatemala",
        "aliases": [
            "guatemala"
        ],
        "categorization": []
    },
    "guinea": {
        "title": "Guinea",
        "aliases": [
            "guinea"
        ],
        "categorization": []
    },
    "guinea_bissau": {
        "title": "Guinea-Bissau",
        "aliases": [
            "guinea-bissau",
            "guinea_bissau"
        ],
        "categorization": []
    },
    "guyana": {
        "title": "Guyana",
        "aliases": [
            "guyana"
        ],
        "categorization": []
    },
    "haiti": {
        "title": "Haiti",
        "aliases": [
            "haiti"
        ],
        "categorization": []
    },
    "honduras": {
        "title": "Honduras",
        "aliases": [
            "honduras"
        ],
        "categorization": []
    },
    "hong_kong": {
        "title": "Hong Kong",
        "aliases": [
            "hong_kong",
            "hong_kong_sar",
            "hong_kong_sar_china"
        ],
        "categorization": []
    },
    "hungary": {
        "title": "Hungary",
        "aliases": [
            "hungary"
        ],
        "categorization": []
    },
    "iceland": {
        "title": "Iceland",
        "aliases": [
            "iceland"
        ],
        "categorization": []
    },
    "india": {
        "title": "India",
        "aliases": [
            "india"
        ],
        "categorization": []
    },
    "indonesia": {
        "title": "Indonesia",
        "aliases": [
            "indonesia"
        ],
        "categorization": []
    },
    "iran": {
        "title": "Iran",
        "aliases": [
            "iran",
            "iran_islamic_rep",
            "islamic_republic_of_iran"
        ],
        "categorization": []
    },
    "iraq": {
        "title": "Iraq",
        "aliases": [
            "iraq"
        ],
        "categorization": []
    },
    "ireland": {
        "title": "Ireland",
        "aliases": [
            "ireland"
        ],
        "categorization": []
    },
    "isle_of_man": {
        "title": "Isle of Man",
        "aliases": [
            "isle_of_man"
        ],
        "categorization": []
    },
    "israel": {
        "title": "Israel",
        "aliases": [
            "israel"
        ],
        "categorization": []
    },
    "italy": {
        "title": "Italy",
        "aliases": [
            "italy"
        ],
        "categorization": []
    },
    "jamaica": {
        "title": "Jamaica",
        "aliases": [
            "jamaica"
        ],
        "categorization": []
    },
    "japan": {
        "title": "Japan",
        "aliases": [
            "japan"
        ],
        "categorization": []
    },
    "jordan": {
        "title": "Jordan",
        "aliases": [
            "jordan"
        ],
        "categorization": []
    },
    "kazakhstan": {
        "title": "Kazakhstan",
        "aliases": [
            "kazakhstan"
        ],
        "categorization": []
    },
    "kenya": {
        "title": "Kenya",
        "aliases": [
            "kenya"
        ],
        "categorization": []
    },
    "kiribati": {
        "title": "Kiribati",
        "aliases": [
            "kiribati"
        ],
        "categorization": []
    },
    "north_korea": {
        "title": "North Korea",
        "aliases": [
            "korea_dem._people's_rep.",
            "korea_dem_peoples_rep",
            "north_korea"
        ],
        "categorization": []
    },
    "south_korea": {
        "title": "South Korea",
        "aliases": [
            "korea_rep",
            "south_korea"
        ],
        "categorization": []
    },
    "kosovo": {
        "title": "Kosovo",
        "aliases": [
            "kosovo"
        ],
        "categorization": []
    },
    "kuwait": {
        "title": "Kuwait",
        "aliases": [
            "kuwait"
        ],
        "categorization": []
    },
    "kyrgyzstan": {
        "title": "Kyrgyzstan",
        "aliases": [
            "kyrgyz_republic",
            "kyrgyzstan"
        ],
        "categorization": []
    },
    "laos": {
        "title": "Laos",
        "aliases": [
            "lao_pdr",
            "laos"
        ],
        "categorization": []
    },
    "latvia": {
        "title": "Latvia",
        "aliases": [
            "latvia"
        ],
        "categorization": []
    },
    "lebanon": {
        "title": "Lebanon",
        "aliases": [
            "lebanon"
        ],
        "categorization": []
    },
    "lesotho": {
        "title": "Lesotho",
        "aliases": [
            "lesotho"
        ],
        "categorization": []
    },
    "liberia": {
        "title": "Liberia",
        "aliases": [
            "liberia"
        ],
        "categorization": []
    },
    "libya": {
        "title": "Libya",
        "aliases": [
            "libya"
        ],
        "categorization": []
    },
    "liechtenstein": {
        "title": "Liechtenstein",
        "aliases": [
            "liechtenstein"
        ],
        "categorization": []
    },
    "lithuania": {
        "title": "Lithuania",
        "aliases": [
            "lithuania"
        ],
        "categorization": []
    },
    "luxembourg": {
        "title": "Luxembourg",
        "aliases": [
            "luxembourg"
        ],
        "categorization": []
    },
    "macao": {
        "title": "Macao",
        "aliases": [
            "macao",
            "macao_sar",
            "macao_sar_china"
        ],
        "categorization": []
    },
    "madagascar": {
        "title": "Madagascar",
        "aliases": [
            "madagascar"
        ],
        "categorization": []
    },
    "malawi": {
        "title": "Malawi",
        "aliases": [
            "malawi"
        ],
        "categorization": []
    },
    "malaysia": {
        "title": "Malaysia",
        "aliases": [
            "malaysia"
        ],
        "categorization": []
    },
    "maldives": {
        "title": "Maldives",
        "aliases": [
            "maldives"
        ],
        "categorization": []
    },
    "mali": {
        "title": "Mali",
        "aliases": [
            "mali"
        ],
        "categorization": []
    },
    "malta": {
        "title": "Malta",
        "aliases": [
            "malta"
        ],
        "categorization": []
    },
    "marshall_islands": {
        "title": "Marshall Islands",
        "aliases": [
            "marshall_islands"
        ],
        "categorization": []
    },
    "mauritania": {
        "title": "Mauritania",
        "aliases": [
            "mauritania"
        ],
        "categorization": []
    },
    "mauritius": {
        "title": "Mauritius",
        "aliases": [
            "mauritius"
        ],
        "categorization": []
    },
    "mexico": {
        "title": "Mexico",
        "aliases": [
            "mexico"
        ],
        "categorization": []
    },
    "micronesia": {
        "title": "Micronesia",
        "aliases": [
            "federated_states_of_micronesia",
            "micronesia",
            "micronesia_fed_sts"
        ],
        "categorization": []
    },
    "moldova": {
        "title": "Moldova",
        "aliases": [
            "moldova"
        ],
        "categorization": []
    },
    "monaco": {
        "title": "Monaco",
        "aliases": [
            "monaco"
        ],
        "categorization": []
    },
    "mongolia": {
        "title": "Mongolia",
        "aliases": [
            "mongolia"
        ],
        "categorization": []
    },
    "montenegro": {
        "title": "Montenegro",
        "aliases": [
            "montenegro"
        ],
        "categorization": []
    },
    "morocco": {
        "title": "Morocco",
        "aliases": [
            "morocco"
        ],
        "categorization": []
    },
    "mozambique": {
        "title": "Mozambique",
        "aliases": [
            "mozambique"
        ],
        "categorization": []
    },
    "myanmar": {
        "title": "Myanmar",
        "aliases": [
            "burma",
            "myanmar"
        ],
        "categorization": []
    },
    "namibia": {
        "title": "Namibia",
        "aliases": [
            "namibia"
        ],
        "categorization": []
    },
    "nauru": {
        "title": "Nauru",
        "aliases": [
            "nauru"
        ],
        "categorization": []
    },
    "nepal": {
        "title": "Nepal",
        "aliases": [
            "nepal"
        ],
        "categorization": []
    },
    "netherlands": {
        "title": "Netherlands",
        "aliases": [
            "netherlands"
        ],
        "categorization": []
    },
    "new_caledonia": {
        "title": "New Caledonia",
        "aliases": [
            "new_caledonia"
        ],
        "categorization": []
    },
    "new_zealand": {
        "title": "New Zealand",
        "aliases": [
            "new_zealand"
        ],
        "categorization": []
    },
    "nicaragua": {
        "title": "Nicaragua",
        "aliases": [
            "nicaragua"
        ],
        "categorization": []
    },
    "niger": {
        "title": "Niger",
        "aliases": [
            "niger"
        ],
        "categorization": []
    },
    "nigeria": {
        "title": "Nigeria",
        "aliases": [
            "nigeria"
        ],
        "categorization": []
    },
    "north_macedonia": {
        "title": "North Macedonia",
        "aliases": [
            "north_macedonia"
        ],
        "categorization": []
    },
    "northern_mariana_islands": {
        "title": "Northern Mariana Islands",
        "aliases": [
            "northern_mariana_islands"
        ],
        "categorization": []
    },
    "norway": {
        "title": "Norway",
        "aliases": [
            "norway"
        ],
        "categorization": []
    },
    "oman": {
        "title": "Oman",
        "aliases": [
            "oman"
        ],
        "categorization": []
    },
    "pakistan": {
        "title": "Pakistan",
        "aliases": [
            "pakistan"
        ],
        "categorization": []
    },
    "palau": {
        "title": "Palau",
        "aliases": [
            "palau"
        ],
        "categorization": []
    },
    "panama": {
        "title": "Panama",
        "aliases": [
            "panama"
        ],
        "categorization": []
    },
    "papua_new_guinea": {
        "title": "Papua New Guinea",
        "aliases": [
            "papua_new_guinea"
        ],
        "categorization": []
    },
    "paraguay": {
        "title": "Paraguay",
        "aliases": [
            "paraguay"
        ],
        "categorization": []
    },
    "peru": {
        "title": "Peru",
        "aliases": [
            "peru"
        ],
        "categorization": []
    },
    "philippines": {
        "title": "Philippines",
        "aliases": [
            "philippines"
        ],
        "categorization": []
    },
    "poland": {
        "title": "Poland",
        "aliases": [
            "poland"
        ],
        "categorization": []
    },
    "portugal": {
        "title": "Portugal",
        "aliases": [
            "portugal"
        ],
        "categorization": []
    },
    "puerto_rico": {
        "title": "Puerto Rico",
        "aliases": [
            "puerto_rico"
        ],
        "categorization": []
    },
    "qatar": {
        "title": "Qatar",
        "aliases": [
            "qatar"
        ],
        "categorization": []
    },
    "romania": {
        "title": "Romania",
        "aliases": [
            "romania"
        ],
        "categorization": []
    },
    "rwanda": {
        "title": "Rwanda",
        "aliases": [
            "rwanda"
        ],
        "categorization": []
    },
    "samoa": {
        "title": "Samoa",
        "aliases": [
            "samoa"
        ],
        "categorization": []
    },
    "san_marino": {
        "title": "San Marino",
        "aliases": [
            "san_marino"
        ],
        "categorization": []
    },
    "sao_tome_and_principe": {
        "title": "Sao Tome and Principe",
        "aliases": [
            "sao_tome_and_principe"
        ],
        "categorization": []
    },
    "saudi_arabia": {
        "title": "Saudi Arabia",
        "aliases": [
            "saudi_arabia"
        ],
        "categorization": []
    },
    "senegal": {
        "title": "Senegal",
        "aliases": [
            "senegal"
        ],
        "categorization": []
    },
    "serbia": {
        "title": "Serbia",
        "aliases": [
            "serbia"
        ],
        "categorization": []
    },
    "seychelles": {
        "title": "Seychelles",
        "aliases": [
            "seychelles"
        ],
        "categorization": []
    },
    "sierra_leone": {
        "title": "Sierra Leone",
        "aliases": [
            "sierra_leone"
        ],
        "categorization": []
    },
    "singapore": {
        "title": "Singapore",
        "aliases": [
            "singapore"
        ],
        "categorization": []
    },
    "sint_maarten": {
        "title": "Sint Maarten",
        "aliases": [
            "sint_maarten",
            "sint_maarten_dutch_part"
        ],
        "categorization": []
    },
    "slovenia": {
        "title": "Slovenia",
        "aliases": [
            "slovenia"
        ],
        "categorization": []
    },
    "solomon_islands": {
        "title": "Solomon Islands",
        "aliases": [
            "solomon_islands"
        ],
        "categorization": []
    },
    "somalia": {
        "title": "Somalia",
        "aliases": [
            "somalia"
        ],
        "categorization": []
    },
    "south_africa": {
        "title": "South Africa",
        "aliases": [
            "south_africa"
        ],
        "categorization": []
    },
    "south_sudan": {
        "title": "South Sudan",
        "aliases": [
            "south_sudan"
        ],
        "categorization": []
    },
    "spain": {
        "title": "Spain",
        "aliases": [
            "spain"
        ],
        "categorization": []
    },
    "sri_lanka": {
        "title": "Sri Lanka",
        "aliases": [
            "sri_lanka"
        ],
        "categorization": []
    },
    "st_kitts_and_nevis": {
        "title": "St. Kitts and Nevis",
        "aliases": [
            "st_kitts_and_nevis"
        ],
        "categorization": []
    },
    "st_lucia": {
        "title": "St. Lucia",
        "aliases": [
            "st_lucia"
        ],
        "categorization": []
    },
    "st_martin": {
        "title": "St. Martin (French Part)",
        "aliases": [
            "st_martin",
            "st_martin_french_part"
        ],
        "categorization": []
    },
    "st_vincent_and_the_grenadines": {
        "title": "St. Vincent and the Grenadines",
        "aliases": [
            "st_vincent_and_the_grenadines"
        ],
        "categorization": []
    },
    "sudan": {
        "title": "Sudan",
        "aliases": [
            "sudan"
        ],
        "categorization": []
    },
    "suriname": {
        "title": "Suriname",
        "aliases": [
            "suriname"
        ],
        "categorization": []
    },
    "sweden": {
        "title": "Sweden",
        "aliases": [
            "sweden"
        ],
        "categorization": []
    },
    "switzerland": {
        "title": "Switzerland",
        "aliases": [
            "switzerland"
        ],
        "categorization": []
    },
    "tajikistan": {
        "title": "Tajikistan",
        "aliases": [
            "tajikistan"
        ],
        "categorization": []
    },
    "tanzania": {
        "title": "Tanzania",
        "aliases": [
            "tanzania"
        ],
        "categorization": []
    },
    "thailand": {
        "title": "Thailand",
        "aliases": [
            "thailand"
        ],
        "categorization": []
    },
    "timor_leste": {
        "title": "Timor-Leste",
        "aliases": [
            "timor-leste",
            "timor_leste"
        ],
        "categorization": []
    },
    "togo": {
        "title": "Togo",
        "aliases": [
            "togo"
        ],
        "categorization": []
    },
    "tonga": {
        "title": "Tonga",
        "aliases": [
            "tonga"
        ],
        "categorization": []
    },
    "trinidad_and_tobago": {
        "title": "Trinidad and Tobago",
        "aliases": [
            "trinidad_and_tobago"
        ],
        "categorization": []
    },
    "tunisia": {
        "title": "Tunisia",
        "aliases": [
            "tunisia"
        ],
        "categorization": []
    },
    "turkey": {
        "title": "Turkey",
        "aliases": [
            "turkey",
            "turkiye"
        ],
        "categorization": []
    },
    "turkmenistan": {
        "title": "Turkmenistan",
        "aliases": [
            "turkmenistan"
        ],
        "categorization": []
    },
    "turks_and_caicos_islands": {
        "title": "Turks and Caicos Islands",
        "aliases": [
            "turks_and_caicos_islands"
        ],
        "categorization": []
    },
    "tuvalu": {
        "title": "Tuvalu",
        "aliases": [
            "tuvalu"
        ],
        "categorization": []
    },
    "uganda": {
        "title": "Uganda",
        "aliases": [
            "uganda"
        ],
        "categorization": []
    },
    "ukraine": {
        "title": "Ukraine",
        "aliases": [
            "ukraine"
        ],
        "categorization": []
    },
    "united_arab_emirates": {
        "title": "United Arab Emirates",
        "aliases": [
            "uae",
            "united_arab_emirates"
        ],
        "categorization": []
    },
    "uruguay": {
        "title": "Uruguay",
        "aliases": [
            "uruguay"
        ],
        "categorization": []
    },
    "uzbekistan": {
        "title": "Uzbekistan",
        "aliases": [
            "uzbekistan"
        ],
        "categorization": []
    },
    "vanuatu": {
        "title": "Vanuatu",
        "aliases": [
            "vanuatu"
        ],
        "categorization": []
    },
    "venezuela": {
        "title": "Venezuela",
        "aliases": [
            "venezuela",
            "venezuela_rb"
        ],
        "categorization": []
    },
    "virgin_islands_us": {
        "title": "Virgin Islands (U.S.)",
        "aliases": [
            "virgin_islands_(u.s.)",
            "virgin_islands_us"
        ],
        "categorization": []
    },
    "west_bank_and_gaza": {
        "title": "West Bank and Gaza",
        "aliases": [
            "palestinian_territories",
            "west_bank_and_gaza"
        ],
        "categorization": []
    },
    "yemen": {
        "title": "Yemen",
        "aliases": [
            "yemen",
            "yemen_rep"
        ],
        "categorization": []
    },
    "zambia": {
        "title": "Zambia",
        "aliases": [
            "zambia"
        ],
        "categorization": []
    },
    "zimbabwe": {
        "title": "Zimbabwe",
        "aliases": [
            "zimbabwe"
        ],
        "categorization": []
    },
    "british_virgin_islands": {
        "title": "British Virgin Islands",
        "aliases": [
            "british_virgin_islands"
        ],
        "categorization": []
    },
    "curacao": {
        "title": "Cura\u00e7ao",
        "aliases": [
            "curacao"
        ],
        "categorization": []
    },
    "djibouti": {
        "title": "Djibouti",
        "aliases": [
            "djibouti"
        ],
        "categorization": []
    },
    "dominica": {
        "title": "Dominica",
        "aliases": [
            "dominica"
        ],
        "categorization": []
    },
    "dominican_republic": {
        "title": "Dominican Republic",
        "aliases": [
            "dominican_republic"
        ],
        "categorization": []
    },
    "bahamas": {
        "title": "Bahamas",
        "aliases": [
            "bahamas",
            "bahamas,_the",
            "bahamas_the",
            "the_bahamas"
        ],
        "categorization": []
    },
    "chad": {
        "title": "Chad",
        "aliases": [
            "chad"
        ],
        "categorization": []
    },
    "cayman_islands": {
        "title": "Cayman Islands",
        "aliases": [
            "cayman_islands"
        ],
        "categorization": []
    },
    "central_african_republic": {
        "title": "Central African Republic",
        "aliases": [
            "central_african_republic"
        ],
        "categorization": []
    },
    "channel_islands": {
        "title": "Channel Islands",
        "aliases": [
            "channel_islands"
        ],
        "categorization": []
    },
    "regions_patch": {
        "title": "Regions Patch",
        "aliases": [
            "early_demographic_dividend",
            "east_asia_&_pacific_(excluding_high_income)",
            "east_asia_&_pacific_(ida_&_ibrd_countries)",
            "europe_&_central_asia_(excluding_high_income)",
            "europe_&_central_asia_(ida_&_ibrd_countries)",
            "heavily_indebted_poor_countries_(hipc)",
            "late_demographic_dividend",
            "latin_america_&_caribbean_(excluding_high_income)",
            "latin_america_&_the_caribbean_(ida_&_ibrd_countries)",
            "middle_east_&_north_africa_(excluding_high_income)",
            "middle_east_&_north_africa_(ida_&_ibrd_countries)",
            "post_demographic_dividend",
            "pre_demographic_dividend",
            "south_asia_(ida_&_ibrd)",
            "sub_saharan_africa",
            "sub_saharan_africa_(excluding_high_income)",
            "sub_saharan_africa_(ida_&_ibrd_countries)",
            "upper_middle_income"
        ],
        "categorization": []
    },
    "afghanistan": {
        "title": "Afghanistan",
        "aliases": [
            "afghanistan"
        ],
        "categorization": []
    },
    "albania": {
        "title": "Albania",
        "aliases": [
            "albania"
        ],
        "categorization": []
    },
    "algeria": {
        "title": "Algeria",
        "aliases": [
            "algeria"
        ],
        "categorization": []
    },
    "american_samoa": {
        "title": "American Samoa",
        "aliases": [
            "american_samoa"
        ],
        "categorization": []
    },
    "andorra": {
        "title": "Andorra",
        "aliases": [
            "andorra"
        ],
        "categorization": []
    },
    "angola": {
        "title": "Angola",
        "aliases": [
            "angola"
        ],
        "categorization": []
    },
    "antigua_and_barbuda": {
        "title": "Antigua and Barbuda",
        "aliases": [
            "antigua_and_barbuda"
        ],
        "categorization": []
    },
    "argentina": {
        "title": "Argentina",
        "aliases": [
            "argentina"
        ],
        "categorization": []
    },
    "armenia": {
        "title": "Armenia",
        "aliases": [
            "armenia"
        ],
        "categorization": []
    },
    "aruba": {
        "title": "Aruba",
        "aliases": [
            "aruba"
        ],
        "categorization": []
    },
    "australia": {
        "title": "Australia",
        "aliases": [
            "australia"
        ],
        "categorization": []
    },
    "austria": {
        "title": "Austria",
        "aliases": [
            "austria"
        ],
        "categorization": []
    },
    "azerbaijan": {
        "title": "Azerbaijan",
        "aliases": [
            "azerbaijan"
        ],
        "categorization": []
    },
    "bahrain": {
        "title": "Bahrain",
        "aliases": [
            "bahrain"
        ],
        "categorization": []
    },
    "bangladesh": {
        "title": "Bangladesh",
        "aliases": [
            "bangladesh"
        ],
        "categorization": []
    },
    "barbados": {
        "title": "Barbados",
        "aliases": [
            "barbados"
        ],
        "categorization": []
    },
    "belarus": {
        "title": "Belarus",
        "aliases": [
            "belarus"
        ],
        "categorization": []
    },
    "belgium": {
        "title": "Belgium",
        "aliases": [
            "belgium"
        ],
        "categorization": []
    },
    "belize": {
        "title": "Belize",
        "aliases": [
            "belize"
        ],
        "categorization": []
    },
    "benin": {
        "title": "Benin",
        "aliases": [
            "benin"
        ],
        "categorization": []
    },
    "bermuda": {
        "title": "Bermuda",
        "aliases": [
            "bermuda"
        ],
        "categorization": []
    },
    "bhutan": {
        "title": "Bhutan",
        "aliases": [
            "bhutan"
        ],
        "categorization": []
    },
    "bolivia": {
        "title": "Bolivia",
        "aliases": [
            "bolivia"
        ],
        "categorization": []
    },
    "bosnia_and_herzegovina": {
        "title": "Bosnia and Herzegovina",
        "aliases": [
            "bosnia_and_herzegovina"
        ],
        "categorization": []
    },
    "botswana": {
        "title": "Botswana",
        "aliases": [
            "botswana"
        ],
        "categorization": []
    },
    "brazil": {
        "title": "Brazil",
        "aliases": [
            "brazil"
        ],
        "categorization": []
    },
    "brunei": {
        "title": "Brunei",
        "aliases": [
            "brunei",
            "brunei_darussalam"
        ],
        "categorization": []
    },
    "bulgaria": {
        "title": "Bulgaria",
        "aliases": [
            "bulgaria"
        ],
        "categorization": []
    },
    "burkina_faso": {
        "title": "Burkina Faso",
        "aliases": [
            "burkina_faso"
        ],
        "categorization": []
    },
    "burundi": {
        "title": "Burundi",
        "aliases": [
            "burundi"
        ],
        "categorization": []
    },
    "cabo_verde": {
        "title": "Cabo Verde",
        "aliases": [
            "cabo_verde",
            "cape_verde"
        ],
        "categorization": []
    },
    "cambodia": {
        "title": "Cambodia",
        "aliases": [
            "cambodia"
        ],
        "categorization": []
    },
    "cameroon": {
        "title": "Cameroon",
        "aliases": [
            "cameroon"
        ],
        "categorization": []
    },
    "canada": {
        "title": "Canada",
        "aliases": [
            "canada"
        ],
        "categorization": []
    },
    "chile": {
        "title": "Chile",
        "aliases": [
            "chile"
        ],
        "categorization": []
    },
    "china": {
        "title": "China",
        "aliases": [
            "china",
            "mainland_china",
            "people_republic_of_china"
        ],
        "categorization": []
    },
    "colombia": {
        "title": "Colombia",
        "aliases": [
            "colombia"
        ],
        "categorization": []
    },
    "comoros": {
        "title": "Comoros",
        "aliases": [
            "comoros"
        ],
        "categorization": []
    },
    "congo_democratic_republic": {
        "title": "Congo, Dem. Rep.",
        "aliases": [
            "congo,_dem._rep.",
            "congo_dem_rep",
            "congo_democratic_republic",
            "congo_kinshasa",
            "dr_congo",
            "drc"
        ],
        "categorization": []
    },
    "congo_republic": {
        "title": "Congo, Rep.",
        "aliases": [
            "congo,_rep.",
            "congo_brazzaville",
            "congo_rep",
            "congo_republic"
        ],
        "categorization": []
    },
    "costa_rica": {
        "title": "Costa Rica",
        "aliases": [
            "costa_rica"
        ],
        "categorization": []
    },
    "croatia": {
        "title": "Croatia",
        "aliases": [
            "croatia"
        ],
        "categorization": []
    },
    "cuba": {
        "title": "Cuba",
        "aliases": [
            "cuba"
        ],
        "categorization": []
    },
    "cyprus": {
        "title": "Cyprus",
        "aliases": [
            "cyprus"
        ],
        "categorization": []
    },
    "czechia": {
        "title": "Czechia",
        "aliases": [
            "czech_republic",
            "czechia"
        ],
        "categorization": []
    },
    "denmark": {
        "title": "Denmark",
        "aliases": [
            "denmark"
        ],
        "categorization": []
    }
}




    

if __name__ == "__main__":
    main()