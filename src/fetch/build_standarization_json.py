
import os
import json
from src.utils import get_metadata_dir


def main():

    metadata_path = get_metadata_dir()

    os.makedirs(metadata_path, exist_ok=True)

    standardization_json = metadata_path / "standardization.json"

    with open(standardization_json, 'w') as file:
        json.dump(standardization, file, indent=4)
        print(f"standarization.json saved to: {standardization_json.parent}")



standardization = {
    "countries": {
        "united_states": {
            "title": "United States",
            "aliases": ["USA", "United States", "US", "United States of America"],
            "categorization": ["Western offshoot", "high"]
        },
        "united_kingdom": {
            "title": "United Kingdom",
            "aliases": ["UK", "United Kingdom", "Great Britain", "England"],
            "categorization": ["Western Europe", "high"]
        },
        "germany": {
            "title": "Germany",
            "aliases": ["Germany", "DE", "Deutschland"],
            "categorization": ["Western Europe", "high"]
        },
        "france": {
            "title": "France",
            "aliases": ["France", "FR", "French Republic"],
            "categorization": ["Western Europe", "high"]
        },
        "japan": {
            "title": "Japan",
            "aliases": ["Japan", "JP", "Nippon"],
            "categorization": ["East Asia", "high"]
        },
        "china": {
            "title": "China",
            "aliases": ["China", "CN", "People's Republic of China"],
            "categorization": ["East Asia", "upper-middle"]
        },
        "india": {
            "title": "India",
            "aliases": ["India", "IN", "Republic of India"],
            "categorization": ["South and South East Asia", "lower-middle"]
        },
        "brazil": {
            "title": "Brazil",
            "aliases": ["Brazil", "BR", "Brasil"],
            "categorization": ["Latin America", "upper-middle"]
        },
        "russia": {
            "title": "Russia",
            "aliases": ["Russia", "RU", "Russian Federation"],
            "categorization": ["Eastern Europe", "upper-middle"]
        },
        "canada": {
            "title": "Canada",
            "aliases": ["Canada", "CA", "Canadian"],
            "categorization": ["Western offshoot", "high"]
        },
        "australia": {
            "title": "Australia",
            "aliases": ["Australia", "AU", "Commonwealth of Australia"],
            "categorization": ["Western offshoot", "high"]
        },
        "italy": {
            "title": "Italy",
            "aliases": ["Italy", "IT", "Italia"],
            "categorization": ["Western Europe", "high"]
        },
        "spain": {
            "title": "Spain",
            "aliases": ["Spain", "ES", "España"],
            "categorization": ["Western Europe", "high"]
        },
        "mexico": {
            "title": "Mexico",
            "aliases": ["Mexico", "MX", "México"],
            "categorization": ["Latin America", "upper-middle"]
        },
        "south_korea": {
            "title": "South Korea",
            "aliases": ["South Korea", "KR", "Republic of Korea", "Korea, Republic of"],
            "categorization": ["East Asia", "high"]
        },
        "indonesia": {
            "title": "Indonesia",
            "aliases": ["Indonesia", "ID", "Republic of Indonesia"],
            "categorization": ["South and South East Asia", "lower-middle"]
        },
        "turkey": {
            "title": "Turkey",
            "aliases": ["Turkey", "TR", "Türkiye"],
            "categorization": ["Middle East and North Africa", "upper-middle"]
        },
        "saudi_arabia": {
            "title": "Saudi Arabia",
            "aliases": ["Saudi Arabia", "SA", "Kingdom of Saudi Arabia"],
            "categorization": ["Middle East and North Africa", "high"]
        },
        "argentina": {
            "title": "Argentina",
            "aliases": ["Argentina", "AR", "Argentine Republic"],
            "categorization": ["Latin America", "upper-middle"]
        },
        "south_africa": {
            "title": "South Africa",
            "aliases": ["South Africa", "ZA", "Republic of South Africa"],
            "categorization": ["Sub Saharan Africa", "upper-middle"]
        },
        "netherlands": {
            "title": "Netherlands",
            "aliases": ["Netherlands", "NL", "Holland", "Nederland"],
            "categorization": ["Western Europe", "high"]
        },
        "sweden": {
            "title": "Sweden",
            "aliases": ["Sweden", "SE", "Kingdom of Sweden"],
            "categorization": ["Western Europe", "high"]
        },
        "switzerland": {
            "title": "Switzerland",
            "aliases": ["Switzerland", "CH", "Swiss Confederation"],
            "categorization": ["Western Europe", "high"]
        },
        "belgium": {
            "title": "Belgium",
            "aliases": ["Belgium", "BE", "Kingdom of Belgium"],
            "categorization": ["Western Europe", "high"]
        },
        "norway": {
            "title": "Norway",
            "aliases": ["Norway", "NO", "Kingdom of Norway"],
            "categorization": ["Western Europe", "high"]
        },
        "denmark": {
            "title": "Denmark",
            "aliases": ["Denmark", "DK", "Kingdom of Denmark"],
            "categorization": ["Western Europe", "high"]
        },
        "finland": {
            "title": "Finland",
            "aliases": ["Finland", "FI", "Republic of Finland"],
            "categorization": ["Western Europe", "high"]
        },
        "poland": {
            "title": "Poland",
            "aliases": ["Poland", "PL", "Republic of Poland"],
            "categorization": ["Eastern Europe", "high"]
        },
        "austria": {
            "title": "Austria",
            "aliases": ["Austria", "AT", "Republic of Austria"],
            "categorization": ["Western Europe", "high"]
        },
        "ireland": {
            "title": "Ireland",
            "aliases": ["Ireland", "IE", "Republic of Ireland"],
            "categorization": ["Western Europe", "high"]
        },
        "new_zealand": {
            "title": "New Zealand",
            "aliases": ["New Zealand", "NZ", "Aotearoa"],
            "categorization": ["Western offshoot", "high"]
        },
        "singapore": {
            "title": "Singapore",
            "aliases": ["Singapore", "SG", "Republic of Singapore"],
            "categorization": ["South and South East Asia", "high"]
        },
        "malaysia": {
            "title": "Malaysia",
            "aliases": ["Malaysia", "MY", "Federation of Malaysia"],
            "categorization": ["South and South East Asia", "upper-middle"]
        },
        "thailand": {
            "title": "Thailand",
            "aliases": ["Thailand", "TH", "Kingdom of Thailand"],
            "categorization": ["South and South East Asia", "upper-middle"]
        },
        "philippines": {
            "title": "Philippines",
            "aliases": ["Philippines", "PH", "Republic of the Philippines"],
            "categorization": ["South and South East Asia", "lower-middle"]
        },
        "vietnam": {
            "title": "Vietnam",
            "aliases": ["Vietnam", "VN", "Socialist Republic of Vietnam"],
            "categorization": ["South and South East Asia", "lower-middle"]
        },
        "nigeria": {
            "title": "Nigeria",
            "aliases": ["Nigeria", "NG", "Federal Republic of Nigeria"],
            "categorization": ["Sub Saharan Africa", "low"]
        },
        "egypt": {
            "title": "Egypt",
            "aliases": ["Egypt", "EG", "Arab Republic of Egypt"],
            "categorization": ["Middle East and North Africa", "lower-middle"]
        },
        "pakistan": {
            "title": "Pakistan",
            "aliases": ["Pakistan", "PK", "Islamic Republic of Pakistan"],
            "categorization": ["South and South East Asia", "lower-middle"]
        },
        "bangladesh": {
            "title": "Bangladesh",
            "aliases": ["Bangladesh", "BD", "People's Republic of Bangladesh"],
            "categorization": ["South and South East Asia", "lower-middle"]
        },
        "iran": {
            "title": "Iran",
            "aliases": ["Iran", "IR", "Islamic Republic of Iran"],
            "categorization": ["Middle East and North Africa", "upper-middle"]
        },
        "iraq": {
            "title": "Iraq",
            "aliases": ["Iraq", "IQ", "Republic of Iraq"],
            "categorization": ["Middle East and North Africa", "upper-middle"]
        },
        "israel": {
            "title": "Israel",
            "aliases": ["Israel", "IL", "State of Israel"],
            "categorization": ["Middle East and North Africa", "high"]
        },
        "greece": {
            "title": "Greece",
            "aliases": ["Greece", "GR", "Hellenic Republic"],
            "categorization": ["Western Europe", "high"]
        },
        "portugal": {
            "title": "Portugal",
            "aliases": ["Portugal", "PT", "Portuguese Republic"],
            "categorization": ["Western Europe", "high"]
        },
        "hungary": {
            "title": "Hungary",
            "aliases": ["Hungary", "HU", "Hungarian Republic"],
            "categorization": ["Eastern Europe", "upper-middle"]
        },
        "czech_republic": {
            "title": "Czech Republic",
            "aliases": ["Czech Republic", "CZ", "Czechia"],
            "categorization": ["Eastern Europe", "high"]
        },
        "romania": {
            "title": "Romania",
            "aliases": ["Romania", "RO", "Republic of Romania"],
            "categorization": ["Eastern Europe", "upper-middle"]
        },
        "chile": {
            "title": "Chile",
            "aliases": ["Chile", "CL", "Republic of Chile"],
            "categorization": ["Latin America", "upper-middle"]
        },
        "colombia": {
            "title": "Colombia",
            "aliases": ["Colombia", "CO", "Republic of Colombia"],
            "categorization": ["Latin America", "upper-middle"]
        },
        "peru": {
            "title": "Peru",
            "aliases": ["Peru", "PE", "Republic of Peru"],
            "categorization": ["Latin America", "upper-middle"]
        },
        "venezuela": {
            "title": "Venezuela",
            "aliases": ["Venezuela", "VE", "Bolivarian Republic of Venezuela"],
            "categorization": ["Latin America", "upper-middle"]
        },
        "uae": {
            "title": "United Arab Emirates",
            "aliases": ["UAE", "United Arab Emirates", "AE"],
            "categorization": ["Middle East and North Africa", "high"]
        }
    },
    "categories": {
        1: {
            "title": "demographic",
            "classes": ["East Asia", "Eastern Europe", "Latin America", "Middle East and North Africa", "South and South East Asia", "Sub Saharan Africa", "Western Europe", "Western offshoot"]
        },
        2: {
            "title": "income",
            "classes": ["low", "lower-middle", "upper-middle", "high"]
        }
    }
}

if __name__ == "__main__":
    main()