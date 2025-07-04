from src.fetch.apis import BRGovAPI
from src.utils import get_data_dir
from src.logger import setup_logger


def main():
    logger = setup_logger()    
    parent = get_data_dir() / "00--raw/macro"
    index_csv = parent / "indexes.csv"

    start_date = "2000-01-01"

    logger.info("downloading data...")

    api = BRGovAPI(csv_path=index_csv)
    api.save_indexes_csv(series_codes, start_date, last_date=None)




series_codes = [
    # Índices de Preços
    ("selic", 432),

    ("inpc", 188),
    ("ipca", 433),
    ("ipca_15", 7478),
    ("ipca_e", 10764),
    ("ipca_12m", 13522),
    ("igp_10", 7447),
    ("igp_di", 190),
    ("igp_m_1_dec", 7448),
    ("igp_m_2_dec", 7449),
    ("igp_m", 189),
    ("ipc_fipe_1_q", 7463),
    ("ipc_fipe_2_q", 272),
    ("ipc_fipe_3_q", 7464),
    ("ipc_fipe_mensal", 193),

    # Índices Gerais de Preços e Índices de Preços por Atacado
    ("igp_m", 189),
    ("ipa_m", 7450),
    ("igp_di", 190),
    ("ipa_di_geral", 225),
    ("ipa_di_industriais", 7459),
    ("ipa_di_agricolas", 7460),

    # Índices de Preços ao Consumidor
    ("ipc_di", 191),
    ("ipc_c1", 17680),
    ("ipc_3i", 17679),
    ("inpc", 188),
    
    # IPCA - Evolução dos Preços
    ("ipca_livres", 11428),
    ("ipca_comercializaveis", 4447),
    ("ipca_nao_comercializaveis", 4448),
    ("ipca_monitorados", 4449),

    # IPCA - Variações Percentuais Mensais
    ("ipca", 433),
    ("ipca_alimentacao_bebidas", 1635),
    ("ipca_habitacao", 1636),
    ("ipca_artigos_residencia", 1637),
    ("ipca_vestuario", 1638),
    ("ipca_transportes", 1639),
    ("ipca_saude_cuidados_pessoais", 1641),
    ("ipca_despesas_pessoais", 1642),
    ("ipca_educacao", 1643),
    ("ipca_comunicacao", 1640),

    # IPCA - Variações Percentuais nos Últimos 12 Meses
    ("ipca_12m", 433),  # Using the same code as IPCA

    # IPCA - Evolução dos Preços dos Bens
    ("ipca_duraveis", 10843),
    ("ipca_semiduraveis", 10842),
    ("ipca_nao_duraveis", 10841),
    ("ipca_servicos", 10844),
    ("ipca_monitorados", 4449),

    # IPCA - Núcleos
    ("ipca_ex1", 1621),
    ("ipca_ms", 4466),
    ("ipca_dp", 16122),

    # Valor da Cesta Básica
    ("cesta_aracaju", 7479),
    ("cesta_belem", 7480),
    ("cesta_belo_horizonte", 7481),
    ("cesta_brasilia", 7482),
    ("cesta_curitiba", 7483),
    ("cesta_florianopolis", 7484),
    ("cesta_fortaleza", 7485),
    ("cesta_goiania", 7486),
    ("cesta_joao_pessoa", 7487),
    ("cesta_natal", 7488),
    ("cesta_porto_alegre", 7489),
    ("cesta_recife", 7490),
    ("cesta_rio_de_janeiro", 7491),
    ("cesta_salvador", 7492),
    ("cesta_sao_paulo", 7493),
    ("cesta_vitoria", 7494),

    # Indicadores da Conjuntura Econômica
    ("producao_industrial_total", 21859),
    ("producao_industrial_transformacao", 21862),
    ("producao_industrial_extrativa", 21861),
    ("producao_aco_bruto", 7357),
    ("producao_insumos_construcao_civil", 21868),
    ("consultas_spc", 1453),
    ("consultas_usecheque", 1454),
    ("vendas_reais_abras", 7414),

    # Tráfego de veículos pesados nas estradas pedagiadas
    ("trafego_veiculos_pesados", 28552),

    # Consultas ao Serasa
    ("consultas_serasa", 28547),

    # UCI FGV
    ("uci_fgv", 24352),

    # UCI CNI
    ("uci_cni", 24351),

    # Vendas industriais reais
    ("vendas_industriais_reais", 1338),

    # Horas trabalhadas na produção da indústria de transformação
    ("horas_trabalhadas_transformacao", 24348),

    # Salário real na indústria de transformação
    ("salario_real_transformacao", 28558),

    # Massa salarial real na indústria de transformação
    ("massa_salarial_real_transformacao", 24349),

    # Produção de petróleo óleo bruto
    ("producao_petroleo_bruto", 1389),

    # Produção de gás natural
    ("producao_gas_natural", 1392),

    # Indicadores da Conjuntura Econômica - Séries Dessazonalizadas
    ("producao_industrial_total_dessaz", 28503),
    ("producao_industrial_transformacao_dessaz", 28505),
    ("producao_industrial_extrativa_dessaz", 28504),
    ("producao_aco_bruto_dessaz", 28546),
    ("trafego_veiculos_pesados_dessaz", 28553),
    ("producao_insumos_construcao_civil_dessaz", 28511),
    ("consultas_spc_usecheque_dessaz", 28550 + 28551),
    ("consultas_serasa_dessaz", 28548),
    ("vendas_reais_abras_dessaz", 28549),
    ("uci_fgv_dessaz", 28561),
    ("uci_cni_dessaz", 28554),
    ("vendas_industriais_reais_dessaz", 28555),
    ("horas_trabalhadas_transformacao_dessaz", 28556),
    ("salario_real_transformacao_dessaz", 28559),
    ("massa_salarial_real_transformacao_dessaz", 28560),

    # Índices da Produção Industrial
    ("producao_industrial_geral", 21858),
    ("producao_bens_capital", 21863),
    ("producao_bens_intermediarios", 21864),
    ("producao_bens_consumo_geral", 21865),
    ("producao_bens_duraveis", 21866),
    ("producao_bens_nao_duraveis_semiduraveis", 21867),

    # Índices da Produção Industrial - Dessazonalizados
    ("producao_industrial_geral_dessaz", 28503),
    ("producao_bens_capital_dessaz", 28506),
    ("producao_bens_intermediarios_dessaz", 28507),
    ("producao_bens_consumo_geral_dessaz", 28508),
    ("producao_bens_duraveis_dessaz", 28509),
    ("producao_bens_nao_duraveis_semiduraveis_dessaz", 28510),

    # Indústria Automobilística e Importação de Autoveículos
    ("producao_autoveiculos", 1373),
    ("licenciamentos_autoveiculos_novos", 1379),
    ("exportacao_autoveiculos", 1380),
    ("total_licenciamentos_exportacoes", 1378),

    # Indústria Automobilística e Importação de Autoveículos - Dessazonalizados
    ("producao_autoveiculos_dessaz", 28527),
    ("licenciamentos_autoveiculos_novos_dessaz", 28529),
    ("exportacao_autoveiculos_dessaz", 28530),
    ("total_licenciamentos_exportacoes_dessaz", 28528),

    # Produção de Autoveículos
    ("producao_autoveiculos_total", 1373),
    ("producao_automoveis_comerciais_leves", 1374),
    ("producao_caminhoes", 1375),
    ("producao_onibus", 1376),

    # Índices de Expectativas do Consumidor e do Empresário Industrial
    ("icc_geral", 4393),
    ("icc_expectativas_futuras", 4394),
    ("icc_condicoes_economicas_atuais", 4395),
    ("icei_geral", 7341),
    ("icei_condicoes_atuais", 7342),
    ("icei_expectativas", 7343),

    # Índice de Volume de Vendas no Varejo
    ("vendas_varejo_geral", 1455),
    ("vendas_comercio_ampliado", 20106),
    ("vendas_hiper_super_produtos_alimenticios", 1496),
    ("vendas_moveis_eletrodomesticos", 1522),
    ("vendas_automoveis_motocicletas_pecas", 1548),
    ("vendas_material_construcao", 20105),

    # Índice de Volume de Vendas no Varejo - Dessazonalizados
    ("vendas_varejo_geral_dessaz", 28473),
    ("vendas_comercio_ampliado_dessaz", 28485),
    ("vendas_hiper_super_produtos_alimenticios_dessaz", 28475),
    ("vendas_moveis_eletrodomesticos_dessaz", 28478),
    ("vendas_automoveis_motocicletas_pecas_dessaz", 28479),
    ("vendas_material_construcao_dessaz", 28484),

    # Indicadores de Investimento
    ("prod_insumos_construcao_civil", 21868),
    ("producao_bens_capital", 21863),
    ("faturamento_real_bk_mecanicos", 7358),
    ("desembolso_bndes", 7415),

    # Indicadores de Investimento - Dessazonalizados
    ("exportacao_bens_capital", 28567),
    ("importacao_bens_capital", 28568),
    ("exportacao_bens_capital_dessaz", 28569),
    ("importacao_bens_capital_dessaz", 28570),

    # Índice do Nível de Emprego Formal
    ("emprego_formal_total", 25239),
    ("emprego_formal_transformacao", 25241),
    ("emprego_formal_comercio", 25256),
    ("emprego_formal_servicos", 25257),
    ("emprego_formal_construcao_civil", 25255),

    # Índice do Nível de Emprego Formal - Dessazonalizados
    ("emprego_formal_total_dessaz", 28512),
    ("emprego_formal_transformacao_dessaz", 28513),
    ("emprego_formal_comercio_dessaz", 28514),
    ("emprego_formal_servicos_dessaz", 28515),
    ("emprego_formal_construcao_civil_dessaz", 28516),

    # Força de Trabalho e População em Idade Ativa
    ("forca_trabalho_ocupadas", 24379),
    ("forca_trabalho_desocupadas", 24380),
    ("forca_trabalho_total", 24378),
    ("pessoas_idade_trabalhar", 24370),
    ("taxa_desocupacao", 24369),

    # Taxa de Desocupação
    ("taxa_desocupacao_media", 24369),
    ("taxa_desocupacao_norte", 28562),
    ("taxa_desocupacao_centro_oeste", 28563),
    ("taxa_desocupacao_nordeste", 28564),
    ("taxa_desocupacao_sudeste", 28565),
    ("taxa_desocupacao_sul", 28566),

    # Rendimento Médio Real Efetivo de Todos os Trabalhos
    ("remuneracao_media_deflac", 24381),
    ("remuneracao_media_nominal", 24382),
    ("pessoal_ocupado_rendimento", 28543),
    ("massa_salarial", 28544),

    # Produto Interno Bruto e Taxas Médias de Crescimento
    ("pib_preco_corrente", 1207),
    ("pib_em_rs_ultimo_ano", 1208),
    ("pib_em_usd", 7324),
    ("populacao", 21774),
    ("pib_per_capita_preco_corrente", 21775),
    ("pib_per_capita_em_rs_ultimo_ano", 21777),
    ("pib_per_capita_em_usd", 21776),

    # Contas Nacionais Trimestrais
    ("pib_trimestral", 22099),
    ("consumo_familias", 22100),
    ("consumo_governo", 22101),
    ("fbcf", 22102),
    ("exportacao", 22103),
    ("importacao", 22104),
    ("pib_dessaz", 22109),
    ("consumo_familias_dessaz", 22110),
    ("consumo_governo_dessaz", 22111),
    ("fbcf_dessaz", 22113),
    ("exportacao_dessaz", 22114),
    ("importacao_dessaz", 22115),

    # Produto Interno Bruto Trimestral
    ("pib_trimestral", 22099),
    ("pib_valor_adicionado", 22097),
    ("pib_agropecuaria", 22083),
    ("pib_industria", 22084),
    ("pib_servicos", 22089),
    ("pib_trimestral_dessaz", 22109),
    ("pib_valor_adicionado_dessaz", 22108),
    ("pib_agropecuaria_dessaz", 22105),
    ("pib_industria_dessaz", 22106),
    ("pib_servicos_dessaz", 22107)
]



if __name__ == "__main__":

    main()

