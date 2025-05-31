# -*- coding: utf-8 -*-
"""
meu_modelo_lasso.py

Módulo Python contendo funções para carregar dados, engenheirar atributos,
treinar, avaliar e visualizar um modelo Lasso para previsão de séries temporais.
"""

# --- Importações ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Configurações Iniciais ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Funções Auxiliares e Métricas ---

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcula o Erro Percentual Absoluto Médio (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# --- Função de Carregamento e Preparação ---

def load_and_prepare_data(file_path):
    """
    Carrega os dados de um arquivo CSV e prepara o índice 'Year'.

    Args:
        file_path (str): O caminho para o arquivo CSV.

    Returns:
        pd.DataFrame or None: DataFrame com dados carregados e índice 'Year'
                              ou None se ocorrer um erro.
    """
    try:
        data_yearly = pd.read_csv(file_path)
        print(f"Dados carregados de '{file_path}'.")
    except FileNotFoundError:
        print(f"Erro: '{file_path}' não encontrado.")
        return None

    # Limpa colunas 'Unnamed'
    unnamed_cols = [col for col in data_yearly.columns if 'Unnamed:' in col]
    if unnamed_cols:
        if 'Unnamed: 0' in unnamed_cols and 'Year' not in data_yearly.columns and 'year' not in data_yearly.columns:
            try:
                # Tenta verificar se a coluna parece ser ano
                if data_yearly['Unnamed: 0'].iloc[0] >= 1900 and data_yearly['Unnamed: 0'].iloc[0] <= 2050:
                    print("Renomeando 'Unnamed: 0' para 'Year'.")
                    data_yearly = data_yearly.rename(columns={'Unnamed: 0': 'Year'})
                    unnamed_cols.remove('Unnamed: 0')
            except Exception as e:
                print(f"Aviso: Não foi possível usar 'Unnamed: 0' como ano: {e}")

        if unnamed_cols: # Se ainda houver colunas 'Unnamed'
            print(f"Removendo colunas 'Unnamed': {unnamed_cols}")
            data_yearly = data_yearly.drop(columns=unnamed_cols)

    # Prepara o Índice (Year)
    year_col_found = None
    for col_name in ['Year', 'year']:
        if col_name in data_yearly.columns:
            year_col_found = col_name
            break

    if year_col_found:
        print(f"Usando '{year_col_found}' como índice.")
        data_yearly = data_yearly.set_index(year_col_found)
        if 'year' in data_yearly.columns and year_col_found != 'year':
             data_yearly = data_yearly.drop('year', axis=1)
    elif data_yearly.shape[0] >= 10:
        start_year = pd.Timestamp.now().year - data_yearly.shape[0] + 1
        end_year = pd.Timestamp.now().year
        print(f"Aviso: Coluna de ano não encontrada. Assumindo índice como {start_year}-{end_year}.")
        data_yearly.index = range(start_year, end_year + 1)
        data_yearly.index.name = 'Year'
    else:
        print("Erro: Não foi possível determinar o índice 'Year'.")
        return None

    return data_yearly

# --- Função de Engenharia de Atributos ---

def engineer_features(df, target_variable):
    """
    Cria features (lags, termos polinomiais de grau 2 e 3) e trata valores ausentes.

    Args:
        df (pd.DataFrame): DataFrame original com índice de tempo.
        target_variable (str): Nome da coluna alvo.

    Returns:
        tuple (pd.DataFrame, pd.Series) or (None, None):
              Um tuple contendo X (atributos) e y (alvo),
              ou (None, None) se a variável alvo não for encontrada.
    """
    if target_variable not in df.columns:
        print(f"Erro: A variável alvo '{target_variable}' não foi encontrada.")
        return None, None

    print("\nIniciando Engenharia de Atributos...")
    df_features = pd.DataFrame(index=df.index)

    for col in df.columns:
        # Não criar features para a própria variável alvo (se ela já estiver no loop de colunas)
        # As features são baseadas nos lags das *outras* colunas e da própria alvo.
        # A coluna alvo original será adicionada depois.
        
        lag1_col_name = f'{col}_lag1'
        df_features[lag1_col_name] = df[col].shift(1)
        

        # Adiciona termos polinomiais
        df_features[f'{col}_lag1_sq'] = df_features[lag1_col_name] ** 2  # Grau 2 (Quadrado)
    
    # Adiciona a variável alvo ao df_features para alinhamento e remoção de NaNs
    # Ela será removida de X posteriormente.
    df_features[target_variable] = df[target_variable]
    print("   - Lags e Termos Polinomiais (grau 2 e 3) criados.")

    # Trata valores infinitos que podem surgir de operações como x**3 com números muito grandes
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_rows = df_features.shape[0]
    df_features = df_features.dropna() # Remove linhas com NaN (principalmente devido ao shift(1))
    final_rows = df_features.shape[0]
    print(f"   - Valores ausentes/infinitos removidos ({initial_rows - final_rows} linhas).")

    if df_features.empty:
        print("Erro: Nenhum dado restante após remover NaNs. Verifique seus dados de entrada e a criação de features.")
        return None, None

    # Separa features (X) e alvo (y)
    # Garante que a variável alvo não esteja em X
    X = df_features.drop(columns=[target_variable])
    y = df_features[target_variable]

    print(f"Engenharia concluída. X: {X.shape}, y: {y.shape}")
    return X, y

# --- Função de Divisão e Normalização ---

def split_and_scale_data(X, y, train_limit_year):
    """
    Divide os dados em treino/teste e normaliza X e y.

    Args:
        X (pd.DataFrame): DataFrame de atributos.
        y (pd.Series): Series da variável alvo.
        train_limit_year (int): O último ano para o conjunto de treino.

    Returns:
        tuple or None: Contém X_train_scaled, X_test_scaled, y_train_scaled,
                       y_test_scaled, y_train, y_test, x_scaler, y_scaler.
                       Retorna None se o teste ficar vazio.
    """
    print(f"\nDividindo dados em {train_limit_year} e normalizando...")
    X_train = X[X.index <= train_limit_year]
    X_test = X[X.index > train_limit_year]
    y_train = y[y.index <= train_limit_year]
    y_test = y[y.index > train_limit_year]

    if X_test.empty or X_train.empty:
        print("Erro: Conjunto de treino ou teste está vazio. Verifique 'train_limit_year'.")
        return None

    print(f"   - Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    print("Divisão e normalização concluídas.")
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            y_train, y_test, x_scaler, y_scaler)

# --- Função de Treinamento ---

def train_lasso_model(X_train_scaled, y_train_scaled, **kwargs):
    """
    Treina um modelo LassoCV.

    Args:
        X_train_scaled (pd.DataFrame): Atributos de treino normalizados.
        y_train_scaled (np.array): Alvo de treino normalizado.
        **kwargs: Argumentos adicionais para LassoCV (cv, max_iter, etc.).

    Returns:
        sklearn.linear_model.LassoCV: O modelo treinado.
    """
    print("\nTreinando o Modelo LassoCV...")
    lasso_cv_model = LassoCV(**kwargs)
    lasso_cv_model.fit(X_train_scaled, y_train_scaled)
    print(f"Treinamento concluído. Melhor Alpha (α): {lasso_cv_model.alpha_:.6f}")
    return lasso_cv_model


def train_lasso_model_guided_cv(X_train_scaled, y_train_scaled, alpha_list, **kwargs):
    """
    Treina um modelo LassoCV usando uma lista específica de alfas.
    """
    print(f"\nTreinando o Modelo LassoCV com Alfas Definidos (de {min(alpha_list):.6f} a {max(alpha_list):.6f})...")
    # Remove 'n_alphas' se existir, pois vamos fornecer a lista
    kwargs.pop('n_alphas', None) 
    
    lasso_cv_model = LassoCV(alphas=alpha_list, **kwargs) # Usa LassoCV com a lista 'alphas'
    lasso_cv_model.fit(X_train_scaled, y_train_scaled)
    print(f"Treinamento concluído. Melhor Alpha (α) escolhido: {lasso_cv_model.alpha_:.6f}")
    return lasso_cv_model

# --- Função de Avaliação ---

# --- Função de Avaliação (Modificada) ---

def evaluate_model(model, X_test_scaled, y_test, y_test_scaled, y_scaler):
    """
    Realiza previsões e calcula métricas de avaliação, incluindo
    as métricas na escala normalizada e original.

    Args:
        model: O modelo treinado.
        X_test_scaled (pd.DataFrame): Atributos de teste normalizados.
        y_test (pd.Series): Alvo de teste original.
        y_test_scaled (np.array): Alvo de teste normalizado.
        y_scaler (StandardScaler): O scaler usado para y.

    Returns:
        tuple: (y_pred_original, metrics_dict)
    """
    print("\nAvaliando o Modelo...")
    y_pred_scaled = model.predict(X_test_scaled)

    # --- Métricas Normalizadas ---
    mse_normalized = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse_normalized = np.sqrt(mse_normalized)

    # --- Métricas na Escala Original ---
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    mse_original = mean_squared_error(y_test, y_pred_original)
    rmse_original = np.sqrt(mse_original)
    mape = mean_absolute_percentage_error(y_test, y_pred_original)

    metrics = {
        'MSE_Original': mse_original,
        'RMSE_Original': rmse_original,
        'MAPE': mape,
        'MSE_Normalized': mse_normalized,
        'RMSE_Normalized': rmse_normalized
    }

    print("--- Resultados da Avaliação ---")
    print("   --- Escala Normalizada ---")
    print(f"      - EQM (MSE):  {metrics['MSE_Normalized']:.4f}")
    print(f"      - REQM (RMSE): {metrics['RMSE_Normalized']:.4f}")
    print("\n   --- Escala Original ---")
    print(f"      - EQM (MSE):  {metrics['MSE_Original']:,.2f}")
    print(f"      - REQM (RMSE): {metrics['RMSE_Original']:,.2f}")
    print(f"      - MAPE:       {metrics['MAPE']:.2f}%")

    return y_pred_original, metrics
# --- Funções de Visualização ---

def plot_predictions(y_train, y_test, y_pred, target_variable):
    """Plota os valores reais vs. previstos."""
    print("\nGerando Gráfico: Real vs. Previsto...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 7))
    plt.plot(y_train.index, y_train, label='Real (Treino)', marker='.', linestyle='-', color='gray', alpha=0.7)
    plt.plot(y_test.index, y_test, label='Real (Teste)', marker='o', linestyle='-', color='blue', markersize=7)
    plt.plot(y_test.index, y_pred, label='Previsto (Teste)', marker='x', linestyle='--', color='red', markersize=7)
    plt.title(f'Previsão de {target_variable.replace("_", " ").title()} com Lasso', fontsize=16)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel(f'{target_variable.replace("_", " ").title()} (Escala Original)', fontsize=12)
    plt.legend(fontsize=11)
    max_val = max(y_train.max(), y_test.max())
    if max_val >= 1e12: plt.ticklabel_format(style='sci', axis='y', scilimits=(12,12))
    elif max_val >= 1e9: plt.ticklabel_format(style='sci', axis='y', scilimits=(9,9))
    all_indices = y_train.index.union(y_test.index)
    plt.xticks(np.arange(min(all_indices), max(all_indices)+2, 2))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_coefficients(model, feature_names):
    """Plota os coeficientes do modelo Lasso."""
    print("\nGerando Gráfico: Importância dos Atributos...")
    coefficients = pd.DataFrame({
        'Atributo': feature_names,
        'Coeficiente_Padronizado': model.coef_
    })
    plot_features = coefficients[coefficients['Coeficiente_Padronizado'] != 0].copy()
    plot_features = plot_features.sort_values(by='Coeficiente_Padronizado')

    if not plot_features.empty:
        plt.figure(figsize=(10, max(6, len(plot_features) * 0.4)))
        plt.barh(plot_features['Atributo'], plot_features['Coeficiente_Padronizado'],
                 color=np.where(plot_features['Coeficiente_Padronizado'] > 0, 'g', 'r'))
        plt.title('Importância dos Atributos (Coeficientes Lasso Padronizados)', fontsize=16)
        plt.xlabel('Valor do Coeficiente Padronizado', fontsize=12)
        plt.ylabel('Atributo', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        plt.show()
    else:
        print("   Nenhum atributo selecionado para plotar.")

def display_coefficients(model, feature_names):
    """Exibe os coeficientes selecionados em formato de tabela."""
    print("\nAnalisando os Coeficientes do Modelo...")
    coefficients = pd.DataFrame({
        'Atributo': feature_names,
        'Coeficiente_Padronizado': model.coef_
    })
    selected_features = coefficients[coefficients['Coeficiente_Padronizado'] != 0].copy()
    selected_features['Coef_Abs'] = selected_features['Coeficiente_Padronizado'].abs()
    selected_features = selected_features.sort_values(by='Coef_Abs', ascending=False).drop('Coef_Abs', axis=1)
    selected_features['Coeficiente_Padronizado'] = selected_features['Coeficiente_Padronizado'].apply(lambda x: f"{x:,.4f}")

    print("\n--- Atributos Selecionados pelo Lasso ---")
    if selected_features.empty:
        print("   Nenhum atributo foi selecionado.")
    else:
        print(selected_features.to_string(index=False))

# --- Bloco Opcional para Teste Direto ---

def main_pipeline_test():
    """Função de teste para rodar o script diretamente."""
    # --- Configurações de Teste ---
    FILE_PATH = 'data_yearly.csv' # Assume que o arquivo está no mesmo dir
    TARGET_VARIABLE = 'gdp_current_usd'
    TRAIN_LIMIT_YEAR = 2018
    LASSO_PARAMS = {'cv': 5, 'max_iter': 100000, 'n_jobs': -1, 'tol': 0.001, 'random_state': 42}

    print("="*60)
    print("Iniciando Teste do Pipeline do Módulo")
    print("="*60 + "\n")

    data = load_and_prepare_data(FILE_PATH)
    if data is None: return

    X, y = engineer_features(data, TARGET_VARIABLE)
    if X is None: return

    split_result = split_and_scale_data(X, y, TRAIN_LIMIT_YEAR)
    if split_result is None: return

    (X_train_s, X_test_s, y_train_s, _,
     y_train, y_test, _, y_scaler) = split_result

    model = train_lasso_model(X_train_s, y_train_s, **LASSO_PARAMS)
    y_pred, metrics = evaluate_model(model, X_test_s, y_test, y_scaler)
    display_coefficients(model, X.columns)
    plot_predictions(y_train, y_test, y_pred, TARGET_VARIABLE)
    plot_coefficients(model, X.columns)

    print("\n" + "="*60)
    print("Teste do Módulo Concluído!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main_pipeline_test()