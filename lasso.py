# -*- coding: utf-8 -*-
"""
meu_modelo_lasso.py

Módulo Python simplificado para treinar, avaliar e interpretar
um modelo Lasso para previsão de séries temporais.
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
sns.set_style("whitegrid")

# --- Funções Auxiliares ---
def mean_absolute_percentage_error(y_true, y_pred):
    """Calcula o Erro Percentual Absoluto Médio (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.inf

# --- Funções do Pipeline ---

def load_and_prepare_data(file_path):
    """
    Carrega dados de um CSV, assumindo que ele tem uma coluna 'year' para ser o índice.
    """
    try:
        return pd.read_csv(file_path, index_col='year')
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
        return None

def engineer_features(df, target_variable):
    """
    Cria features de lag (1 ano) e termos quadráticos para todas as colunas.
    """
    if target_variable not in df.columns:
        print(f"Erro: Variável alvo '{target_variable}' não encontrada.")
        return None, None

    df_features = pd.DataFrame(index=df.index)
    for col in df.columns:
        df_features[f'{col}_lag1'] = df[col].shift(1)
        df_features[f'{col}_lag1_sq'] = df_features[f'{col}_lag1'] ** 2

    df_features[target_variable] = df[target_variable]
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.dropna(inplace=True)

    X = df_features.drop(columns=[target_variable])
    y = df_features[target_variable]
    return X, y

def split_and_scale_data(X, y, train_limit_year):
    """
    Divide os dados em treino/teste e os padroniza (StandardScaler).
    """
    X_train, X_test = X[X.index <= train_limit_year], X[X.index > train_limit_year]
    y_train, y_test = y[y.index <= train_limit_year], y[y.index > train_limit_year]

    x_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(x_scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(x_scaler.transform(X_test), columns=X.columns, index=X_test.index)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train, y_test, x_scaler, y_scaler

def train_lasso_model(X_train_scaled, y_train_scaled, **kwargs):
    """
    Treina um modelo LassoCV e reporta o melhor alpha encontrado.
    """
    print("\nTreinando o Modelo LassoCV...")
    lasso_cv_model = LassoCV(**kwargs).fit(X_train_scaled, y_train_scaled)
    print(f"Treinamento concluído. Melhor Alpha (α) encontrado: {lasso_cv_model.alpha_:.6f}")
    return lasso_cv_model


def evaluate_model(model, X_test_scaled, y_test, y_test_scaled, y_scaler):
    """
    Realiza previsões e calcula métricas de avaliação, incluindo
    as métricas na escala padronizada e original.

    Args:
        model: O modelo treinado.
        X_test_scaled (pd.DataFrame): Atributos de teste padronizados.
        y_test (pd.Series): Alvo de teste original.
        y_test_scaled (np.array): Alvo de teste padronizado.
        y_scaler (StandardScaler): O scaler usado para y.

    Returns:
        tuple: (y_pred_original, metrics_dict)
    """
    print("\nAvaliando o Modelo...")
    y_pred_scaled = model.predict(X_test_scaled)

    # --- Métricas Padronizadas ---
    mse_standardized = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse_standardized = np.sqrt(mse_standardized)

    # --- Métricas na Escala Original ---
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    mse_original = mean_squared_error(y_test, y_pred_original)
    rmse_original = np.sqrt(mse_original)
    mape = mean_absolute_percentage_error(y_test, y_pred_original)

    metrics = {
        'MSE_Original': mse_original,
        'RMSE_Original': rmse_original,
        'MAPE': mape,
        'MSE_Standardized': mse_standardized,
        'RMSE_Standardized': rmse_standardized
    }

    print("--- Resultados da Avaliação ---")
    print("   --- Escala Padronizada (Média 0, Desvio Padrão 1) ---")
    print(f"      - EQM (MSE):  {metrics['MSE_Standardized']:.4f}")
    print(f"      - REQM (RMSE): {metrics['RMSE_Standardized']:.4f}")
    print("\n   --- Escala Original ---")
    print(f"      - EQM (MSE):  {metrics['MSE_Original']:,.2f}")
    print(f"      - REQM (RMSE): {metrics['RMSE_Original']:,.2f}")
    print(f"      - MAPE:       {metrics['MAPE']:.2f}%")

    return y_pred_original, metrics


def display_coefficients(model, feature_names):
    """
    Exibe os atributos selecionados pelo Lasso e seus coeficientes.
    """
    print("\n--- Atributos Selecionados pelo Modelo (Coeficientes) ---")
    coeffs = pd.Series(model.coef_, index=feature_names).sort_values(ascending=False)
    selected = coeffs[coeffs != 0]

    if selected.empty:
        print("   Nenhum atributo foi selecionado pelo modelo.")
    else:
        print(selected.to_string(float_format='{:,.4f}'.format))

def explain_coefficients(model, feature_names, target_variable_name, x_scaler, y_scaler):
    """
    Fornece uma explicação dos coeficientes focada na escala padronizada,
    com o contexto da escala original da variável alvo fornecido separadamente.
    """

    print("AVISO: O modelo mostra associação, não causalidade. A interpretação descreve padrões nos dados.")

    # Extrai o desvio padrão da variável alvo para dar contexto
    y_std = y_scaler.scale_[0]

    # Seleciona os coeficientes relevantes
    coefficients = pd.Series(model.coef_, index=feature_names)
    selected_coeffs = coefficients[coefficients != 0].sort_values(key=abs, ascending=False)

    if selected_coeffs.empty:
        print("\nNenhum atributo foi selecionado pelo modelo.")
        return

    # Apresenta o "fator de conversão" da variável alvo, separadamente
    print(f"\nPara converter o impacto para a escala original, utilize a seguinte referência:")
    print(f"  -> 1 desvio padrão (DP) de '{target_variable_name}' equivale a {y_std:,.2f} unidades.")
    print("-" * 60)

    print("Associações encontradas pelo modelo (na escala padronizada):\n")
    for feature, coef in selected_coeffs.items():
        # A impressão segue exatamente o template que você pediu
        print(f"🔹 Atributo: {feature}")
        print(f"   - Uma variação de 1 desvio padrão neste atributo está associada a uma mudança média de")
        print(f"     {coef:+.4f} desvios padrão (DPs) na variável '{target_variable_name}'.\n")



def plot_predictions(y_train, y_test, y_pred, target_variable):
    """Plota os valores reais (treino e teste) vs. os valores previstos (teste)."""
    plt.figure(figsize=(15, 7))
    plt.plot(y_train.index, y_train, label='Real (Treino)', marker='.', linestyle='-', color='gray', alpha=0.7)
    plt.plot(y_test.index, y_test, label='Real (Teste)', marker='o', linestyle='-', color='blue')
    plt.plot(y_test.index, y_pred, label='Previsto (Teste)', marker='x', linestyle='--', color='red')
    plt.title(f'Previsão de {target_variable.replace("_", " ").title()} com Lasso')
    plt.xlabel('Ano')
    plt.ylabel(target_variable.replace("_", " ").title())
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_coefficients(model, feature_names):
    """Plota um gráfico de barras com os coeficientes dos atributos selecionados."""
    coeffs = pd.Series(model.coef_, index=feature_names)
    selected = coeffs[coeffs != 0].sort_values()

    if not selected.empty:
        plt.figure(figsize=(10, len(selected) * 0.4))
        selected.plot(kind='barh', color=np.where(selected > 0, 'g', 'r'))
        plt.title('Importância dos Atributos (Coeficientes Lasso)')
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel('Valor do Coeficiente Padronizado')
        plt.tight_layout()
        plt.show()