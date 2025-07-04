import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, Select, Button, Div, Slider,
    DataTable, TableColumn, StringFormatter, NumberFormatter,
    HoverTool, Panel, Tabs, TabPanel, Range1d
)
from bokeh.layouts import column, row, layout
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import warnings
import tempfile
import os
import traceback
import time

warnings.filterwarnings('ignore')

class ForecastDash:
    def __init__(self, years: List[str]):
        self.years = years
        self.data_manager = None
        self.current_indicator = ""
        self.current_country = ""
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.forecast_data = {}
        
        self.model_data = None
        
        self.hist_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.test_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.pred_test_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.forecast_source = ColumnDataSource(data=dict(x=[], y=[]))
        
        self._build_ui()
        
    def _build_ui(self):
        self.header_div = Div(
            text="""
            <div style='background: linear-gradient(135deg, #324376 0%, #1e2a4a 100%); 
                        padding: 30px; 
                        border-radius: 10px; 
                        margin-bottom: 30px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='color: white; margin: 0; font-weight: 300; font-size: 28px;'>
                    Machine Learning Forecast Analysis
                </h2>
                <p style='color: #b8c5e0; margin: 10px 0 0 0; font-size: 16px;'>
                    Advanced time series prediction using Lasso regression with multiple economic indicators
                </p>
            </div>
            """,
            width=1100
        )
        
        self.controls_header = Div(
            text="""
            <h3 style='margin: 0 0 20px 0; 
                      color: #324376; 
                      font-weight: 500; 
                      font-size: 20px;
                      border-bottom: 2px solid #324376;
                      padding-bottom: 10px;'>
                Model Configuration
            </h3>
            """,
            width=250
        )
        
        self.country_select = Select(
            title="Select Country",
            options=[("", "Choose a country...")],
            value="",
            width=250,
            height=40,
            styles={
                "margin-bottom": "15px",
                "font-size": "14px"
            }
        )
        
        self.train_split_slider = Slider(
            title="Training Data Until",
            start=2015,
            end=2023,
            value=2018,
            step=1,
            width=250,
            height=60,
            styles={"margin-bottom": "15px"}
        )
        
        self.target_year_slider = Slider(
            title="Forecast Target Year",
            start=2025,
            end=2030,
            value=2025,
            step=1,
            width=250,
            height=60,
            styles={"margin-bottom": "20px"}
        )
        
        self.forecast_button = Button(
            label="Run Forecast Analysis",
            button_type="primary",
            width=250,
            height=50,
            styles={
                "background": "#324376",
                "border": "none",
                "font-size": "16px",
                "font-weight": "600",
                "margin-bottom": "10px",
                "border-radius": "5px"
            }
        )
        
        self.clear_button = Button(
            label="Clear Results",
            button_type="default",
            width=250,
            height=40,
            styles={
                "background": "#f5f5f5",
                "border": "1px solid #ddd",
                "color": "#666",
                "font-size": "14px",
                "font-weight": "500",
                "border-radius": "5px"
            }
        )
        
        self.info_div = Div(
            text="""
            <div style='background: #f8f9fa; 
                       padding: 15px; 
                       border-radius: 5px; 
                       margin-top: 20px;
                       border-left: 4px solid #324376;'>
                <h4 style='margin: 0 0 10px 0; color: #324376; font-size: 14px;'>
                    ℹ️ How it works
                </h4>
                <p style='margin: 0; font-size: 12px; color: #666; line-height: 1.6;'>
                    The model uses historical data from multiple economic indicators to predict future values. 
                    Lasso regression automatically selects the most relevant features while preventing overfitting.
                </p>
            </div>
            """,
            width=250
        )
        
        self.results_header = Div(
            text="""
            <h3 style='margin: 0 0 20px 0; 
                      color: #324376; 
                      font-weight: 500; 
                      font-size: 20px;
                      border-bottom: 2px solid #324376;
                      padding-bottom: 10px;'>
                Forecast Results
            </h3>
            """,
            width=800,
            visible=False
        )
        
        self.results_div = Div(
            text="""
            <div style='text-align:center; 
                       padding: 60px 20px; 
                       background: #fafafa; 
                       border: 2px dashed #ddd; 
                       border-radius: 10px;'>
                <p style='margin:0; font-size:18px; color:#999;'>
                    Select a country and click "Run Forecast Analysis" to see results
                </p>
            </div>
            """,
            width=800,
            height=150,
            styles={
                "margin-bottom": "30px"
            }
        )
        
        self.fig = self._build_figure()
        
        self.coef_header = Div(
            text="""
            <h3 style='margin: 30px 0 20px 0; 
                      color: #324376; 
                      font-weight: 500; 
                      font-size: 20px;
                      border-bottom: 2px solid #324376;
                      padding-bottom: 10px;'>
                Feature Importance Analysis
            </h3>
            """,
            width=800,
            visible=False
        )
        
        self.coef_fig = self._build_coefficient_figure()
        
        self.coef_explanation_div = Div(
            text="",
            width=800,
            height=400,
            styles={
                "padding": "0",
                "background": "transparent",
                "margin-top": "30px",
                "overflow-y": "auto"
            }
        )
        
        self.metrics_div = Div(
            text="",
            width=800,
            height=200,
            styles={
                "margin-bottom": "50px"
            }
        )
        
        self.metrics_spacer = Div(
            text="",
            width=800,
            height=60
        )
        
        self.forecast_button.on_click(self._run_forecast)
        self.clear_button.on_click(self._clear_analysis)
        
    def _build_figure(self):
        p = figure(
            title="",
            toolbar_location="right",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
            x_axis_label="Year",
            y_axis_label="Value",
            background_fill_color="#ffffff",
            border_fill_color="#ffffff"
        )
        
        p.title.text_font_size = "16pt"
        p.title.text_color = "#324376"
        p.title.text_font_style = "normal"
        p.title.align = "center"
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.axis_label_text_font_style = "bold"
        p.yaxis.axis_label_text_font_style = "bold"
        p.grid.grid_line_alpha = 0.15
        p.grid.grid_line_dash = [6, 4]
        
        p.aspect_ratio = None
        p.match_aspect = False
        
        p.outline_line_width = 1
        p.outline_line_alpha = 0.3
        p.outline_line_color = "#666"
        
        hover = HoverTool(
            tooltips=[
                ("Year", "@x"),
                ("Value", "@y{0,0.00}")
            ],
            mode='mouse'
        )
        p.add_tools(hover)
        
        return p
    
    def _build_coefficient_figure(self):
        p = figure(
            y_range=[],
            title="",
            toolbar_location=None,
            width=800,
            height=350,
            x_axis_label="Standardized Coefficient Value",
            min_border_left=20
        )
        
        p.title.text_font_size = "14pt"
        p.title.text_color = "#324376"
        p.xaxis.axis_label_text_font_size = "12pt"
        p.xaxis.axis_label_text_font_style = "bold"
        p.yaxis.axis_label_text_font_size = "11pt"
        p.grid.grid_line_alpha = 0.15
        p.background_fill_color = "#fafafa"
        
        self.coef_source = ColumnDataSource(data=dict(
            features=[],
            coefficients=[],
            colors=[],
            abs_coef=[]
        ))
        
        p.hbar(y='features', right='coefficients', height=0.7, 
               color='colors', source=self.coef_source, alpha=0.8,
               hover_fill_alpha=1.0, hover_line_color="black")
        
        p.line([0, 0], [-1, 100], line_color='#333', line_width=2, 
               line_dash='solid', alpha=0.5)
        
        hover = HoverTool(tooltips=[
            ("Feature", "@features"),
            ("Coefficient", "@coefficients{0.0000}"),
            ("Impact", "@abs_coef{0.0000}")
        ])
        p.add_tools(hover)
        
        return p
    
    def get_layout(self):
        controls_panel = column(
            self.controls_header,
            self.country_select,
            Div(text="", height=10),
            self.train_split_slider,
            self.target_year_slider,
            Div(text="", height=10),
            self.forecast_button,
            self.clear_button,
            self.info_div,
            width=280,
            styles={
                "padding": "15px",
                "background": "#ffffff",
                "border": "1px solid #e0e0e0",
                "border-radius": "10px",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.05)"
            }
        )
        
        self.chart_panel = column(
            self.fig,
            width=800,
            styles={
                "background": "#ffffff",
                "border": "1px solid #e0e0e0",
                "border-radius": "10px",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.05)",
                "padding": "20px",
                "margin-bottom": "30px"
            }
        )
        
        coef_panel = column(
            self.coef_header,
            self.coef_fig,
            self.coef_explanation_div,
            width=800,
            styles={
                "background": "#ffffff",
                "border": "1px solid #e0e0e0",
                "border-radius": "10px",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.05)",
                "padding": "20px",
                "margin-top": "30px"
            }
        )
        
        results_content = column(
            self.results_header,
            self.results_div,
            self.metrics_div,
            self.chart_panel,
            coef_panel,
            width=800,
            styles={
                "padding": "0",
                "background": "transparent"
            }
        )
        
        results_panel = column(
            results_content,
            width=800,
            styles={
                "padding": "20px",
                "background": "#ffffff",
                "border": "1px solid #e0e0e0",
                "border-radius": "10px",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.05)"
            }
        )
        
        main_content = row(
            controls_panel,
            Div(text="", width=20),
            results_panel,
            styles={
                "margin-top": "0"
            }
        )
        
        complete_layout = column(
            self.header_div,
            main_content,
            width=1100,
            styles={
                "padding": "20px",
                "background": "#f5f6fa"
            }
        )
        
        return complete_layout
    
    def update(self, data_manager, indicator: str):
        self.data_manager = data_manager
        self.current_indicator = indicator
        
        countries = data_manager.get_filtered_countries(False)
        options = [("", "Select Country")]
        options.extend([(c, c.title() if isinstance(c, str) else c) for c in countries])
        self.country_select.options = options
        
        if self.years:
            years_int = [int(y) for y in self.years]
            self.train_split_slider.start = min(years_int)
            self.train_split_slider.end = max(years_int) - 1
            self.train_split_slider.value = min(2018, max(years_int) - 1)
            
            self.target_year_slider.start = max(years_int) + 1
            self.target_year_slider.end = max(years_int) + 10
            self.target_year_slider.value = max(years_int) + 1
        
        self._clear_results()
        
    def _clear_results(self):
        self.hist_source.data = dict(x=[], y=[])
        self.test_source.data = dict(x=[], y=[])
        self.pred_test_source.data = dict(x=[], y=[])
        self.forecast_source.data = dict(x=[], y=[])
        self.coef_source.data = dict(features=[], coefficients=[], colors=[], abs_coef=[])
        
        self.results_div.text = """
        <div style='text-align:center; 
                   padding: 60px 20px; 
                   background: #fafafa; 
                   border: 2px dashed #ddd; 
                   border-radius: 10px;'>
            <p style='margin:0; font-size:18px; color:#999;'>
                Select a country and click "Run Forecast Analysis" to see results
            </p>
        </div>
        """
        
        self.metrics_div.text = ""
        self.coef_explanation_div.text = ""
        self.fig.title.text = ""
        self.coef_fig.y_range.factors = []
        
        self.results_header.visible = False
        self.coef_header.visible = False
        
    def _clear_analysis(self):
        self._clear_results()
        self.country_select.value = ""
        self.current_country = ""
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.forecast_data = {}
        self.model_data = None
        
        self.results_div.text = """
        <div style='text-align:center; 
                   padding: 40px 20px; 
                   background: #e8f4f8; 
                   border: 1px solid #b8e0e8; 
                   border-radius: 10px;'>
            <p style='margin:0; font-size:16px; color:#324376;'>
                ✓ Analysis cleared. Ready for new forecast.
            </p>
        </div>
        """
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
    def _run_forecast(self):
        country = self.country_select.value
        if not country or not self.current_indicator:
            self.results_div.text = """
            <div style='text-align:center; 
                       padding: 30px 20px; 
                       background: #fff3cd; 
                       border: 1px solid #ffeaa7; 
                       border-radius: 10px;'>
                <p style='margin:0; font-size:16px; color:#856404;'>
                    ⚠️ Please select a country first
                </p>
            </div>
            """
            return
        
        self.results_div.text = """
        <div style='text-align:center; 
                   padding: 40px 20px; 
                   background: #e3f2fd; 
                   border: 1px solid #90caf9; 
                   border-radius: 10px;'>
            <p style='margin:0; font-size:18px; color:#1976d2;'>
                🔄 Processing forecast analysis...
            </p>
            <p style='margin:10px 0 0 0; font-size:14px; color:#64b5f6;'>
                This may take a few moments
            </p>
        </div>
        """
        
        self.results_header.visible = True
        self.coef_header.visible = True
        
        try:
            country_df = self.data_manager.get_country_dataframe(country.lower())
            
            if country_df is None:
                years, values = self.data_manager.get_country_data(country, self.current_indicator)
                
                if len(years) < 10:
                    self.results_div.text = """
                    <div style='color:#d32f2f; text-align:center; padding:30px;'>
                        <h3 style='margin:0; font-size:18px;'>Insufficient Data</h3>
                        <p style='margin:10px 0; font-size:14px;'>Need at least 10 years of data for reliable forecast.</p>
                    </div>
                    """
                    return
                
                df = pd.DataFrame({
                    'year': [int(y) for y in years],
                    self.current_indicator: values
                }).set_index('year')
            else:
                df = country_df.copy()
                if 'year' in df.columns:
                    df = df.set_index('year')
                elif df.index.name != 'year':
                    df.index.name = 'year'
                
                if self.current_indicator not in df.columns:
                    self.results_div.text = f"""
                    <div style='color:#d32f2f; text-align:center; padding:20px;'>
                        <h4 style='margin:0;'>❌ Indicator Not Found</h4>
                        <p style='margin:5px 0;'>'{self.current_indicator}' not found in country data.</p>
                    </div>
                    """
                    return
            
            df_features = pd.DataFrame(index=df.index)
            
            for col in df.columns:
                if col != 'year':
                    df_features[f'{col}_lag1'] = df[col].shift(1)
            
            df_features[self.current_indicator] = df[self.current_indicator]
            
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            df_features = df_features.dropna()
            
            if len(df_features) < 5:
                self.results_div.text = """
                <div style='color:#d32f2f; text-align:center; padding:20px;'>
                    <h4 style='margin:0;'>❌ Insufficient Data</h4>
                    <p style='margin:5px 0;'>Not enough data points after feature engineering.</p>
                </div>
                """
                return
            
            X = df_features.drop(columns=[self.current_indicator])
            y = df_features[self.current_indicator]
            
            train_limit = int(self.train_split_slider.value)
            X_train = X[X.index <= train_limit]
            X_test = X[X.index > train_limit]
            y_train = y[y.index <= train_limit]
            y_test = y[y.index > train_limit]
            
            if len(X_train) < 5:
                self.results_div.text = """
                <div style='color:#d32f2f; text-align:center; padding:20px;'>
                    <h4 style='margin:0;'>❌ Insufficient Training Data</h4>
                    <p style='margin:5px 0;'>Need at least 5 years of training data.</p>
                </div>
                """
                return
            
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            X_train_scaled = x_scaler.fit_transform(X_train)
            X_test_scaled = x_scaler.transform(X_test) if len(X_test) > 0 else np.array([])
            
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel() if len(y_test) > 0 else np.array([])
            
            start_time = time.time()
            
            LASSO_PARAMS = {'cv': 5, 'max_iter': 100000, 'tol': 0.001}
            
            if X_train.shape[1] <= 5:
                LASSO_PARAMS['n_alphas'] = 10000
            elif X_train.shape[1] <= 10:
                LASSO_PARAMS['n_alphas'] = 5000
            else:
                LASSO_PARAMS['n_alphas'] = 1000
            
            LASSO_PARAMS['n_jobs'] = 1
            
            model = LassoCV(**LASSO_PARAMS).fit(X_train_scaled, y_train_scaled)
            
            train_time = time.time() - start_time
            
            y_pred_test = None
            metrics = {}
            
            if len(X_test) > 0:
                y_pred_test_scaled = model.predict(X_test_scaled)
                y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
                
                mse = mean_squared_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                mape = self._calculate_mape(y_test.values, y_pred_test)
                
                mse_norm = mean_squared_error(y_test_scaled, y_pred_test_scaled)
                rmse_norm = np.sqrt(mse_norm)
                
                metrics = {
                    'MSE': mse, 'RMSE': rmse, 'MAPE': mape,
                    'MSE_norm': mse_norm, 'RMSE_norm': rmse_norm
                }
            
            forecast_year = int(self.target_year_slider.value)
            
            last_year = df.index.max()
            last_year_data = df.loc[last_year]
            
            future_features = {}
            for feature in X.columns:
                if feature.endswith('_lag1'):
                    orig_col = feature[:-5]
                    if orig_col in last_year_data.index:
                        future_features[feature] = float(last_year_data[orig_col])
                    else:
                        future_features[feature] = 0.0
                else:
                    future_features[feature] = 0.0
            
            future_input_df = pd.DataFrame([future_features], columns=X.columns)
            future_input_df = future_input_df.fillna(0)
            
            future_input_scaled = x_scaler.transform(future_input_df)
            prediction_scaled = model.predict(future_input_scaled)
            forecast_value = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()[0]
            
            self.model_data = {
                'model': model,
                'x_scaler': x_scaler,
                'y_scaler': y_scaler,
                'X': X,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred_test': y_pred_test,
                'metrics': metrics,
                'forecast_value': forecast_value,
                'forecast_year': forecast_year,
                'feature_names': X.columns
            }
            
            self.current_country = country
            
            self._update_visualizations()
            
            self._update_coefficient_explanations()
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            
            self.results_div.text = f"""
            <div style='color:#d32f2f; 
                       text-align:center; 
                       padding:30px 20px; 
                       background: #ffebee; 
                       border: 1px solid #ffcdd2; 
                       border-radius: 10px;'>
                <h4 style='margin:0;'>❌ Analysis Error</h4>
                <p style='margin:10px 0; font-size:14px;'>{str(e)}</p>
            </div>
            """
    
    def _update_coefficient_explanations(self):
        if not self.model_data:
            return
        
        model = self.model_data['model']
        x_scaler = self.model_data['x_scaler']
        y_scaler = self.model_data['y_scaler']
        feature_names = self.model_data['feature_names']
        
        coefficients = pd.Series(model.coef_, index=feature_names)
        selected_coeffs = coefficients[coefficients != 0].sort_values(key=abs, ascending=False)
        
        if selected_coeffs.empty:
            self.coef_explanation_div.text = """
            <div style='text-align:center; padding: 40px; background: #f5f5f5; border-radius: 10px;'>
                <p style='color: #666; font-size: 16px;'>No features were selected by the model.</p>
            </div>
            """
            return
        
        features = selected_coeffs.index.tolist()
        coef_values = selected_coeffs.values.tolist()
        colors = ['#2E8B57' if c > 0 else '#FF6B6B' for c in coef_values]
        abs_coef = [abs(c) for c in coef_values]
        
        self.coef_source.data = dict(
            features=features[::-1],
            coefficients=coef_values[::-1],
            colors=colors[::-1],
            abs_coef=abs_coef[::-1]
        )
        
        self.coef_fig.y_range.factors = features[::-1]
        
        y_std = y_scaler.scale_[0]
        
        explanation_html = """
        <div style='background: #fff; 
                   padding: 30px; 
                   border-radius: 10px; 
                   border: 1px solid #e0e0e0;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            
            <div style='background: #fff3cd; 
                       padding: 15px 20px; 
                       border-radius: 5px; 
                       margin-bottom: 25px;
                       border-left: 4px solid #ffc107;'>
                <h4 style='color: #856404; margin: 0 0 10px 0; font-size: 16px;'>
                    ⚠️ Important Note
                </h4>
                <p style='margin: 0; color: #856404; font-size: 14px; line-height: 1.5;'>
                    The model shows association, not causation. The interpretation describes patterns in the data.
                </p>
            </div>
            
            <div style='background: #e8f4f8; 
                       padding: 15px 20px; 
                       border-radius: 5px; 
                       margin-bottom: 25px;
                       border-left: 4px solid #2196F3;'>
                <p style='margin: 0; font-size: 14px; line-height: 1.6;'>
                    To convert the impact to the original scale:<br>
                    <strong style='color: #1976d2; font-size: 16px;'>
                        → 1 standard deviation (SD) of '{indicator}' equals {y_std:,.2f} units
                    </strong>
                </p>
            </div>
            
            <h4 style='color: #324376; margin-bottom: 20px; font-size: 18px;'>
                Model Associations (Standardized Scale):
            </h4>
            
            <div style='display: grid; gap: 15px;'>
        """.format(indicator=self.current_indicator, y_std=y_std)
        
        for i, (feature, coef) in enumerate(selected_coeffs.items()):
            feature_display = feature.replace('_', ' ').title()
            color = '#2E8B57' if coef > 0 else '#FF6B6B'
            sign = '+' if coef > 0 else ''
            
            explanation_html += f"""
            <div style='padding: 20px; 
                       background: #fafafa; 
                       border-left: 4px solid {color};
                       border-radius: 5px;
                       transition: all 0.3s ease;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <strong style='color: #324376; font-size: 16px;'>
                            {i+1}. {feature_display}
                        </strong>
                        <p style='margin: 10px 0 0 0; color: #666; font-size: 14px; line-height: 1.6;'>
                            A 1 SD increase in this attribute is associated with a 
                            <strong style='color: {color}; font-size: 16px;'>{sign}{coef:.4f} SD</strong> 
                            change in {self.current_indicator.replace('_', ' ')}
                        </p>
                    </div>
                    <div style='text-align: center; padding: 10px;'>
                        <div style='font-size: 24px; color: {color}; font-weight: bold;'>
                            {sign}{coef:.3f}
                        </div>
                        <div style='font-size: 12px; color: #999;'>
                            coefficient
                        </div>
                    </div>
                </div>
            </div>
            """
        
        explanation_html += """
            </div>
        </div>
        """
        
        self.coef_explanation_div.text = explanation_html
            
    def _update_visualizations(self):
        if not self.model_data:
            return
            
        y_train = self.model_data['y_train']
        y_test = self.model_data['y_test']
        y_pred_test = self.model_data['y_pred_test']
        forecast_value = self.model_data['forecast_value']
        forecast_year = self.model_data['forecast_year']
        metrics = self.model_data['metrics']
        model = self.model_data['model']
        
        new_fig = figure(
            title="",
            toolbar_location="right",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800,
            height=500,
            x_axis_label="Year",
            y_axis_label="Value",
            background_fill_color="#ffffff",
            border_fill_color="#ffffff"
        )
        
        new_fig.title.text_font_size = "16pt"
        new_fig.title.text_color = "#324376"
        new_fig.title.text_font_style = "normal"
        new_fig.title.align = "center"
        new_fig.xaxis.axis_label_text_font_size = "12pt"
        new_fig.yaxis.axis_label_text_font_size = "12pt"
        new_fig.xaxis.axis_label_text_font_style = "bold"
        new_fig.yaxis.axis_label_text_font_style = "bold"
        new_fig.grid.grid_line_alpha = 0.15
        new_fig.grid.grid_line_dash = [6, 4]
        
        new_fig.outline_line_width = 1
        new_fig.outline_line_alpha = 0.3
        new_fig.outline_line_color = "#666"
        
        hover = HoverTool(
            tooltips=[
                ("Year", "@x"),
                ("Value", "@y{0,0.00}")
            ],
            mode='mouse'
        )
        new_fig.add_tools(hover)
        
        years_train = [int(y) for y in y_train.index]
        years_test = [int(y) for y in y_test.index] if len(y_test) > 0 else []
        
        all_years = sorted(set(years_train + years_test + [forecast_year]))
        min_year, max_year = min(all_years), max(all_years)
        
        all_values = list(y_train.values)
        if len(y_test) > 0:
            all_values.extend(y_test.values)
            if y_pred_test is not None:
                all_values.extend(y_pred_test)
        all_values.append(forecast_value)
        
        min_value = min(all_values) * 0.9 if min(all_values) > 0 else min(all_values) * 1.1
        max_value = max(all_values) * 1.1
        
        new_fig.x_range = Range1d(min_year - 0.5, max_year + 0.5)
        new_fig.y_range = Range1d(min_value, max_value)
        
        self.hist_source.data = {
            'x': years_train,
            'y': y_train.values.tolist()
        }
        
        if len(y_test) > 0 and y_pred_test is not None:
            self.test_source.data = {
                'x': years_test,
                'y': y_test.values.tolist()
            }
            self.pred_test_source.data = {
                'x': years_test,
                'y': y_pred_test.tolist()
            }
        else:
            self.test_source.data = {'x': [], 'y': []}
            self.pred_test_source.data = {'x': [], 'y': []}
        
        self.forecast_source.data = {
            'x': [forecast_year],
            'y': [forecast_value]
        }
        
        new_fig.line(
            x='x', y='y', source=self.hist_source, 
            color='#324376', 
            line_width=3, 
            legend_label="Historical Data",
            alpha=0.9
        )
        new_fig.circle(
            x='x', y='y', source=self.hist_source, 
            size=8, 
            color='#324376', 
            alpha=0.9
        )
        
        if len(y_test) > 0 and y_pred_test is not None:
            new_fig.line(
                x='x', y='y', source=self.test_source, 
                color='#2E8B57', 
                line_width=3, 
                legend_label="Actual Test Data",
                alpha=0.9
            )
            new_fig.circle(
                x='x', y='y', source=self.test_source, 
                size=8, 
                color='#2E8B57', 
                alpha=0.9
            )
            
            new_fig.line(
                x='x', y='y', source=self.pred_test_source, 
                color='#FF6B6B', 
                line_width=3, 
                line_dash='dashed', 
                legend_label="Model Predictions",
                alpha=0.9
            )
            new_fig.scatter(
                x='x', y='y', source=self.pred_test_source, 
                size=10, 
                color='#FF6B6B', 
                marker='x',
                line_width=2,
                alpha=0.9
            )
        
        new_fig.scatter(
            x='x', y='y', source=self.forecast_source, 
            size=20, 
            color='#FFD700', 
            line_color='#FF8C00',
            line_width=3, 
            marker='star',
            alpha=1.0,
            legend_label="Future Forecast"
        )
        
        indicator_name = self._format_indicator_name(self.current_indicator)
        new_fig.title.text = f"{indicator_name} Forecast Analysis - {self.current_country.title()}"
        
        new_fig.legend.location = "top_right"
        new_fig.legend.orientation = "horizontal"
        new_fig.legend.click_policy = "hide"
        new_fig.legend.background_fill_alpha = 0.7
        
        self.fig = new_fig
        
        self.chart_panel.children[0] = self.fig
        
        n_features = sum(1 for coef in model.coef_ if coef != 0)
        total_features = len(model.coef_)
        
        results_html = f"""
        <div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                   padding: 30px; 
                   border-radius: 10px;
                   box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                   margin-bottom: 20px;'>
            
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='flex: 3;'>
                    <h3 style='margin: 0; color: #1b5e20; font-size: 26px; font-weight: 600;'>
                        {forecast_year} Forecast
                    </h3>
                    <p style='margin: 10px 0 0 0; color: #2e7d32; font-size: 18px;'>
                        {indicator_name} • {self.current_country.title()}
                    </p>
                </div>
                <div style='text-align: center; flex: 2; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-size: 40px; color: #1b5e20; font-weight: bold; line-height: 1.2;'>
                        {forecast_value:,.2f}
                    </div>
                    <div style='font-size: 16px; color: #388e3c; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;'>
                        predicted value
                    </div>
                </div>
            </div>
        </div>
        
        <div style='height: 30px;'></div>
        """
        
        self.results_div.text = results_html
        
        metrics_html = f"""
        <div style='display: flex; gap: 10px; margin-bottom: 30px;'>
            <div style='background: #fff; 
                      padding: 15px; 
                      border-radius: 8px; 
                      border: 1px solid #e0e0e0;
                      text-align: center;
                      flex: 1;'>
                <div style='color: #666; font-size: 13px; margin-bottom: 5px;'>Model Alpha</div>
                <div style='color: #324376; font-size: 18px; font-weight: bold;'>{model.alpha_:.6f}</div>
            </div>
            
            <div style='background: #fff; 
                      padding: 15px; 
                      border-radius: 8px; 
                      border: 1px solid #e0e0e0;
                      text-align: center;
                      flex: 1;'>
                <div style='color: #666; font-size: 13px; margin-bottom: 5px;'>Features Used</div>
                <div style='color: #324376; font-size: 18px; font-weight: bold;'>{n_features}/{total_features}</div>
            </div>
            
            <div style='background: #fff; 
                      padding: 15px; 
                      border-radius: 8px; 
                      border: 1px solid #e0e0e0;
                      text-align: center;
                      flex: 1;'>
                <div style='color: #666; font-size: 13px; margin-bottom: 5px;'>Training Period</div>
                <div style='color: #324376; font-size: 18px; font-weight: bold;'>≤ {self.train_split_slider.value}</div>
            </div>
        </div>
        """
        
        if metrics:
            metrics_html += f"""
            <div style='background: #f5f5f5; 
                       padding: 15px; 
                       border-radius: 8px; 
                       margin-top: 20px;
                       margin-bottom: 30px;
                       border: 1px solid #e0e0e0;'>
                <h4 style='margin: 0 0 10px 0; color: #324376; font-size: 14px;'>
                    Model Performance Metrics
                </h4>
                <div style='display: flex; gap: 15px;'>
                    <div style='flex: 1;'>
                        <span style='color: #666; font-size: 12px;'>RMSE (Original)</span><br>
                        <span style='color: #2E8B57; font-size: 16px; font-weight: bold;'>{metrics['RMSE']:,.2f}</span>
                    </div>
                    <div style='flex: 1;'>
                        <span style='color: #666; font-size: 12px;'>MAPE</span><br>
                        <span style='color: #FF6B6B; font-size: 16px; font-weight: bold;'>{metrics['MAPE']:.1f}%</span>
                    </div>
                    <div style='flex: 1;'>
                        <span style='color: #666; font-size: 12px;'>RMSE (Normalized)</span><br>
                        <span style='color: #324376; font-size: 16px; font-weight: bold;'>{metrics['RMSE_norm']:.4f}</span>
                    </div>
                </div>
            </div>
            
            <div style='height: 60px;'></div>
            """
        
        self.metrics_div.text = metrics_html
    
    def _format_indicator_name(self, indicator):
        formatted = indicator.replace('_', ' ')
        words = formatted.split()
        capitalized_words = []
        for word in words:
            if len(word) > 2:
                capitalized_words.append(word.capitalize())
            else:
                if len(capitalized_words) == 0:
                    capitalized_words.append(word.capitalize())
                else:
                    capitalized_words.append(word.lower())
        return ' '.join(capitalized_words)
    
    def clear(self):
        self._clear_results()
        self.current_indicator = ""
        self.current_country = ""
        self.country_select.value = ""
        self.model_data = None