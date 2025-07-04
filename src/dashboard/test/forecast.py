import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, Select, Button, Div, Slider,
    DataTable, TableColumn, StringFormatter, NumberFormatter
)
from bokeh.layouts import column, row
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import warnings

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
        
        # UI Components
        self._build_ui()
        
    def _build_ui(self):
        """Build the UI components for the forecast panel"""
        # Country selector with clean styling
        self.country_select = Select(
            title="Select Country",
            options=[("", "Choose a country...")],
            value="",
            width=240,
            styles={"margin-bottom": "5px"}
        )
        
        # Target year slider
        self.target_year_slider = Slider(
            title="Forecast Year",
            start=2025,
            end=2030,
            value=2025,
            step=1,
            width=240,
            styles={"margin-bottom": "5px"}
        )
        
        # Training split year
        self.train_split_slider = Slider(
            title="Training Data Until",
            start=2015,
            end=2023,
            value=2018,
            step=1,
            width=240,
            styles={"margin-bottom": "5px"}
        )
        
        # Run forecast button
        self.forecast_button = Button(
            label="Run Forecast",
            button_type="primary",
            width=240,
            height=45,
            styles={"background": "#324376", "border": "none", "font-size": "14px", "font-weight": "500"}
        )
        
        # Clear analysis button
        self.clear_button = Button(
            label="Clear Analysis",
            button_type="default",
            width=240,
            height=45,
            styles={"background": "#f5f5f5", "border": "1px solid #ddd", "color": "#666", "font-size": "14px", "font-weight": "500"}
        )
        
        # Model info display with improved styling - removed from main layout
        # Will only show in results now
        
        # Results display - clean and organized
        self.results_div = Div(
            text="",
            width=1000,
            height=100,
            styles={
                "padding": "20px",
                "background": "#ffffff",
                "border": "1px solid #e0e0e0",
                "border-radius": "8px",
                "font-size": "14px",
                "box-shadow": "0 1px 3px rgba(0,0,0,0.1)"
            }
        )
        
        # Build main figure
        self.fig = self._build_figure()
        
        # Attach callbacks
        self.forecast_button.on_click(self._run_forecast)
        self.clear_button.on_click(self._clear_analysis)
        
    def _build_figure(self):
        """Build the main forecast visualization figure with clean styling"""
        p = figure(
            title="Forecast Analysis",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=1000,
            height=500,
            x_axis_label="Year",
            y_axis_label="Value",
            background_fill_color="#ffffff",
            border_fill_color="white"
        )
        
        # Improved styling
        p.title.text_font_size = "18pt"
        p.title.text_color = "#324376"
        p.title.text_font_style = "normal"
        p.title.align = "center"
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.grid.grid_line_alpha = 0.2
        p.border_fill_color = "#ffffff"
        p.background_fill_color = "#fafafa"
        
        # Historical data line
        self.hist_source = ColumnDataSource(data=dict(x=[], y=[]))
        p.line('x', 'y', source=self.hist_source, 
               line_width=3, color='#324376', legend_label="Historical Data", alpha=0.8)
        p.circle('x', 'y', source=self.hist_source, 
                 size=8, color='#324376', legend_label="Historical Data", alpha=0.8)
        
        # Test data line
        self.test_source = ColumnDataSource(data=dict(x=[], y=[]))
        p.line('x', 'y', source=self.test_source,
               line_width=3, color='#2E8B57', legend_label="Test Data", alpha=0.8)
        p.circle('x', 'y', source=self.test_source,
                 size=8, color='#2E8B57', legend_label="Test Data", alpha=0.8)
        
        # Predicted test data
        self.pred_test_source = ColumnDataSource(data=dict(x=[], y=[]))
        p.line('x', 'y', source=self.pred_test_source,
               line_width=3, color='#FF6B6B', line_dash='dashed', 
               legend_label="Model Predictions", alpha=0.9)
        p.x('x', 'y', source=self.pred_test_source,
             size=12, color='#FF6B6B', legend_label="Model Predictions", alpha=0.9)
        
        # Future forecast point
        self.forecast_source = ColumnDataSource(data=dict(x=[], y=[]))
        p.star('x', 'y', source=self.forecast_source,
               size=20, color='#FFD700', line_color='#FF8C00', 
               line_width=2, legend_label="Forecast", alpha=1.0)
        
        # Improved legend styling
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.background_fill_alpha = 0.8
        p.legend.border_line_color = "#cccccc"
        p.legend.padding = 10
        
        return p
    
    def update(self, data_manager, indicator: str):
        """Update the forecast panel with new data"""
        self.data_manager = data_manager
        self.current_indicator = indicator
        
        # Update country options
        countries = data_manager.get_filtered_countries(False)
        options = [("", "Select Country")]
        options.extend([(c, c.title() if isinstance(c, str) else c) for c in countries])
        self.country_select.options = options
        
        # Update year ranges
        if self.years:
            years_int = [int(y) for y in self.years]
            self.train_split_slider.start = min(years_int)
            self.train_split_slider.end = max(years_int) - 1
            self.train_split_slider.value = min(2018, max(years_int) - 1)
            
            self.target_year_slider.start = max(years_int) + 1
            self.target_year_slider.end = max(years_int) + 10
            self.target_year_slider.value = max(years_int) + 1
        
        # Clear previous results
        self._clear_results()
        
    def _clear_results(self):
        """Clear all visualization results"""
        self.hist_source.data = dict(x=[], y=[])
        self.test_source.data = dict(x=[], y=[])
        self.pred_test_source.data = dict(x=[], y=[])
        self.forecast_source.data = dict(x=[], y=[])
        self.results_div.text = ""
        self.fig.title.text = "Forecast Visualization"
        
    def _clear_analysis(self):
        """Clear analysis and reset for new country selection"""
        self._clear_results()
        self.country_select.value = ""
        self.current_country = ""
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.forecast_data = {}
        
        # Reset with clear message
        self.results_div.text = """
        <div style='text-align:center; color:#666; padding:25px;'>
            <p style='margin:0; font-size:16px;'>Analysis cleared. Ready for new forecast.</p>
        </div>
        """
        
    def _run_forecast(self):
        """Run the forecast when button is clicked"""
        country = self.country_select.value
        if not country or not self.current_indicator:
            self.results_div.text = """
            <div style='text-align:center; color:#666; padding:25px;'>
                <p style='margin:0; font-size:16px;'>Select a country and run forecast to see results</p>
            </div>
            """
            return
        
        # Show processing message
        self.results_div.text = """
        <div style='text-align:center; color:#324376; padding:25px;'>
            <p style='margin:0; font-size:16px;'>Processing forecast analysis...</p>
        </div>
        """
        
        try:
            # Get country data
            years, values = self.data_manager.get_country_data(country, self.current_indicator)
            
            if len(years) < 10:
                self.results_div.text = """
                <div style='color:#d32f2f; text-align:center; padding:30px;'>
                    <h3 style='margin:0; font-size:18px;'>Insufficient Data</h3>
                    <p style='margin:10px 0; font-size:14px;'>Need at least 10 years of data for reliable forecast.</p>
                </div>
                """
                return
            
            # Create DataFrame with all available indicators for the country
            df_dict = {'year': [int(y) for y in years]}
            
            # Get all available indicators for this country
            for indicator in self.data_manager._all_raw.keys():
                try:
                    indicator_years, indicator_values = self.data_manager.get_country_data(country, indicator)
                    if len(indicator_years) == len(years):
                        df_dict[indicator] = indicator_values
                except:
                    continue
            
            df = pd.DataFrame(df_dict).set_index('year')
            
            # Ensure the target indicator is in the DataFrame
            if self.current_indicator not in df.columns:
                self.results_div.text = """
                <div style='color:#d32f2f; text-align:center; padding:20px;'>
                    <h4 style='margin:0;'>❌ Data Error</h4>
                    <p style='margin:5px 0;'>Target indicator not found in country data.</p>
                </div>
                """
                self.model_info_div.text = "<div style='color:#d32f2f; font-size:14px;'><b>Model Status:</b> Failed - data error</div>"
                return
            
            # Engineer features
            X, y = self._engineer_features(df, self.current_indicator)
            
            if X is None or len(X) < 5:
                self.results_div.text = """
                <div style='color:#d32f2f; text-align:center; padding:20px;'>
                    <h4 style='margin:0;'>❌ Feature Engineering Failed</h4>
                    <p style='margin:5px 0;'>Insufficient data after feature engineering.</p>
                </div>
                """
                self.model_info_div.text = "<div style='color:#d32f2f; font-size:14px;'><b>Model Status:</b> Failed - feature engineering</div>"
                return
            
            # Train model
            train_limit = int(self.train_split_slider.value)
            model_results = self._train_lasso_model(X, y, train_limit)
            
            if model_results is None:
                return
            
            # Make forecast
            forecast_year = int(self.target_year_slider.value)
            forecast_value = self._predict_future(
                model_results['model'],
                df,
                model_results['x_scaler'],
                model_results['y_scaler'],
                X.columns,
                forecast_year
            )
            
            # Store current country for reference
            self.current_country = country
            
            # Update visualizations
            self._update_visualizations(
                model_results,
                forecast_value,
                forecast_year,
                country
            )
            
        except Exception as e:
            self.results_div.text = f"""
            <div style='color:#d32f2f; text-align:center; padding:20px;'>
                <h4 style='margin:0;'>❌ Analysis Error</h4>
                <p style='margin:5px 0;'>Error: {str(e)}</p>
            </div>
            """
            self.model_info_div.text = "<div style='color:#d32f2f; font-size:14px;'><b>Model Status:</b> Failed - analysis error</div>"
            import traceback
            print(f"Forecast error: {traceback.format_exc()}")
            
    def _engineer_features(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Create lag features for the model"""
        # Ensure we're working with numeric data
        df = df.select_dtypes(include=[np.number])
        
        if target not in df.columns:
            return None, None
        
        df_features = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            df_features[f'{col}_lag1'] = df[col].shift(1)
        
        df_features[target] = df[target]
        
        # Remove rows with NaN values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.dropna()
        
        if len(df_features) == 0:
            return None, None
        
        X = df_features.drop(columns=[target])
        y = df_features[target]
        
        # Final check for NaN values
        if X.isnull().any().any() or y.isnull().any():
            # If there are still NaN values, drop those rows
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
        
        return X, y
    
    def _train_lasso_model(self, X: pd.DataFrame, y: pd.Series, train_limit: int) -> Optional[Dict]:
        """Train the Lasso model"""
        # Split data
        X_train = X[X.index <= train_limit]
        X_test = X[X.index > train_limit]
        y_train = y[y.index <= train_limit]
        y_test = y[y.index > train_limit]
        
        if len(X_train) < 5:
            self.results_div.text = """
            <div style='color:#d32f2f; text-align:center; padding:20px;'>
                <h4 style='margin:0;'>❌ Training Data Insufficient</h4>
                <p style='margin:5px 0;'>Need at least 5 years of training data.</p>
            </div>
            """
            self.model_info_div.text = "<div style='color:#d32f2f; font-size:14px;'><b>Model Status:</b> Failed - insufficient training data</div>"
            return None
        
        # Additional check for NaN values before scaling
        if X_train.isnull().any().any():
            X_train = X_train.dropna()
            y_train = y_train[X_train.index]
        
        if len(X_test) > 0 and X_test.isnull().any().any():
            X_test = X_test.dropna()
            y_test = y_test[X_test.index]
        
        # Scale data
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        try:
            X_train_scaled = x_scaler.fit_transform(X_train)
            X_test_scaled = x_scaler.transform(X_test) if len(X_test) > 0 else np.array([])
            
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel() if len(y_test) > 0 else np.array([])
        except Exception as e:
            self.results_div.text = f"""
            <div style='color:#d32f2f; text-align:center; padding:20px;'>
                <h4 style='margin:0;'>❌ Data Scaling Error</h4>
                <p style='margin:5px 0;'>Error in data scaling: {str(e)}</p>
            </div>
            """
            self.model_info_div.text = "<div style='color:#d32f2f; font-size:14px;'><b>Model Status:</b> Failed - scaling error</div>"
            return None
        
        # Train model
        try:
            model = LassoCV(cv=min(5, len(X_train)), max_iter=10000, n_jobs=-1).fit(X_train_scaled, y_train_scaled)
        except Exception as e:
            self.results_div.text = f"""
            <div style='color:#d32f2f; text-align:center; padding:20px;'>
                <h4 style='margin:0;'>❌ Model Training Error</h4>
                <p style='margin:5px 0;'>Error training model: {str(e)}</p>
            </div>
            """
            self.model_info_div.text = "<div style='color:#d32f2f; font-size:14px;'><b>Model Status:</b> Failed - training error</div>"
            return None
        
        # Make predictions on test set if available
        y_pred_test = None
        metrics = {}
        
        if len(X_test) > 0:
            y_pred_test_scaled = model.predict(X_test_scaled)
            y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
            
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            mape = self._calculate_mape(y_test.values, y_pred_test)
            
            metrics = {'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
        
        return {
            'model': model,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'metrics': metrics,
            'feature_names': X.columns
        }
    
    def _predict_future(self, model, df_original: pd.DataFrame, x_scaler, y_scaler, 
                       feature_names: List[str], future_year: int) -> float:
        """Make future prediction"""
        try:
            last_year = df_original.index.max()
            last_year_data = df_original.loc[last_year]
            
            # Create feature vector with all required features
            future_features = {}
            for feature in feature_names:
                # Extract the original column name from the lag feature name
                if feature.endswith('_lag1'):
                    orig_col = feature[:-5]  # Remove '_lag1' suffix
                    if orig_col in last_year_data:
                        future_features[feature] = last_year_data[orig_col]
                    else:
                        # If column doesn't exist, use 0 or NaN
                        future_features[feature] = 0
                else:
                    future_features[feature] = 0
            
            future_input_df = pd.DataFrame([future_features], columns=feature_names)
            
            # Check for NaN values and replace with 0
            future_input_df = future_input_df.fillna(0)
            
            # Scale and predict
            future_input_scaled = x_scaler.transform(future_input_df)
            prediction_scaled = model.predict(future_input_scaled)
            prediction_original = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()[0]
            
            return prediction_original
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _update_visualizations(self, model_results: Dict, forecast_value: float, 
                              forecast_year: int, country: str):
        """Update all visualizations with results"""
        # Update main plot
        y_train = model_results['y_train']
        y_test = model_results['y_test']
        y_pred_test = model_results['y_pred_test']
        
        # Historical data
        self.hist_source.data = {
            'x': [str(y) for y in y_train.index],
            'y': y_train.values.tolist()
        }
        
        # Test data
        if len(y_test) > 0:
            self.test_source.data = {
                'x': [str(y) for y in y_test.index],
                'y': y_test.values.tolist()
            }
            self.pred_test_source.data = {
                'x': [str(y) for y in y_test.index],
                'y': y_pred_test.tolist()
            }
        
        # Forecast point
        self.forecast_source.data = {
            'x': [str(forecast_year)],
            'y': [forecast_value]
        }
        
        # Update title with clean formatting
        self.fig.title.text = f"{country.title()} - {self._format_indicator_name(self.current_indicator)}"
        
        # Update results with improved styling
        metrics = model_results['metrics']
        model = model_results['model']
        
        # Count non-zero features
        n_features = sum(1 for coef in model.coef_ if coef != 0)
        
        # Count non-zero features
        n_features = sum(1 for coef in model.coef_ if coef != 0)
        
        # Clean, organized results display
        results_html = f"""
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <h3 style='margin:0; color:#324376; font-size:18px; font-weight:600;'>
                    {country.title()} Forecast Results
                </h3>
            </div>
            <div style='text-align:right;'>
                <span style='color:#2E8B57; font-size:20px; font-weight:bold;'>{forecast_year}: {forecast_value:,.2f}</span>
            </div>
        </div>
        <div style='margin-top:15px; padding-top:15px; border-top:1px solid #eee; font-size:14px; color:#666;'>
            <span style='margin-right:30px;'><strong>Model Alpha:</strong> {model.alpha_:.6f}</span>
            <span style='margin-right:30px;'><strong>Features Used:</strong> {n_features}/{len(model.coef_)}</span>
        """
        
        if metrics:
            results_html += f"""
            <span style='margin-right:30px;'><strong>RMSE:</strong> <span style='color:#2E8B57;'>{metrics['RMSE']:,.2f}</span></span>
            <span><strong>MAPE:</strong> <span style='color:#FF6B6B;'>{metrics['MAPE']:.1f}%</span></span>
            """
        
        results_html += "</div>"
        
        results_html += "</div>"
        
        self.results_div.text = results_html
    
    def _format_indicator_name(self, indicator):
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
    
    def clear(self):
        """Clear all data and reset the panel"""
        self._clear_results()
        self.current_indicator = ""
        self.current_country = ""
        self.country_select.value = ""