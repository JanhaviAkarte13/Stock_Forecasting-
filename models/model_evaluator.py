import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    """Evaluate and compare model performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, actual, predicted, model_name):
        """Calculate comprehensive metrics for a model"""
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Calculate R-squared
        try:
            r2 = r2_score(actual, predicted)
        except:
            r2 = 0
        
        self.metrics[model_name] = {
            'MSE': round(mse, 4),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'MAPE': round(mape, 4),
            'R2': round(r2, 4)
        }
        
        return self.metrics[model_name]
    
    def compare_models(self):
        """Return comparison DataFrame of all models"""
        if not self.metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics).T
        df = df.sort_values('RMSE')
        
        return df
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.metrics:
            return None
        
        df = self.compare_models()
        best_model = df.index[0]
        
        return best_model, self.metrics[best_model]
    
    def generate_report(self):
        """Generate detailed evaluation report"""
        if not self.metrics:
            return "No models evaluated yet."
        
        report = "📊 MODEL EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        df = self.compare_models()
        
        for idx, (model_name, row) in enumerate(df.iterrows(), 1):
            report += f"{idx}. {model_name}\n"
            report += f"   RMSE: {row['RMSE']:.4f}\n"
            report += f"   MAE: {row['MAE']:.4f}\n"
            report += f"   MAPE: {row['MAPE']:.2f}%\n"
            report += f"   R²: {row['R2']:.4f}\n"
            report += "-" * 50 + "\n"
        
        best_model, best_metrics = self.get_best_model()
        report += f"\n🏆 Best Model: {best_model}\n"
        report += f"   RMSE: {best_metrics['RMSE']:.4f}\n"
        
        return report
    
    def get_accuracy_percentage(self, model_name):
        """Calculate accuracy percentage based on MAPE"""
        if model_name not in self.metrics:
            return 0
        
        mape = self.metrics[model_name]['MAPE']
        accuracy = 100 - mape
        
        return max(0, min(100, accuracy))  # Clamp between 0 and 100