import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import os
from typing import Dict, List, Any

class VisualizationService:
    def __init__(self, charts_dir: str = "charts"):
        self.charts_dir = charts_dir
        if not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir)

    def save_static_chart(self, df: pd.DataFrame, analysis: Dict[str, Any]):
        """
        Generate and save a static Matplotlib chart.
        """
        symbol = analysis['symbol']
        date_str = analysis['analysis_date']
        filename = f"{symbol}_{date_str}.png"
        filepath = os.path.join(self.charts_dir, filename)

        plt.figure(figsize=(15, 8))
        plt.plot(df.index, df['Close'], label='Price', color='blue', alpha=0.6)
        
        # Plot supports
        for sup in analysis['supports']:
            plt.axhline(y=sup, color='green', linestyle='--', alpha=0.5)
        
        # Plot resistances
        for res in analysis['resistances']:
            plt.axhline(y=res, color='red', linestyle='--', alpha=0.5)

        plt.title(f"Technical Analysis: {symbol} (S/R Detection)")
        plt.legend(['Close Price', 'Support', 'Resistance'])
        plt.grid(True, alpha=0.3)
        plt.savefig(filepath)
        plt.close()
        return filepath

    def save_interactive_chart(self, df: pd.DataFrame, analysis: Dict[str, Any]):
        """
        Generate and save an interactive Plotly chart.
        """
        symbol = analysis['symbol']
        date_str = analysis['analysis_date']
        filename = f"{symbol}_{date_str}.html"
        filepath = os.path.join(self.charts_dir, filename)

        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Market Data'
        ))

        # Add support lines
        for sup in analysis['supports']:
            fig.add_hline(y=sup, line_dash="dash", line_color="green", 
                         annotation_text=f"Support: {sup:.4f}")

        # Add resistance lines
        for res in analysis['resistances']:
            fig.add_hline(y=res, line_dash="dash", line_color="red", 
                         annotation_text=f"Resistance: {res:.4f}")

        fig.update_layout(
            title=f"Forex AI Analysis - {symbol}",
            yaxis_title="Price",
            xaxis_title="Date",
            template="plotly_dark",
            height=800
        )

        fig.write_html(filepath)
        return filepath
