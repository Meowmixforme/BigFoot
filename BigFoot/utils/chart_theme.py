"""
Chart theming utilities for dark theme
"""

from config.settings import COLOR_PALETTE

def get_dark_chart_layout():
    """Get standard layout configuration for dark theme charts"""
    return {
        'plot_bgcolor': COLOR_PALETTE['light'],
        'paper_bgcolor': COLOR_PALETTE['light'],
        'font': {'color': COLOR_PALETTE['primary']},
        'title': {'font': {'color': COLOR_PALETTE['dark'], 'size': 16}},
        'xaxis': {
            'gridcolor': '#4A5568',
            'linecolor': '#4A5568',
            'tickcolor': COLOR_PALETTE['primary'],
            'titlefont': {'color': COLOR_PALETTE['primary']},
            'tickfont': {'color': COLOR_PALETTE['primary']}
        },
        'yaxis': {
            'gridcolor': '#4A5568',
            'linecolor': '#4A5568',
            'tickcolor': COLOR_PALETTE['primary'],
            'titlefont': {'color': COLOR_PALETTE['primary']},
            'tickfont': {'color': COLOR_PALETTE['primary']}
        },
        'legend': {
            'font': {'color': COLOR_PALETTE['primary']},
            'bgcolor': 'rgba(74, 85, 104, 0.8)'
        }
    }

def apply_dark_theme_to_fig(fig):
    """Apply dark theme styling to a plotly figure"""
    dark_layout = get_dark_chart_layout()
    fig.update_layout(dark_layout)
    return fig