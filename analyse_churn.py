import os
import socket
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def find_free_port(start_port, end_port):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except socket.error:
                continue
    raise IOError("No free ports available in the specified range")

def load_data():
    df = pd.read_csv('high_churn_list.csv')
    df = df.drop('Surname', axis=1)  # Assuming 'Surname' is not needed for analysis
    df = bin_data(df)
    return df

def bin_data(df):
    bins = {
        'Age': np.linspace(df['Age'].min(), df['Age'].max(), 7),
        'CreditScore': np.linspace(df['CreditScore'].min(), df['CreditScore'].max(), 7),
        'EstimatedSalary': np.linspace(df['EstimatedSalary'].min(), df['EstimatedSalary'].max(), 11).astype(int),
        'Tenure': np.linspace(df['Tenure'].min(), df['Tenure'].max(), 11).astype(int),
        'Balance': np.linspace(df['Balance'].min(), df['Balance'].max(), 11).astype(int)
    }
    bin_labels = {
        'Age': ["18-28", "28-38", "38-48", "48-58", "58-68", "68-78"],
        'CreditScore': ["370-450", "450-530", "530-610", "610-690", "690-770", "770-850"],
        'EstimatedSalary': [f"{int(a)}-{int(b)}" for a, b in zip(bins['EstimatedSalary'][:-1], bins['EstimatedSalary'][1:])],
        'Tenure': [f"{int(a)}-{int(b)}" for a, b in zip(bins['Tenure'][:-1], bins['Tenure'][1:])],
        'Balance': [f"{int(a)}-{int(b)}" for a, b in zip(bins['Balance'][:-1], bins['Balance'][1:])]
    }

    for column, edges in bins.items():
        df[f'{column}_binned'] = pd.cut(df[column], bins=edges, labels=bin_labels[column], include_lowest=True)

    # Binning categorical data
    categorical_cols = ['Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Geography']
    for col in categorical_cols:
        df[f'{col}_binned'] = pd.Categorical(df[col])

    return df

def compute_statistics(df, features):
    if not os.path.exists('static'):
        os.makedirs('static')  # Ensure the directory exists

    results = {}
    for feature in features:
        binned_feature = feature + '_binned'
        if binned_feature in df.columns:
            grouped = df.groupby(binned_feature, observed=True)['prediction_probability']
            results[feature] = {
                'mean': grouped.mean(),
                'median': grouped.median()
            }

            # Save each set of statistics to a CSV file
            stats_df = pd.DataFrame(results[feature])
            stats_df.to_csv(f'static/{feature}_stats.csv')

    return results

def create_interactive_plot(results):
    if not results:
        print("No data available to plot.")
        return  # Exit the function if no results to plot

    fig = go.Figure()

    for feature, stats in results.items():
        fig.add_trace(
            go.Bar(
                x=stats['mean'].index,
                y=stats['mean'],
                name='Mean Churn Probability',
                text=np.round(stats['mean'], 2),
                textposition='auto',
                marker_color='red',
                visible=False  # Initially invisible
            )
        )
        fig.add_trace(
            go.Bar(
                x=stats['median'].index,
                y=stats['median'],
                name='Median Churn Probability',
                text=np.round(stats['median'], 2),
                textposition='auto',
                marker_color='blue',
                visible=False  # Initially invisible
            )
        )

    # Update traces to be visible or invisible initially
    buttons = []
    for i, feature in enumerate(results.keys()):
        visibility = [False] * len(results) * 2
        visibility[i*2] = True  # Mean visibility
        visibility[i*2 + 1] = True  # Median visibility
        buttons.append(
            dict(
                label=feature,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Churn Probability By {feature}"}]
            )
        )

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,  # Set to 0 for left alignment
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ],
        title="Select Feature to Display Churn Probability"
    )

    # Set common layout properties
    fig.update_layout(
        barmode='group',
        title_x=0.5,
        xaxis_title="Bins",
        yaxis_title="Churn Prediction Probability",
        legend_title="Statistics"
    )

    # Find a free port and run the server
    port = find_free_port(42455, 42465)
    fig.show(port=port)

if __name__ == "__main__":
    df_hcl = load_data()
    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    results = compute_statistics(df_hcl, features)
    create_interactive_plot(results)
