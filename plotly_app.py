from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np

df = pd.read_csv('/home/tanalp/toxicainment/simple-inference-benchmark/notebooks/merged_comments.csv')

df_long = pd.concat([
    df.assign(source='human', 
              rationale_hashtag=df['comment_contains_hashtag_human'],
              rationale_audio=df['comment_contains_audio_human'],
              label=df['value_human'],
              comment=df['comment_human'],
              model=df['Model ID']),
    df.assign(source='ai', 
              rationale_hashtag=df['comment_contains_hashtag_ai'],
              rationale_audio=df['comment_contains_audio_ai'],
              label=df['value_ai'],
              comment=df['comment_ai'],
              model=df['Model ID'])
])

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Human vs AI Rationales (Cognitive Map of Disagreement)', style={'textAlign':'center'}),
    dcc.Dropdown(
        options=[{'label': m, 'value': m} for m in sorted(df['Model ID'].dropna().unique())],
        value=sorted(df['Model ID'].dropna().unique())[0],
        id='model-dropdown'
    ),
    dcc.Dropdown(
        options=[
            {'label': 'Hashtag', 'value': 'rationale_hashtag'},
            {'label': 'Audio', 'value': 'rationale_audio'}
        ],
        value='rationale_hashtag',
        id='rationale-dropdown'
    ),
    dcc.Graph(id='graph-content')
])

@callback(
    Output('graph-content', 'figure'),
    Input('model-dropdown', 'value'),
    Input('rationale-dropdown', 'value')
)

def update_graph(selected_model, rationale_type):
    dff = df_long[df_long['model'] == selected_model].copy()

    jitter_strength = 0.1
    dff['jittered_x'] = dff['rationale_hashtag'].astype(float) + np.random.uniform(-jitter_strength, jitter_strength, size=len(dff))
    dff['jittered_y'] = dff['rationale_audio'].astype(float) + np.random.uniform(-jitter_strength, jitter_strength, size=len(dff))

    fig = px.scatter(
        dff,
        x='jittered_x',
        y='jittered_y',
        color='source',
        symbol='label',
        hover_data=['variable', 'post_id', 'comment'],
        labels={'jittered_x': 'Hashtag', 'jittered_y': 'Audio'}
    )

    fig.update_layout(
        title=f'Rationale Comparison for Model: {selected_model}',
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['False', 'True'],
            title='Hashtag'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['False', 'True'],
            title='Audio'
        )
    )

    return fig

if __name__ == '__main__':
    app.run(debug=False)