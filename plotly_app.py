import dash
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

base_path = "/home/tanalp/toxicainment/simple-inference-benchmark/notebooks/merged_data"

model_names = set()
df_long_list = []

for file in os.listdir(base_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(base_path, file))
        df_long_list.append(
            pd.concat(
                [
                    df.assign(
                        source="human",
                        rationale_hashtag=df["comment_contains_hashtag_human"],
                        rationale_audio=df["comment_contains_audio_human"],
                        label=df["value_human"],
                        comment=df["comment_human"],
                        model=df["Model ID"],
                        annotator=df["classification_by"],
                    ),
                    df.assign(
                        source="ai",
                        rationale_hashtag=df["comment_contains_hashtag_ai"],
                        rationale_audio=df["comment_contains_audio_ai"],
                        label=df["value_ai"],
                        comment=df["comment_ai"],
                        model=df["Model ID"],
                        annotator="",
                    ),
                ]
            )
        )
        if "Model ID" in df.columns:
            model_names.update(df["Model ID"].dropna().unique())

model_names = sorted(model_names)
df_long = pd.concat(df_long_list, ignore_index=True)


app = Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME])

app.layout = html.Div(
    [
        html.H1(
            children="Human vs AI Rationales (Cognitive Map of Disagreement)",
            style={"textAlign": "center"},
        ),
        dcc.Dropdown(
            options=[{"label": m, "value": m} for m in model_names],
            value=model_names[0] if model_names else None,
            id="model-dropdown",
        ),
        dcc.Input(
            id="post-id-input",
            type="text",
            placeholder="Enter Post ID (optional)",
            debounce=True,
            style={"marginBottom": "10px"},
        ),
        dcc.Graph(id="graph-content"),
    ]
)


@callback(
    Output("graph-content", "figure"),
    Input("model-dropdown", "value"),
    Input("post-id-input", "value"),
)
def update_graph(selected_model, search_postid):
    dff = df_long[df_long["model"] == selected_model].copy()

    hover_cols = [
        "variable",
        "post_id",
        "comment",
        "source",
        "label",
        "annotator",
        "label_alignment",
    ]

    jitter_strength = 0.3
    dff["jittered_x"] = dff["rationale_hashtag"].astype(float) + np.random.uniform(
        -jitter_strength, jitter_strength, size=len(dff)
    )
    dff["jittered_y"] = dff["rationale_audio"].astype(float) + np.random.uniform(
        -jitter_strength, jitter_strength, size=len(dff)
    )
    dff["label_alignment"] = (
        dff["label"] == dff.groupby(["post_id", "variable"])["label"].transform("max")
    ).astype(int)

    if search_postid and search_postid.isdigit():
        dff["highlight"] = dff["post_id"].astype(str) == search_postid
    else:
        dff["highlight"] = False

    dff_highlight = dff[dff["highlight"]]
    dff_rest = dff[~dff["highlight"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=dff_rest["jittered_x"],
            y=dff_rest["jittered_y"],
            z=dff_rest["label_alignment"],
            mode="markers",
            marker=dict(
                size=5,
                color=dff_rest["source"].map({"human": "blue", "ai": "red"}),
                opacity=0.3,
                symbol=dff_rest["label"]
                .map({0: "circle", 1: "diamond"})
                .fillna("circle"),
            ),
            customdata=dff_rest[hover_cols].values,
            hovertemplate=(
                "Variable: %{customdata[0]}<br>"
                "Post ID: %{customdata[1]}<br>"
                "Comment: %{customdata[2]}<br>"
                "Source: %{customdata[3]}<br>"
                "Label: %{customdata[4]}<br>"
                "Annotator: %{customdata[5]}<br>"
                "Label Alignment: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
            name="Other Posts",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=dff_highlight["jittered_x"],
            y=dff_highlight["jittered_y"],
            z=dff_highlight["label_alignment"],
            mode="markers",
            marker=dict(
                size=10,
                color="gold",
                opacity=1.0,
                symbol=dff_highlight["label"]
                .map({0: "circle", 1: "diamond"})
                .fillna("circle"),
            ),
            customdata=dff_highlight[hover_cols].values,
            hovertemplate=(
                "Variable: %{customdata[0]}<br>"
                "Post ID: %{customdata[1]}<br>"
                "Comment: %{customdata[2]}<br>"
                "Source: %{customdata[3]}<br>"
                "Label: %{customdata[4]}<br>"
                "Annotator: %{customdata[5]}<br>"
                "Label Alignment: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
            name="Selected Post (ID: " + str(search_postid) + ")",
        )
    )

    fig.update_layout(
        title=f"Rationale & Label Alignment for Model: {selected_model}",
        scene=dict(
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["False", "True"],
                title="Hashtag",
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["False", "True"],
                title="Audio",
            ),
            zaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Disagree", "Agree"],
                title="Label Alignment",
            ),
        ),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=False)
