import pandas as pd
import matplotlib.pyplot as plt


def visualize_runtime(df: pd.DataFrame) -> None:
    # The model takes longer for more input frames and for more output tokens generated
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(
        df["Tokens_Generated"], df["Model_Runtime"], c=df["Total_Frames"]
    )
    plt.colorbar(scatter, label="Input Frames Per Video")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("Output Tokens Generated")
    ax.set_ylabel("Model Runtime")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    return fig
