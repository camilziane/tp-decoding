from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_prediction_vs_truth(
    y_true,
    y_pred,
    *,
    sample_rate,
    class_labels,
    title,
    predicted_label="Model Prediction",
    start=0,
    max_points=None,
    show=False,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    if max_points is None:
        end = y_true.shape[0]
    else:
        end = min(start + max_points, y_true.shape[0])

    if end <= start:
        raise ValueError("Empty plotting interval")

    time_axis = np.arange(start, end) / sample_rate

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=y_true[start:end],
            mode="lines",
            name="Ground Truth",
            line=dict(shape="hv", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=y_pred[start:end],
            mode="lines",
            name=predicted_label,
            line=dict(shape="hv", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Class",
        hovermode="x unified",
        template="plotly_white",
        height=550,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(class_labels))),
        ticktext=list(class_labels),
    )

    if show:
        fig.show()

    return fig
