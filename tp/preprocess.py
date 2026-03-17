from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import sosfilt

from .utils import buffering_power, common_average_reference, compute_band_filters


def select_channels(X, channel_selection=None):
    if channel_selection is None:
        return np.asarray(X)
    return np.asarray(X)[:, channel_selection]


def apply_car(X, car_method="car", axis=1):
    if car_method in (None, False, "none"):
        return np.asarray(X).copy()

    method = str(car_method).lower()
    if method in {"car", "common_average_reference", "mean"}:
        return common_average_reference(np.asarray(X), axis=axis)

    raise ValueError(f"Unsupported car_method: {car_method}")


def apply_band_filter_bank(train_ecog, test_ecog, bands, fs=1000):
    band_filters = compute_band_filters(bands, fs=fs)
    train_band_signals = [sosfilt(filt, train_ecog, axis=0) for filt in band_filters]
    test_band_signals = [sosfilt(filt, test_ecog, axis=0) for filt in band_filters]

    return {
        "band_filters": band_filters,
        "train_band_signals": train_band_signals,
        "test_band_signals": test_band_signals,
        "X_train_filtered": np.concatenate(train_band_signals, axis=1),
        "X_test_filtered": np.concatenate(test_band_signals, axis=1),
    }


def normalize_features(train_features, test_features, normalization="zscore", train_slice=None):
    if normalization in (None, False, "none"):
        return {
            "feature_mean": None,
            "feature_std": None,
            "X_train_features": np.asarray(train_features).copy(),
            "X_test_features": np.asarray(test_features).copy(),
        }

    method = str(normalization).lower()
    if method not in {"zscore", "standard", "standardize", "standardise"}:
        raise ValueError(f"Unsupported normalization method: {normalization}")

    train_reference = np.asarray(train_features)
    if train_slice is not None:
        train_reference = train_reference[train_slice]

    feature_mean = train_reference.mean(axis=0, keepdims=True)
    feature_std = train_reference.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "X_train_features": (np.asarray(train_features) - feature_mean) / feature_std,
        "X_test_features": (np.asarray(test_features) - feature_mean) / feature_std,
    }


def run_preprocessing_pipeline(
    train_ecog,
    test_ecog,
    *,
    bands: Iterable[tuple[float, float]],
    fs=1000,
    channel_selection=None,
    car_method="car",
    normalization="zscore",
    feature_win_size=1000,
    hop_size=40,
    n_buffer=1,
    train_normalization_slice=None,
):
    train_selected = select_channels(train_ecog, channel_selection)
    test_selected = select_channels(test_ecog, channel_selection)

    train_car = apply_car(train_selected, car_method=car_method, axis=1)
    test_car = apply_car(test_selected, car_method=car_method, axis=1)

    filtered = apply_band_filter_bank(train_car, test_car, bands=bands, fs=fs)

    X_train_features_raw = buffering_power(
        filtered["X_train_filtered"],
        win_size=feature_win_size,
        hop=hop_size,
        n_buffer=n_buffer,
    )
    X_test_features_raw = buffering_power(
        filtered["X_test_filtered"],
        win_size=feature_win_size,
        hop=hop_size,
        n_buffer=n_buffer,
    )

    normalized = normalize_features(
        X_train_features_raw,
        X_test_features_raw,
        normalization=normalization,
        train_slice=train_normalization_slice,
    )

    return {
        "bands": list(bands),
        "channel_selection": channel_selection,
        "car_method": car_method,
        "normalization": normalization,
        "feature_win_size": feature_win_size,
        "hop_size": hop_size,
        "n_buffer": n_buffer,
        "ecog_train_selected": train_selected,
        "ecog_test_selected": test_selected,
        "ecog_train_car": train_car,
        "ecog_test_car": test_car,
        "band_filters": filtered["band_filters"],
        "train_band_signals": filtered["train_band_signals"],
        "test_band_signals": filtered["test_band_signals"],
        "X_train_filtered": filtered["X_train_filtered"],
        "X_test_filtered": filtered["X_test_filtered"],
        "X_train_features_raw": X_train_features_raw,
        "X_test_features_raw": X_test_features_raw,
        "feature_mean": normalized["feature_mean"],
        "feature_std": normalized["feature_std"],
        "X_train_features": normalized["X_train_features"],
        "X_test_features": normalized["X_test_features"],
        "X_train_features_full": normalized["X_train_features"],
        "X_test_full": normalized["X_test_features"],
    }
