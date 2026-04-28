"""
weather_features.py
-------------------
4 независимые функции для присоединения погодных фичей era5 к снимкам landsat.
без SHORT_NAMES — все имена колонок генерируются из полных оригинальных имён era5.

использование:
  df = pd.read_csv('landsat_data_2019_2025.csv')
  era5 = pd.read_csv('join_every_hour_era5_data.zip')
  df = add_all_weather_features(df, era5)
"""

import numpy as np
import pandas as pd
import time

# ---------------------------------------------------------------------------
# конфигурация
# ---------------------------------------------------------------------------

WEATHER_COLS = [
    "temperature_2m",
    "skin_temperature",
    "total_precipitation_hourly",
    "surface_solar_radiation_downwards_hourly",
    "volumetric_soil_water_layer_1",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "total_evaporation_hourly",
    "dewpoint_temperature_2m",
    "evaporation_from_vegetation_transpiration_hourly",
]

AGG_FUNCS = ["min", "max", "mean"]
ROLLING_WINDOWS_DAYS = [1, 2, 5, 7, 10, 14]
LAG_DAYS = [2, 3, 4, 5]
SEC_PER_HOUR = 3600


# ---------------------------------------------------------------------------
# подготовка данных
# ---------------------------------------------------------------------------


def prepare_era5(era5_hourly):
    """
    подготовка era5: парсинг дат, сортировка, вычисление local hour и масок.
    """
    df = era5_hourly.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_str_utc"], utc=True)
    df["utc_offset"] = np.round(df["center_lon"] / 15.0).astype(int)
    df["local_hour"] = (
        df["datetime_utc"] + pd.to_timedelta(df["utc_offset"], unit="h")
    ).dt.hour
    df["is_daytime"] = df["local_hour"].between(10, 15, inclusive="both")
    df["is_nighttime"] = (df["local_hour"] >= 22) | (df["local_hour"] <= 3)
    df = df.sort_values(["ID", "datetime_utc"]).reset_index(drop=True)
    return df


def prepare_snapshots(df_snapshots):
    """подготовка снимков: парсинг дат."""
    df = df_snapshots.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_str_utc"], utc=True)
    return df


# ---------------------------------------------------------------------------
# ядро: векторизованный вычислитель агрегатов
# ---------------------------------------------------------------------------


def _compute_windowed_aggs(
    era5_sorted, snap_times_i64, id_vals, window_spec, weather_cols, masks=None
):
    """
    универсальная функция для вычисления min/max/mean по временным окнам.

    параметры:
        era5_sorted: подготовленный era5 (отсортирован по ID, datetime_utc)
        snap_times_i64: ndarray[int64] — время каждого снимка в секундах (utc)
        id_vals: ndarray — ID каждого снимка
        window_spec: список кортежей (t_offset_start, t_offset_end, suffix_name)
            интервал: (t - t_offset_start, t - t_offset_end]
        weather_cols: список колонок для агрегации
        masks: dict {era5_mask_col: prefix} или None
    """
    index = {}
    for id_val, grp in era5_sorted.groupby("ID"):
        times = grp["datetime_utc"].values.astype("datetime64[s]").astype(np.int64)
        vals = {c: grp[c].values.astype(np.float64) for c in weather_cols}
        m = {}
        if masks is not None:
            for mk in masks:
                m[mk] = grp[mk].values
        index[id_val] = (times, vals, m)

    n = len(snap_times_i64)
    result = {}
    mask_iter = masks if masks is not None else {None: None}

    for mask_key, _ in mask_iter.items():
        prefix = f"{masks[mask_key]}_" if mask_key is not None else ""
        for t_off_start, t_off_end, suffix in window_spec:
            for col in weather_cols:
                for agg in AGG_FUNCS:
                    col_name = f"{col}_{prefix}{agg}_{suffix}"
                    result[col_name] = np.full(n, np.nan)

    unique_ids = np.unique(id_vals)
    for uid in unique_ids:
        if uid not in index:
            continue

        e_times, e_vals, e_masks = index[uid]
        snap_mask = id_vals == uid
        s_idx = np.where(snap_mask)[0]
        s_times = snap_times_i64[snap_mask]

        for mask_key, _ in mask_iter.items():
            prefix = f"{masks[mask_key]}_" if mask_key is not None else ""

            if mask_key is not None and mask_key in e_masks:
                base_mask = e_masks[mask_key]
            elif mask_key is not None:
                continue
            else:
                base_mask = None

            for t_off_start, t_off_end, suffix in window_spec:
                left_arr = np.searchsorted(e_times, s_times - t_off_start, side="left")
                right_arr = np.searchsorted(e_times, s_times - t_off_end, side="right")

                for col in weather_cols:
                    e_col = e_vals[col]

                    for agg in AGG_FUNCS:
                        col_name = f"{col}_{prefix}{agg}_{suffix}"
                        out = result[col_name]

                        for j in range(len(s_idx)):
                            l, r = left_arr[j], right_arr[j]
                            if l >= r:
                                continue
                            chunk = e_col[l:r]
                            if base_mask is not None:
                                chunk = chunk[base_mask[l:r]]
                                if len(chunk) == 0:
                                    continue
                            if agg == "min":
                                out[s_idx[j]] = np.nanmin(chunk)
                            elif agg == "max":
                                out[s_idx[j]] = np.nanmax(chunk)
                            elif agg == "mean":
                                out[s_idx[j]] = np.nanmean(chunk)

    return result


# ---------------------------------------------------------------------------
# функция 1: ближайший час "как есть"
# ---------------------------------------------------------------------------


def add_weather_features_1(df_snapshots, era5_hourly):
    """
    для каждого снимка находит ближайший час в era5 и берёт значения "как есть".
    """
    era5 = era5_hourly.copy()
    if "datetime_utc" not in era5.columns:
        era5 = prepare_era5(era5)

    snaps = prepare_snapshots(df_snapshots)

    lookup = {}
    for id_val, grp in era5.groupby("ID"):
        ts = grp["datetime_utc"].values.astype("datetime64[s]").astype(np.int64)
        vals = {c: grp[c].values.astype(np.float64) for c in WEATHER_COLS}
        lookup[id_val] = (ts, vals)

    snap_rounded = (
        pd.Series(snaps["datetime_utc"].values)
        .dt.round("h")
        .values.astype("datetime64[s]")
        .astype(np.int64)
    )
    snap_ids = snaps["ID"].values
    n = len(snaps)

    result_cols = {}
    for col in WEATHER_COLS:
        result_cols[f"{col}_nearest"] = np.full(n, np.nan)

    for uid in np.unique(snap_ids):
        if uid not in lookup:
            continue
        e_ts, e_vals = lookup[uid]
        mask = snap_ids == uid
        idx = np.where(mask)[0]
        queries = snap_rounded[idx]

        pos = np.searchsorted(e_ts, queries, side="left")

        for k in range(len(idx)):
            best = None
            best_dist = np.inf
            for cand in [pos[k] - 1, pos[k]]:
                if 0 <= cand < len(e_ts):
                    d = abs(int(e_ts[cand]) - int(queries[k]))
                    if d < best_dist:
                        best_dist = d
                        best = cand
            if best is not None and best_dist <= 1800:
                for col in WEATHER_COLS:
                    result_cols[f"{col}_nearest"][idx[k]] = e_vals[col][best]

    new_df = pd.DataFrame(result_cols, index=df_snapshots.index)
    return pd.concat([df_snapshots, new_df], axis=1)


# ---------------------------------------------------------------------------
# функция 2: min/max/mean за последние N дней
# ---------------------------------------------------------------------------


def add_weather_features_2(df_snapshots, era5_hourly):
    """
    min/max/mean за 1, 2, 5, 7, 10, 14 дней от часа снимка.
    интервал (snapshot_time - N*24h, snapshot_time].
    """
    era5 = era5_hourly.copy()
    if "datetime_utc" not in era5.columns:
        era5 = prepare_era5(era5)

    snaps = prepare_snapshots(df_snapshots)
    snap_ts = snaps["datetime_utc"].values.astype("datetime64[s]").astype(np.int64)
    snap_ids = snaps["ID"].values

    window_spec = [
        (d * 24 * SEC_PER_HOUR, 0, f"last_{d}d") for d in ROLLING_WINDOWS_DAYS
    ]

    aggs = _compute_windowed_aggs(era5, snap_ts, snap_ids, window_spec, WEATHER_COLS)

    new_df = pd.DataFrame(aggs, index=df_snapshots.index)
    return pd.concat([df_snapshots, new_df], axis=1)


# ---------------------------------------------------------------------------
# функция 3: min/max/mean за конкретные дни назад (лаги)
# ---------------------------------------------------------------------------


def add_weather_features_3(df_snapshots, era5_hourly):
    """
    min/max/mean за лаги 2, 3, 4, 5 дней назад от часа снимка.
    lag2 = (-48h, -24h], lag3 = (-72h, -48h], и т.д.
    """
    era5 = era5_hourly.copy()
    if "datetime_utc" not in era5.columns:
        era5 = prepare_era5(era5)

    snaps = prepare_snapshots(df_snapshots)
    snap_ts = snaps["datetime_utc"].values.astype("datetime64[s]").astype(np.int64)
    snap_ids = snaps["ID"].values

    # lag2=(-48h,-24h], lag3=(-72h,-48h], ...
    window_spec = [
        (lag * 24 * SEC_PER_HOUR, (lag - 1) * 24 * SEC_PER_HOUR, f"lag{lag}")
        for lag in LAG_DAYS
    ]

    aggs = _compute_windowed_aggs(era5, snap_ts, snap_ids, window_spec, WEATHER_COLS)

    new_df = pd.DataFrame(aggs, index=df_snapshots.index)
    return pd.concat([df_snapshots, new_df], axis=1)


# ---------------------------------------------------------------------------
# функция 4: дневные (10-16 local) и ночные (22-04 local) агрегаты
# ---------------------------------------------------------------------------


def add_weather_features_4(df_snapshots, era5_hourly):
    """
    дневные (10-16 local) и ночные (22-04 local) min/max/mean
    для тех же окон, что в п.2 и п.3.
    """
    era5 = era5_hourly.copy()
    if "datetime_utc" not in era5.columns:
        era5 = prepare_era5(era5)

    snaps = prepare_snapshots(df_snapshots)
    snap_ts = snaps["datetime_utc"].values.astype("datetime64[s]").astype(np.int64)
    snap_ids = snaps["ID"].values

    rolling_spec = [
        (d * 24 * SEC_PER_HOUR, 0, f"last_{d}d") for d in ROLLING_WINDOWS_DAYS
    ]
    lag_spec = [
        (lag * 24 * SEC_PER_HOUR, (lag - 1) * 24 * SEC_PER_HOUR, f"lag{lag}")
        for lag in LAG_DAYS
    ]
    window_spec = rolling_spec + lag_spec

    masks = {"is_daytime": "day", "is_nighttime": "night"}

    aggs = _compute_windowed_aggs(
        era5, snap_ts, snap_ids, window_spec, WEATHER_COLS, masks
    )

    new_df = pd.DataFrame(aggs, index=df_snapshots.index)
    return pd.concat([df_snapshots, new_df], axis=1)


# ---------------------------------------------------------------------------
# универсальная обёртка
# ---------------------------------------------------------------------------


def add_all_weather_features(df_snapshots, era5_hourly):
    """применяет все 4 функции последовательно."""
    era5 = prepare_era5(era5_hourly)
    print("  era5 подготовлен")

    t0 = time.time()
    df = add_weather_features_1(df_snapshots, era5)
    print(
        f"  ф-ция 1 (nearest hour):  +{df.shape[1] - df_snapshots.shape[1]} колонок, {time.time()-t0:.1f}с"
    )

    t0 = time.time()
    n1 = df.shape[1]
    df = add_weather_features_2(df, era5)
    print(f"  ф-ция 2 (rolling):      +{df.shape[1]-n1} колонок, {time.time()-t0:.1f}с")

    t0 = time.time()
    n2 = df.shape[1]
    df = add_weather_features_3(df, era5)
    print(f"  ф-ция 3 (lags):         +{df.shape[1]-n2} колонок, {time.time()-t0:.1f}с")

    t0 = time.time()
    n3 = df.shape[1]
    df = add_weather_features_4(df, era5)
    print(f"  ф-ция 4 (day/night):    +{df.shape[1]-n3} колонок, {time.time()-t0:.1f}с")

    return df


if __name__ == "__main__":
    import zipfile

    print("загрузка данных...")
    t_start = time.time()

    df_snapshots = pd.read_csv("landsat_data_2019_2025.csv")
    print(f"  landsat: {df_snapshots.shape}")

    with zipfile.ZipFile("join_every_hour_era5_data.zip") as z:
        era5_raw = pd.read_csv(z.open(z.namelist()[0]))
    print(f"  era5:    {era5_raw.shape}")

    print("\nвычисление погодных фичей...")
    t_start = time.time()
    df = add_all_weather_features(df_snapshots, era5_raw)

    total_new = df.shape[1] - df_snapshots.shape[1]
    print(f"\nитого: {total_new} погодных колонок")
    print(f"финальный размер: {df.shape}")
    print(f"общее время: {time.time()-t_start:.1f}с")

    out = "landsat_with_weather_features.csv"
    df.to_csv(out, index=False)
    print(f"\nсохранено: {out}")
