"""
Exploratory Data Analysis (EDA) Suite for PathSnap Trace-Lite
Performs missing value analysis, observation counts, graph connectivity,
timestamp gap analysis, and generates publication plots and Folium maps.
"""
import json
from pathlib import Path
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import TRAIN_DIR, TEST_DIR, EDA_DIR, MAPS_DIR
from src.data_loader import load_trip, list_trips

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def analyze_dataset_partitions():
    """
    Analyze all 80 train trips and 10 student test trips.
    Compute observation counts, node counts, arc counts, timestamp gaps, missing values.
    """
    train_trips = list_trips("train")
    test_trips = list_trips("student_test")
    
    records = []
    
    print(f"Analyzing {len(train_trips)} train trips...")
    for idx, tid in enumerate(train_trips):
        trip = load_trip(tid, "train")
        T = len(trip.track_times)
        time_diffs = np.diff(trip.track_times) if T > 1 else np.array([0.0])
        
        # Missing values check
        nan_lons = int(np.isnan(trip.track_lons).sum())
        nan_lats = int(np.isnan(trip.track_lats).sum())
        nan_times = int(np.isnan(trip.track_times).sum())
        
        # Degree connectivity
        in_degrees = np.bincount(trip.arcs_to, minlength=trip.num_nodes)
        out_degrees = np.bincount(trip.arcs_from, minlength=trip.num_nodes)
        total_degrees = in_degrees + out_degrees
        isolated_nodes = int(np.sum(total_degrees == 0))
        
        records.append({
            "trip_id": tid,
            "partition": "train",
            "gps_observations": T,
            "num_nodes": trip.num_nodes,
            "num_arcs": trip.num_arcs,
            "ref_route_length": len(trip.reference_route) if trip.reference_route is not None else 0,
            "missing_values_count": nan_lons + nan_lats + nan_times,
            "time_gap_mean": float(np.mean(time_diffs)) if len(time_diffs) > 0 else 0.0,
            "time_gap_max": float(np.max(time_diffs)) if len(time_diffs) > 0 else 0.0,
            "time_gap_1s_pct": float(np.mean(np.abs(time_diffs - 1.0) < 0.05)) * 100.0 if len(time_diffs) > 0 else 100.0,
            "isolated_nodes": isolated_nodes,
            "density_arcs_per_node": float(trip.num_arcs / trip.num_nodes) if trip.num_nodes > 0 else 0.0
        })

    print(f"Analyzing {len(test_trips)} student-test trips...")
    for idx, tid in enumerate(test_trips):
        trip = load_trip(tid, "student_test")
        T = len(trip.track_times)
        time_diffs = np.diff(trip.track_times) if T > 1 else np.array([0.0])
        
        nan_lons = int(np.isnan(trip.track_lons).sum())
        nan_lats = int(np.isnan(trip.track_lats).sum())
        nan_times = int(np.isnan(trip.track_times).sum())
        
        in_degrees = np.bincount(trip.arcs_to, minlength=trip.num_nodes)
        out_degrees = np.bincount(trip.arcs_from, minlength=trip.num_nodes)
        total_degrees = in_degrees + out_degrees
        isolated_nodes = int(np.sum(total_degrees == 0))
        
        records.append({
            "trip_id": tid,
            "partition": "student_test",
            "gps_observations": T,
            "num_nodes": trip.num_nodes,
            "num_arcs": trip.num_arcs,
            "ref_route_length": len(trip.reference_route) if trip.reference_route is not None else 0,
            "missing_values_count": nan_lons + nan_lats + nan_times,
            "time_gap_mean": float(np.mean(time_diffs)) if len(time_diffs) > 0 else 0.0,
            "time_gap_max": float(np.max(time_diffs)) if len(time_diffs) > 0 else 0.0,
            "time_gap_1s_pct": float(np.mean(np.abs(time_diffs - 1.0) < 0.05)) * 100.0 if len(time_diffs) > 0 else 100.0,
            "isolated_nodes": isolated_nodes,
            "density_arcs_per_node": float(trip.num_arcs / trip.num_nodes) if trip.num_nodes > 0 else 0.0
        })

    df = pd.DataFrame(records)
    csv_path = EDA_DIR / "trips_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved trips statistics to {csv_path}")

    # Aggregated summary
    train_df = df[df["partition"] == "train"]
    test_df = df[df["partition"] == "student_test"]

    summary = {
        "train": {
            "total_trips": len(train_df),
            "total_gps_points": int(train_df["gps_observations"].sum()),
            "mean_gps_points_per_trip": float(train_df["gps_observations"].mean()),
            "median_gps_points_per_trip": float(train_df["gps_observations"].median()),
            "min_gps_points_per_trip": int(train_df["gps_observations"].min()),
            "max_gps_points_per_trip": int(train_df["gps_observations"].max()),
            "mean_nodes": float(train_df["num_nodes"].mean()),
            "mean_arcs": float(train_df["num_arcs"].mean()),
            "total_missing_values": int(train_df["missing_values_count"].sum()),
            "mean_time_gap_1s_pct": float(train_df["time_gap_1s_pct"].mean()),
            "max_time_gap_across_all": float(train_df["time_gap_max"].max()),
            "mean_isolated_nodes": float(train_df["isolated_nodes"].mean())
        },
        "student_test": {
            "total_trips": len(test_df),
            "total_gps_points": int(test_df["gps_observations"].sum()),
            "mean_gps_points_per_trip": float(test_df["gps_observations"].mean()),
            "median_gps_points_per_trip": float(test_df["gps_observations"].median()),
            "min_gps_points_per_trip": int(test_df["gps_observations"].min()),
            "max_gps_points_per_trip": int(test_df["gps_observations"].max()),
            "mean_nodes": float(test_df["num_nodes"].mean()),
            "mean_arcs": float(test_df["num_arcs"].mean()),
            "total_missing_values": int(test_df["missing_values_count"].sum()),
            "mean_time_gap_1s_pct": float(test_df["time_gap_1s_pct"].mean()),
            "max_time_gap_across_all": float(test_df["time_gap_max"].max()),
            "mean_isolated_nodes": float(test_df["isolated_nodes"].mean())
        }
    }

    json_path = EDA_DIR / "eda_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved EDA summary to {json_path}")

    # Generate plots
    generate_eda_plots(df)
    return df, summary


def generate_eda_plots(df: pd.DataFrame):
    """Generate high-resolution EDA figures."""
    # 1. GPS observation counts & Node/Arc distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
    
    sns.histplot(df["gps_observations"], ax=axes[0], kde=True, color="#1f77b4", bins=20)
    axes[0].set_title("GPS Observations per Trip", fontweight="bold")
    axes[0].set_xlabel("Number of GPS Observations")
    axes[0].set_ylabel("Trip Count")
    
    sns.scatterplot(data=df, x="num_nodes", y="num_arcs", hue="partition", ax=axes[1], palette="tab10", s=60, alpha=0.8)
    axes[1].set_title("Road Network Scale: Nodes vs Arcs", fontweight="bold")
    axes[1].set_xlabel("Number of Nodes")
    axes[1].set_ylabel("Number of Directed Arcs")
    
    sns.boxplot(data=df, x="partition", y="time_gap_1s_pct", ax=axes[2], palette="Set2")
    axes[2].set_title("Sampling Uniformity (% 1s Intervals)", fontweight="bold")
    axes[2].set_xlabel("Dataset Partition")
    axes[2].set_ylabel("% Observations with Δt ≈ 1.0s")
    
    plt.tight_layout()
    plot_path = EDA_DIR / "eda_distributions.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved distribution plot to {plot_path}")


from matplotlib.collections import LineCollection

def generate_trip_visualization(trip_id: str = "00000001", partition: str = "train"):
    """
    Generate GPS trajectory plot, road network slice, and interactive Folium map.
    """
    trip = load_trip(trip_id, partition)
    
    # Static GPS trajectory and road network plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    
    min_x, max_x = np.min(trip.track_x) - 400, np.max(trip.track_x) + 400
    min_y, max_y = np.min(trip.track_y) - 400, np.max(trip.track_y) + 400
    
    nearby_arcs = (
        (trip.x1 >= min_x) & (trip.x1 <= max_x) &
        (trip.y1 >= min_y) & (trip.y1 <= max_y)
    )
    
    sub_x1 = trip.x1[nearby_arcs]
    sub_y1 = trip.y1[nearby_arcs]
    sub_x2 = trip.x2[nearby_arcs]
    sub_y2 = trip.y2[nearby_arcs]
    
    # Fast LineCollection for road network
    lines = [((sub_x1[i], sub_y1[i]), (sub_x2[i], sub_y2[i])) for i in range(len(sub_x1))]
    lc = LineCollection(lines, colors="#bdc3c7", linewidths=0.8, alpha=0.6)
    ax.add_collection(lc)
        
    # Fast LineCollection for reference route
    if trip.reference_route is not None:
        ref_x1 = trip.x1[trip.reference_route]
        ref_y1 = trip.y1[trip.reference_route]
        ref_x2 = trip.x2[trip.reference_route]
        ref_y2 = trip.y2[trip.reference_route]
        ref_lines = [((ref_x1[i], ref_y1[i]), (ref_x2[i], ref_y2[i])) for i in range(len(ref_x1))]
        lc_ref = LineCollection(ref_lines, colors="#27ae60", linewidths=2.5, alpha=0.8)
        ax.add_collection(lc_ref)
            
    # Plot GPS observations
    ax.scatter(trip.track_x, trip.track_y, color="#e74c3c", s=12, label="Observed GPS Points", alpha=0.8, zorder=5)
    ax.scatter(trip.track_x[0], trip.track_y[0], color="#2980b9", s=90, marker="^", label="Trajectory Start", zorder=6)
    ax.scatter(trip.track_x[-1], trip.track_y[-1], color="#8e44ad", s=90, marker="s", label="Trajectory End", zorder=6)
    
    ax.set_title(f"Trip {trip_id} — GPS Trajectory vs Road Network & Reference Route", fontsize=12, fontweight="bold")
    ax.set_xlabel("Local Metric Easting (m)")
    ax.set_ylabel("Local Metric Northing (m)")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.legend(loc="upper right", frameon=True)
    ax.set_aspect("equal", "box")
    
    plt.tight_layout()
    static_plot_path = EDA_DIR / "gps_trajectory_plot.png"
    plt.savefig(static_plot_path)
    plt.close()
    print(f"Saved GPS trajectory static plot to {static_plot_path}")

    # Generate Representative Folium Map
    center_lat = float(np.mean(trip.track_lats))
    center_lon = float(np.mean(trip.track_lons))
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )
    
    # 1. Road network feature group (subsampled lines within bounding box)
    fg_roads = folium.FeatureGroup(name="Local Road Network", show=True)
    sub_lons1, sub_lats1 = trip.projector.unproject(sub_x1, sub_y1)
    sub_lons2, sub_lats2 = trip.projector.unproject(sub_x2, sub_y2)
    step = max(1, len(sub_x1) // 1000)
    road_coords = [[[float(sub_lats1[i]), float(sub_lons1[i])], [float(sub_lats2[i]), float(sub_lons2[i])]] for i in range(0, len(sub_x1), step)]
    folium.PolyLine(
        locations=road_coords,
        color="#7f8c8d",
        weight=1.5,
        opacity=0.5
    ).add_to(fg_roads)
    fg_roads.add_to(m)

    # 2. Reference Route feature group
    if trip.reference_route is not None:
        fg_ref = folium.FeatureGroup(name="Ground-Truth Reference Route", show=True)
        ref_lons1, ref_lats1 = trip.projector.unproject(ref_x1, ref_y1)
        ref_lons2, ref_lats2 = trip.projector.unproject(ref_x2, ref_y2)
        ref_coords = [[[float(ref_lats1[i]), float(ref_lons1[i])], [float(ref_lats2[i]), float(ref_lons2[i])]] for i in range(len(ref_x1))]
        folium.PolyLine(
            locations=ref_coords,
            color="#2ecc71",
            weight=4.0,
            opacity=0.85
        ).add_to(fg_ref)
        fg_ref.add_to(m)

    # 3. GPS observations feature group
    fg_gps = folium.FeatureGroup(name="Noisy GPS Observations", show=True)
    for t in range(0, len(trip.track_times), 2):
        folium.CircleMarker(
            location=[float(trip.track_lats[t]), float(trip.track_lons[t])],
            radius=3.5,
            color="#e74c3c",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.8,
            popup=f"Obs {t}: t={trip.track_times[t]:.1f}s"
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    folium_path = MAPS_DIR / "representative_folium_map.html"
    m.save(str(folium_path))
    print(f"Saved interactive Folium map to {folium_path}")


if __name__ == "__main__":
    analyze_dataset_partitions()
    generate_trip_visualization("00000001", "train")
