#!/usr/bin/env python3
import sys
import math
from pathlib import Path

import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView

# ──────────────────────────────────────────────────────────
# 사용자 지정: 컬러맵 범위 (dB)
# ──────────────────────────────────────────────────────────
VMIN = 75
VMAX = 100.0

# ──────────────────────────────────────────────────────────
# 파일 경로
# ──────────────────────────────────────────────────────────
DEN_CSV  = Path(r"C:\Users\Junyong\TrafficSim\den_log.csv")
SHP_PATH = Path(r"C:\Users\Junyong\TrafficSim\Analysis\Sources\nlsp_020001001.shp")
VP_CSV   = Path(r"C:\Users\Junyong\TrafficSim\Monitoring\Sources\vertiport.csv")
WP_CSV   = Path(r"C:\Users\Junyong\TrafficSim\Monitoring\Sources\waypoint.csv")

# ──────────────────────────────────────────────────────────
# 1) den_log 읽기
# ──────────────────────────────────────────────────────────
den_df = (
    pd.read_csv(
        DEN_CSV, skiprows=1,
        names=["time","gid","Lden","count"],
        dtype={"gid":"Int64","Lden":"float"}
    )
    .sort_values("time")
    .groupby("gid", as_index=False)
    .last()[["gid","Lden"]]
)

# ──────────────────────────────────────────────────────────
# 2) grid 셰이프 읽기 & 인구수(val) 병합
# ──────────────────────────────────────────────────────────
grid = (
    gpd.read_file(SHP_PATH, encoding="cp949", errors="ignore")
       .to_crs(epsg=4326)
)
grid["gid"] = (
    grid["gid"].astype(str)
         .str.extract(r"(\d+)", expand=False)
         .astype("Int64")
)
# 'val' 컬럼이 인구수라 가정합니다. 없으면 실제 컬럼명으로 변경하세요.
# grid = grid.rename(columns={"POP_COL":"val"})
grid = grid.dropna(subset=["gid"])
gdf = grid.merge(den_df, on="gid", how="left")
gdf["geometry"] = gdf.geometry.simplify(tolerance=0.0005, preserve_topology=True)

# ──────────────────────────────────────────────────────────
# 3) 셀 반경 계산 (도→m, 70% 축소)
# ──────────────────────────────────────────────────────────
minx, miny, maxx, maxy = gdf.total_bounds
n = len(gdf)
deg2m = 111000
dx = (maxx - minx) / math.sqrt(n)
dy = (maxy - miny) / math.sqrt(n)
r_m = math.sqrt((dx*deg2m/2)**2 + (dy*deg2m/2)**2) * 0.7

# ──────────────────────────────────────────────────────────
# 4) vertiport & waypoint 읽기
# ──────────────────────────────────────────────────────────
vp_df = pd.read_csv(
    VP_CSV, dtype={"INR(km)":float,"OTR(km)":float,"MTR(km)":float}
)
wp_df = pd.read_csv(WP_CSV, dtype=str)
wp_df["위도"] = wp_df["위도"].astype(float)
wp_df["경도"] = wp_df["경도"].astype(float)
coord_map = {
    r["Waypoint 명"]:(r["위도"], r["경도"])
    for _,r in wp_df.iterrows()
}

# ──────────────────────────────────────────────────────────
# 7) Folium Map 생성
# ──────────────────────────────────────────────────────────
m = folium.Map(
    location=[(miny + maxy) / 2, (minx + maxx) / 2],
    zoom_start=11,
    tiles="CartoDB Positron"
)

# 컬러맵
colormap = cm.LinearColormap(
    ["orange","red"], vmin=VMIN, vmax=VMAX,
    caption=f"L_den (dB) [{VMIN:.1f}–{VMAX:.1f}]"
)
m.add_child(colormap)

# ① 격자: VMIN≤Lden≤VMAX 인 경우만 그리기
for _, row in gdf.iterrows():
    L = row["Lden"]
    if pd.isna(L) or L < VMIN or L > VMAX:
        continue   # 범위 밖은 스킵
    pt = row.geometry.centroid
    folium.Circle(
        location=(pt.y, pt.x),
        radius=r_m,
        weight=0,
        fill=True,
        fill_color=colormap(L),
        fill_opacity=0.7,
        tooltip=f"L_den: {L:.1f} dB",
        **{"renderer": folium.features.Circle().options.get("renderer")}
    ).add_to(m)

# ② Vertiport 마커 & 반경
for _, r in vp_df.iterrows():
    lat, lon = r["위도"], r["경도"]
    folium.CircleMarker([lat, lon], radius=6,
                        color="#008080", fill=True, fill_opacity=1,
                        tooltip=f"Vertiport").add_to(m)
    for key,col in [("INR(km)","green"),("OTR(km)","red"),("MTR(km)","purple")]:
        km = r.get(key,0) or 0
        if km>0:
            folium.Circle([lat, lon], radius=km*1000,
                          color=col, weight=2, fill=False, opacity=0.5
            ).add_to(m)

# ③ Waypoint: CircleMarker + 링크
for _, r in wp_df.iterrows():
    lat, lon = r["위도"], r["경도"]
    folium.CircleMarker([lat, lon], radius=5,
                        color="blue", fill=True, fill_opacity=0.8,
                        tooltip=r["Waypoint 명"]
    ).add_to(m)
    links = r.get("Link","")
    if pd.notna(links):
        for tgt in map(str.strip, links.split(",")):
            if tgt in coord_map:
                folium.PolyLine(
                    locations=[coord_map[r["Waypoint 명"]], coord_map[tgt]],
                    color="blue", weight=1, opacity=0.7
                ).add_to(m)

# ──────────────────────────────────────────────────────────
# 6) PyQt5 창에 바로 렌더링
# ──────────────────────────────────────────────────────────
app = QApplication(sys.argv)
view = QWebEngineView()
view.setHtml(m.get_root().render())
view.setWindowTitle("L_den + Population Tooltip Map")
view.resize(1200,800)
view.show()
sys.exit(app.exec_())
