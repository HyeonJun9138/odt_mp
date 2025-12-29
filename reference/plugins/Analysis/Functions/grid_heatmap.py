#!/usr/bin/env python
"""
NoiseHeatmapPlayer  ―  CSV(log) ▸ 시간 슬라이더 ▸ 격자 히트맵
──────────────────────────────────────────────────────────────
• 입력 1) noise_log.csv   (sim_time,gid,max_dB,Leq_dB)
• 입력 2) nlsp_020001001.shp  (gid, geometry)
──────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import sys, os, json, tempfile
from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import folium, branca.colormap as cm

from PyQt5.QtCore    import Qt, QUrl
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QFileDialog,QRadioButton, QCheckBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView

# ──────────────────────────────────────────────────────────────
#  파일 경로 설정 (원하는 위치로 수정)
# ──────────────────────────────────────────────────────────────
CSV_PATH = Path(r"C:\Users\Junyong\TrafficSim\noise_log.csv")   # 로그 CSV
SHP_PATH = Path(r"C:\Users\Junyong\TrafficSim\Analysis\Sources\nlsp_020001001.shp")
VP_CSV = Path(r"C:\Users\Junyong\TrafficSim\Monitoring\Sources\vertiport.csv")
WP_CSV = Path(r"C:\Users\Junyong\TrafficSim\Monitoring\Sources\waypoint.csv")
# ──────────────────────────────────────────────────────────────
#  데이터 로드
# ──────────────────────────────────────────────────────────────

import pandas as pd
from pathlib import Path
def load_den_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, skiprows=1,
        names=["time","gid","Lden","cnt"],
        dtype={"gid":"Int64","Lden":"float"}
    )
    df = df.sort_values("time").groupby("gid", as_index=False).last()
    return df[["gid","Lden"]]

def load_data(csv_path: str, shp_path: str):
    df  = pd.read_csv(csv_path, dtype={"gid": "Int64"})

    gdf = gpd.read_file(shp_path, encoding="cp949", errors="ignore").to_crs(epsg=4326)

    # ── ★ gid 문자 → 숫자만 남기기 ──────────────────────────────
    gdf["gid"] = (
        gdf["gid"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)   # 첫 연속 숫자
        .astype("Int64")
    )
    gdf = gdf.dropna(subset=["gid"])          # 숫자 없는 행 제거

    merged = gdf.merge(df, on="gid")
    merged["sim_sec"] = (
        pd.to_timedelta(merged["sim_time"]).dt.total_seconds().astype(int)
    )
    return merged

# ──────────────────────────────────────────────────────────────
#  Folium map 생성
# ──────────────────────────────────────────────────────────────
def make_map(
    gdf: gpd.GeoDataFrame,
    value_col: str = "Leq_dB",
    vp_df: pd.DataFrame = None,
    wp_df: pd.DataFrame = None,
    wp_coord_map: dict | None = None,
    route_coords: list[list[float]] = None, draw_overlay: bool = True 
) -> folium.Map:
    m = folium.Map(location=[37.56, 126.97], zoom_start=11, tiles="CartoDB Positron")
    vals = gdf[value_col].dropna()
    vmin, vmax = (vals.min(), vals.max()) if not vals.empty else (0, 1)
    cmap = cm.linear.YlOrRd_09.scale(vmin, vmax)

    # ← 여기를 비워두면 안 됩니다!
    def style(feat):
        val = feat["properties"].get(value_col)
        if val is None or np.isnan(val):
            return {
                "fillColor": "#000000",
                "color":     "#666666",
                "weight":    0.3,
                "fillOpacity": 0.0
            }
        else:
            return {
                "fillColor": cmap(val),
                "color":     "#555555",
                "weight":    0.3,
                "fillOpacity": 0.7
            }

    folium.GeoJson(
        data=json.loads(gdf.to_json()),
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=["gid", value_col],
            aliases=["gid", "dB"],
            sticky=False
        )
    ).add_to(m)

    cmap.caption = f"{value_col} (dB)"
    m.add_child(cmap)
    if draw_overlay:
        if vp_df is not None:
            for _, vp in vp_df.iterrows():
                lat, lon = vp["위도"], vp["경도"]
                # 마커
                folium.CircleMarker([lat, lon], radius=6,
                                    color="orange", fill=True, fill_opacity=1).add_to(m)
                # 동심원
                for key, col in (("INR(km)","green"), ("OTR(km)","red"), ("MTR(km)","purple")):
                    r_km = vp.get(key, 0) or 0
                    if r_km > 0:
                        folium.Circle([lat, lon],
                                    radius=r_km*1000,
                                    color=col, weight=2, fill=False, opacity=0.4
                        ).add_to(m)
                # 부채꼴 함수
                def sector(lat0, lon0, r_m, brg):
                    pts = []
                    R = 6_371_000
                    for θ in np.linspace(brg-10, brg+10, 30):
                        θr = math.radians(θ)
                        φ0 = math.radians(lat0); λ0 = math.radians(lon0)
                        φ = math.asin(math.sin(φ0)*math.cos(r_m/R) +
                                    math.cos(φ0)*math.sin(r_m/R)*math.cos(θr))
                        λ = λ0 + math.atan2(
                            math.sin(θr)*math.sin(r_m/R)*math.cos(φ0),
                            math.cos(r_m/R)-math.sin(φ0)*math.sin(φ)
                        )
                        pts.append((math.degrees(φ), math.degrees(λ)))
                    return [(lat0, lon0)] + pts

                for deg_key, color in (("INR_Deg","green"), ("OTR_Deg","red")):
                    brg = vp.get(deg_key)
                    r_km = vp.get("MTR(km)",0) or 0
                    if brg is not None and r_km > 0:
                        poly = sector(lat, lon, r_km*1000, float(brg))
                        folium.Polygon(locations=poly,
                                    color=None, fill=True,
                                    fill_color=color, fill_opacity=0.2
                        ).add_to(m)

        if wp_df is not None and wp_coord_map is not None:
            for _, row in wp_df.iterrows():
                src_name = row["Waypoint 명"]
                src_latlon = wp_coord_map[src_name]
                # Link 컬럼: “A, B, C” 형태
                for tgt_name in map(str.strip, row["Link"].split(",")):
                    if tgt_name in wp_coord_map:
                        tgt_latlon = wp_coord_map[tgt_name]
                        folium.PolyLine(
                            locations=[src_latlon, tgt_latlon],
                            color="red", weight=2, opacity=0.7
                        ).add_to(m)
    return m

# ──────────────────────────────────────────────────────────────
#  PyQt5 위젯
# ──────────────────────────────────────────────────────────────
class NoiseHeatmapPlayer(QWidget):
    def __init__(self, csv_path=CSV_PATH, shp_path=SHP_PATH):
        super().__init__()
        self.setWindowTitle("Noise Heatmap Player")
        self.resize(1200, 800)

        self.data = load_data(csv_path, shp_path)
        self.times = sorted(self.data["sim_sec"].unique())
        self._idx = 0  # 슬라이더 위치

        # ── UI ───────────────────────────────────────────────
        layout = QHBoxLayout(self)
        self.web = QWebEngineView()
        layout.addWidget(self.web, stretch=1)

        vbox = QVBoxLayout()
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMaximum(len(self.times) - 1)
        self.slider.valueChanged.connect(self.update_map)
        vbox.addWidget(self.slider, stretch=1)

        self.lbl_time = QLabel("--:--:--")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.lbl_time)
        layout.addLayout(vbox)
        
        self.full_grid = gpd.read_file(shp_path, encoding="cp949", errors="ignore"
              ).to_crs(epsg=4326)
        self.full_grid["gid"] = (
            self.full_grid["gid"].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
        )
        self.full_grid = self.full_grid.dropna(subset=["gid"])
        self.vp_df = pd.read_csv(VP_CSV)
        self.wp_df = pd.read_csv(WP_CSV)

        self.wp_df = pd.read_csv(WP_CSV, dtype=str)
        # 좌표는 float로 변환
        self.wp_df["위도"] = self.wp_df["위도"].astype(float)
        self.wp_df["경도"] = self.wp_df["경도"].astype(float)

        # (2) 이름 → (lat, lon) 매핑
        self.wp_coord_map = {
            row["Waypoint 명"]: (row["위도"], row["경도"])
            for _, row in self.wp_df.iterrows()
        }
        self.route_coords = [ coord for name, coord in self.wp_coord_map.items() ]
        
        # ── Overlay 토글 체크박스 ───────────────────────
        self.chk_overlay = QCheckBox("Overlay On/Off")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(lambda _: self.update_map(self._idx))
        vbox.addWidget(self.chk_overlay)

        # 초기 지도
        self.update_map(0)
        

    # ──────────────────────────────────────────────────────────
    def update_map(self, idx: int):
        self._idx = idx
        sec = self.times[idx]
        hms = str(pd.to_timedelta(sec, unit="s"))
        self.lbl_time.setText(hms)

        subset = self.data[self.data["sim_sec"] == sec][["gid", "Leq_dB"]]

        # ② full_grid 와 left-join → 값 없는 셀은 NaN
        gdf = self.full_grid.merge(subset, on="gid", how="left")

        m = make_map(
        gdf,
        value_col="Leq_dB",               # 반드시 넘겨 줍니다
        vp_df=self.vp_df,
        wp_df=self.wp_df,
        wp_coord_map=self.wp_coord_map,    # waypoint 링크 맵
        route_coords=self.route_coords,     # waypoint 간 단순 연결이 필요하다면
    draw_overlay=self.chk_overlay.isChecked() 
    )

        # Folium HTML 임시파일 → QWebEngine
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        m.save(tmp.name)
        self.web.setUrl(QUrl.fromLocalFile(tmp.name))

    # ──────────────────────────────────────────────────────────
    def _on_mode_change(self):
        if self.btn_time.isChecked():
            self.slider.setEnabled(True)
            self.update_map(self.slider.value())     # 시간 모드 갱신
        else:
            self.slider.setEnabled(False)
            self.show_l_den_map()      
        
        
    def _load_map(self, m: folium.Map):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        m.save(tmp.name)
        self.web.setUrl(QUrl.fromLocalFile(tmp.name))

    def show_l_den_map(self):
        gdf = self.full_grid.merge(self.den_df, on="gid", how="left")
        m = make_map(
    gdf,
    value_col="L_den",
    vp_df=self.vp_df,
    wp_df=self.wp_df,
    wp_coord_map=self.wp_coord_map,
    route_coords=self.route_coords,
    draw_overlay=self.chk_overlay.isChecked()
)
        self._load_map(m)
        
        
    def _toggle_route(self, on):
        js = """
        if (window.routeLayer) { map.removeLayer(routeLayer); }
        if (%s) {
            routeLayer = L.polyline(%s, {color:'#1f78b4', weight:2}).addTo(map);
        }
        """ % ("true" if on else "false",
            json.dumps(self.route_coords))     # [[lat,lon], ...]
        self.web.page().runJavaScript(js)
# ──────────────────────────────────────────────────────────────
#  실행
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # CSV·SHP 파일 선택 다이얼로그 (기본값 유지 시 생략 가능)
    if not CSV_PATH.exists() or not SHP_PATH.exists():
        dbox = QFileDialog()
        csv_path, _ = dbox.getOpenFileName(
            None, "Select noise_log.csv", "", "CSV (*.csv)"
        )
        shp_path, _ = dbox.getOpenFileName(
            None, "Select grid shapefile", "", "SHP (*.shp)"
        )
    else:
        csv_path, shp_path = str(CSV_PATH), str(SHP_PATH)

    win = NoiseHeatmapPlayer(csv_path, shp_path)
    win.show()
    sys.exit(app.exec_())
