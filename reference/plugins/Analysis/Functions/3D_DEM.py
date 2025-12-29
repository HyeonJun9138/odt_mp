#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_dem_3d_plot_real_aspect.py ―
• DEM GeoTIFF 2개 읽어 병합
• 원위치 위경도(°) 메쉬 생성
• Z축 과장(vertical_exag)
• 경도·위도 차이를 물리 거리(미터)로 변환해 box aspect 자동 계산
• 숫자 눈금 제거, 축 이름만 표시
"""

import os, sys
import rasterio
from rasterio.merge import merge
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------ 사용자 설정 ------------
dem_files = [
    r"C:\Users\Junyong\TrafficSim\Analysis\Sources\n37_e126_1arc_v3.tif",
    r"C:\Users\Junyong\TrafficSim\Analysis\Sources\n37_e127_1arc_v3.tif",
]
downsample     = 3     # 렌더 해상도 조절
vertical_exag  = 15.0  # Z축 과장 배율 (1.0 = 원본 높이)
# --------------------------------------

def read_and_merge(paths):
    srcs = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Error: 파일을 찾을 수 없습니다: {p}", file=sys.stderr)
            sys.exit(1)
        srcs.append(rasterio.open(p))
    merged, tf = merge(srcs)
    return merged[0], tf

def make_lonlat_mesh(dem, tf):
    """
    dem: 2D array, tf: affine transform
    return lon, lat in degrees
    """
    rows, cols = dem.shape
    j, i = np.meshgrid(np.arange(cols), np.arange(rows))
    lon = tf.c + i * tf.a + j * tf.b
    lat = tf.f + i * tf.d + j * tf.e
    return lon, lat

def plot_real_aspect(dem, lon, lat):
    # Z축 과장
    Z = dem * vertical_exag

    # 물리적 거리(미터)로 변환
    # 1° 위도 ≈ 111 km, 1° 경도 ≈ 111 km * cos(평균 위도)
    avg_lat_rad = np.deg2rad(lat.mean())
    meters_per_deg_lat = 111_000
    meters_per_deg_lon = 111_000 * np.cos(avg_lat_rad)

    x_min, x_max = lon.min(), lon.max()
    y_min, y_max = lat.min(), lat.max()
    xr = (x_max - x_min) * meters_per_deg_lon
    yr = (y_max - y_min) * meters_per_deg_lat
    zr = (Z.max() - Z.min())

    # 시각적 비율: lon축=1, lat축=yr/xr, z축=zr/xr
    aspect = (1.0, yr/xr, zr/xr)

    fig = plt.figure(figsize=(12,9))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        lon[::downsample, ::downsample],
        lat[::downsample, ::downsample],
        Z  [::downsample, ::downsample],
        rcount=300, ccount=300,
        cmap='viridis', shade=True,
        linewidth=0, antialiased=True
    )

    # 보정된 축 비율 적용
    ax.set_box_aspect(aspect)

    # ─── 경도·위도 축에만 눈금 표시 ───
    # 경도 눈금: lon.min() 부터 lon.max() 까지 5개
    xt = np.linspace(lon.min(), lon.max(), 5)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{x:.2f}°" for x in xt])
    # 위도 눈금: lat.min() 부터 lat.max() 까지 5개
    yt = np.linspace(lat.min(), lat.max(), 5)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:.2f}°" for y in yt])

    # 고도 축 눈금은 제거
    ax.set_zticks([])

    # ─── 축 이름만 표시 ───
    ax.set_xlabel("Longitude", labelpad=10)
    ax.set_ylabel("Latitude",  labelpad=10)
    ax.set_zlabel("Elevation", labelpad=10)

    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dem, tf       = read_and_merge(dem_files)
    lon, lat      = make_lonlat_mesh(dem, tf)
    plot_real_aspect(dem, lon, lat)





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# plot_population_grid_simple.py ― 인구 격자(shapefile)를
# 연속형 컬러맵으로 표시 (mapclassify 불필요)
# """

# import os
# import sys
# import geopandas as gpd
# import matplotlib.pyplot as plt

# # 자신의 환경에 맞춰 경로 수정
# SHAPEFILE = r"C:\Users\Junyong\TrafficSim\Analysis\Sources\nlsp_020001001.shp"

# def plot_population_grid(shp_path: str, pop_field: str = "val"):
#     # 1) shapefile 읽기
#     if not os.path.exists(shp_path):
#         print(f"Error: 파일이 없습니다: {shp_path}", file=sys.stderr)
#         sys.exit(1)

#     gdf = gpd.read_file(shp_path)

#     # 2) 인구 필드 검사
#     if pop_field not in gdf.columns:
#         print(f"Error: '{pop_field}' 필드가 shapefile에 없습니다.", file=sys.stderr)
#         print("사용 가능한 필드:", gdf.columns.tolist(), file=sys.stderr)
#         sys.exit(1)

#     # 3) 플롯 (continuous cmap)
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     gdf.plot(
#         column=pop_field,
#         ax=ax,
#         cmap="OrRd",       # Orange-Red 연속형
#         edgecolor="black",
#         linewidth=0.2,
#         legend=True,
#         legend_kwds={"label": "Population", "shrink": 0.6}
#     )

#     ax.set_title("Population Grid")
#     ax.set_axis_off()  # 축 눈금 제거

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     plot_population_grid(SHAPEFILE, pop_field="val")
