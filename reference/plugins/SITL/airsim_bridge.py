# -*- coding: utf-8 -*-
"""
airsim_bridge.py — SITL/CSV(LLA) → AirSim 브릿지 (Gate 스폰 + coordinateUR 좌표변환)

추가 기능 요약
  • Gate 스폰:
     - --gate-json : Gate 좌표 리소스(JSON, coordinateUR GUI: Vehicles JSON 권장)
     - --fpl-csv   : FPL CSV에서 vehicle_id(또는 acid)→Gate 번호 매핑 추출
     - 첫 스폰 시 Gate 좌표를 사용하여 정확히 스폰 (Yaw 적용)

  • LLA 좌표 변환:
     - snapshot에 x,y,z 없고 lon/lat/alt_m가 있으면 coordinateUR 변환으로 NED(m) 산출
     - 변환식/스케일/부호는 coordinateUR와 동일(PLAYERSTART_Z 기준)

하위호환:
  • 기존 SitlSim(snapshot의 x,y,z m / heading_deg)은 그대로 지원
"""

import sys, math, time, re, json, csv, argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import airsim

# 원본과 동일: 프로젝트 경로 설정
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(THIS_DIR))

# ── SITL (그대로 사용) ─────────────────────────────────────────────
from sitl_sim import SitlSim  # noqa

# ── coordinateUR / coordinate_gui (좌표변환 그대로 사용) ─────────────
# ※ coordinateUR 모듈명이 다를 수 있어 양쪽 시도
try:
    from .sitl_coord_transform import (
        AffineGeoToUEMapper, CITY_HALL_UE, PLAYERSTART_Z_UE_CM, GCP_LIST
    )
except Exception:
    try:
        from sitl_coord_transform import (
            AffineGeoToUEMapper, CITY_HALL_UE, PLAYERSTART_Z_UE_CM, GCP_LIST
        )
    except Exception:
        from coordinateUR import (
            AffineGeoToUEMapper, CITY_HALL_UE, PLAYERSTART_Z_UE_CM, GCP_LIST
        )

# AirSim client version 우회(원본 유지)
airsim.VehicleClient.getClientVersion = lambda self: 4

# ── 사용자 파라미터 (필요시 조정) ────────────────────────────────────
def _resolve_default_fpl_folder() -> Path:
    base = ROOT_DIR / "Scheduler" / "FPL_Result"
    if base.exists():
        folders = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)
        if folders:
            return folders[0]
    return base


FPL_FOLDER_DEFAULT = str(_resolve_default_fpl_folder())
SIM_START_HHMM     = "06:30"
SIM_SPEED          = 5.0
UPDATE_HZ          = 10.0

PHYSICS_ENABLED    = False
FALLBACK_TO_BP     = True

# 좌표/방위 보정(원본 유지)
WORLD_OFFSET_N     = 0.0
WORLD_OFFSET_E     = 0.0
WORLD_OFFSET_D     = -16.0
YAW_BIAS_DEG       = 0.0

# 헤딩/롤 스무딩(원본 유지)
YAW_TAU            = 0.50
YAW_RATE_MAX       = math.radians(60.0)
BANK_ENABLE        = False
BANK_MAX_DEG       = 8.0
BANK_TAU           = 0.50
G                  = 9.80665

# AirSim 에셋 후보(원본 유지)
STATICMESH_CANDIDATES = [
    "KP2StaticMesh",
    "/Game/KP2A/KP2StaticMesh.KP2StaticMesh",
    "StaticMesh'/Game/KP2A/KP2StaticMesh.KP2StaticMesh'",
]
BLUEPRINT_CANDIDATES = [
    "KP2_Blueprint",
    "Blueprint'/Game/KP2/KP2_Blueprint.KP2_Blueprint'",
    "/Game/KP2/KP2_Blueprint.KP2_Blueprint_C",
]


# ───────────────────────────────────────────────────────────────
# 1) 좌표 변환기: coordinateUR 로직을 그대로 래핑 (LLA → NED m)
#    - AffineGeoToUEMapper(GCP_LIST)로 절대 UE(cm) 산출
#    - CityHall UE를 빼서 CityHall-origin(cm) → X/Y: /100 → m
#    - Z: PLAYERSTART_Z 기준 NED(m)  (coordinateUR JSON 저장 로직 동일)
#    참고: coordinateUR GUI 코드.  (LLA→UE, CityHall, PlayerStart Z)  :contentReference[oaicite:4]{index=4}
# ───────────────────────────────────────────────────────────────
class LLAConverter:
    def __init__(self):
        self.mapper = AffineGeoToUEMapper(GCP_LIST)
        self._chx, self._chy, self._chz = [float(v) for v in CITY_HALL_UE]

        # 미세 정합 파라미터 (XY는 유사변환: scale+rotation+translation)
        self._xy_R = np.eye(2, dtype=float)  # 2x2 rotation
        self._xy_s = 1.0                     # scale
        self._xy_t = np.zeros(2, dtype=float)  # translation [N,E]
        self.d_bias_m = 0.0                  # Z(Down+) 상수 보정

    def lla_to_ned(self, lat: float, lon: float, alt_m: float = 0.0):
        ue_abs = self.mapper.geodetic_to_ue(lat, lon, alt_m)  # [cm]
        x_cm, y_cm, z_abs_cm = float(ue_abs[0]), float(ue_abs[1]), float(ue_abs[2])
        N = (x_cm - self._chx) / 100.0
        E = (y_cm - self._chy) / 100.0
        D = -((z_abs_cm - float(PLAYERSTART_Z_UE_CM)) / 100.0)
        return N, E, D

    # XY 유사변환 적용
    def _apply_xy_refine(self, n: float, e: float):
        v = np.array([n, e], dtype=float)
        w = self._xy_s * (self._xy_R @ v) + self._xy_t
        return float(w[0]), float(w[1])

    # AGL(m) → NED:  D = (D_ground) - AGL  ; D_ground = d0 + d_bias
    def ned_from_lla_agl(self, lat: float, lon: float, alt_agl_m: float):
        n0, e0, d0 = self.lla_to_ned(lat, lon, 0.0)  # alt=0 기준의 Down
        n1, e1 = self._apply_xy_refine(n0, e0)
        d_ground = d0 + self.d_bias_m
        d = d_ground - float(alt_agl_m)
        return n1, e1, d

    # resources_vp.csv로 XY/Z 자동 보정 (최소 2점이면 XY, 1점이면 Z만)
    def calibrate_from_vp_csv(self, csv_path: str):
        import csv
        P = []  # 예측 XY (LLA→UE→CityHall)
        A = []  # 실제 XY (resources_vp.csv X_m,Y_m)
        dz_list = []  # Z 보정 후보 (Z_m - d0)

        with open(csv_path, newline="", encoding="utf-8-sig") as fp:
            rdr = csv.DictReader(fp)
            for row in rdr:
                try:
                    lat = float(row.get("pt_lat_deg")) if row.get("pt_lat_deg") else None
                    lon = float(row.get("pt_lon_deg")) if row.get("pt_lon_deg") else None
                    xm  = row.get("X_m"); ym = row.get("Y_m"); zm = row.get("Z_m")
                    x_m = float(xm) if xm not in (None, "") else None
                    y_m = float(ym) if ym not in (None, "") else None
                    z_m = float(zm) if zm not in (None, "") else None
                    if (lat is None) or (lon is None):
                        continue
                    n0, e0, d0 = self.lla_to_ned(lat, lon, 0.0)
                    if (x_m is not None) and (y_m is not None):
                        P.append([n0, e0]); A.append([x_m, y_m])
                    if (z_m is not None):
                        dz_list.append(z_m - d0)
                except Exception:
                    continue

        # Z 바이어스: 평균
        if dz_list:
            self.d_bias_m = float(np.mean(dz_list))

        # XY 유사변환(Procrustes): A ≈ s R P + t
        if len(P) >= 2:
            P = np.asarray(P, dtype=float); A = np.asarray(A, dtype=float)
            Pc = P - P.mean(axis=0, keepdims=True)
            Ac = A - A.mean(axis=0, keepdims=True)
            H = Pc.T @ Ac
            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt
            s = float(np.trace(np.diag(S)) / np.sum(Pc * Pc))
            t = A.mean(axis=0) - s * (R @ P.mean(axis=0))
            self._xy_R = R
            self._xy_s = s
            self._xy_t = t


# ───────────────────────────────────────────────────────────────
# 2) Gate 리소스 로더 + 스폰 매퍼
#    - Vehicles JSON (coordinateUR GUI의 내보내기)를 우선 지원
#      * { "Vehicles": { "<Port>_GATE1": {"X":m, "Y":m, "Z":m, "Yaw":deg}, ... } }
#    - 다른 포맷(resource_vp)이면, _load_gate_positions() 내부 매핑만 바꾸면 됨.
#      (키 정규화: "GATE 1"→"GATE1", 대소문자/공백/언더스코어 무시)
# ───────────────────────────────────────────────────────────────
def _norm_gate_key(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9_]", "", str(s).upper())
    s2 = s2.replace("GATE_", "GATE")
    return s2

class GateSpawnResolver:
    def __init__(self,
                 gate_json: Optional[Path] = None,
                 fpl_csv: Optional[Path] = None,
                 gate_csv: Optional[Path]   = None):
        # gate_key → (N,E,D,YawDeg)
        self.gate_pos: Dict[str, Tuple[float, float, float, float]] = {}
        # acid → gate_key (또는 port_gate_key)
        self.acid_to_gate: Dict[str, str] = {}

        if gate_json:
            self._load_gate_positions(gate_json)
        if gate_csv:
            self._load_gate_positions_csv(gate_csv)
                
        if fpl_csv:
            self._load_acid_gate_map(fpl_csv)

    def _load_gate_positions(self, jp: Path):
        data = json.loads(Path(jp).read_text(encoding="utf-8"))
        # ① coordinateUR Vehicles JSON 패턴 우선 지원
        if isinstance(data, dict) and "Vehicles" in data and isinstance(data["Vehicles"], dict):
            for k, v in data["Vehicles"].items():
                try:
                    x = float(v.get("X", 0.0))
                    y = float(v.get("Y", 0.0))
                    z = float(v.get("Z", 0.0))
                    yaw = float(v.get("Yaw", 0.0)) if "Yaw" in v else 0.0
                    self.gate_pos[_norm_gate_key(k)] = (x, y, z, yaw)
                    # 접미사만으로도 접근 가능하도록 (…_GATE1 → GATE1)
                    tail = re.sub(r"^.*(_GATE\d+)$", r"\1", _norm_gate_key(k))
                    if tail != _norm_gate_key(k):
                        self.gate_pos.setdefault(tail, (x, y, z, yaw))
                except Exception:
                    continue
            return

        # ② 기타 포맷(resource_vp 등) — 여기에 키/스키마를 맞춰 매핑하세요.
        # 예시:
        # {
        #   "Gimpo": {"GATE1": {"X":..,"Y":..,"Z":..,"Yaw":..}, ...},
        #   "Jamsil": {"GATE1": {...}, ...}
        # }
        for port, d in (data or {}).items():
            if not isinstance(d, dict): continue
            for gk, v in d.items():
                try:
                    x = float(v.get("X", 0.0)); y = float(v.get("Y", 0.0))
                    z = float(v.get("Z", 0.0)); yaw = float(v.get("Yaw", 0.0))
                    key_full = _norm_gate_key(f"{port}_{gk}")
                    self.gate_pos[key_full] = (x, y, z, yaw)
                    self.gate_pos.setdefault(_norm_gate_key(gk), (x, y, z, yaw))
                except Exception:
                    continue
    def _load_gate_positions_csv(self, cp: Path):
        """
        resources_vp.csv 형식 지원
        필드 후보:
          - Vertiport / Port
          - Label (예: GATE 1, FATO 2)
          - X_m, Y_m, Z_m  (없으면 pt_lat_deg/pt_lon_deg로 LLA→UE 변환 후 XY 사용)
          - Yaw_deg (없으면 0)
        """
        try:
            import csv as _csv
        except Exception:
            return
        with open(cp, newline="", encoding="utf-8-sig") as fp:
            rdr = _csv.DictReader(fp)
            for row in rdr:
                try:
                    port = row.get("Vertiport") or row.get("Port") or ""
                    label = row.get("Label") or row.get("Name") or ""
                    if not port or not label:
                        continue
                    key_full = _norm_gate_key(f"{port}_{label}")
                    key_tail = _norm_gate_key(label)
                    # 좌표
                    def _f(k, default=None):
                        v = row.get(k)
                        try:
                            return float(v)
                        except Exception:
                            return default
                    x = _f("X_m"); y = _f("Y_m"); z = _f("Z_m")
                    yaw = _f("Yaw_deg", 0.0)
                    # X/Y/Z가 없으면 LLA로 보정(가능할 때만)
                    if (x is None or y is None):
                        lat = _f("pt_lat_deg"); lon = _f("pt_lon_deg")
                        if (lat is not None) and (lon is not None):
                            # LLA→UE(cm)→CityHall 오프셋 후 m (Z는 CSV가 주는 게 우선)
                            conv = LLAConverter()
                            n,e,d0 = conv.lla_to_ned(lat, lon, 0.0)
                            x,y = n,e
                            if z is None:
                                z = d0
                        else:
                            continue
                    if z is None:
                        z = 0.0
                    self.gate_pos[key_full] = (x, y, z, yaw)
                    self.gate_pos.setdefault(key_tail, (x, y, z, yaw))
                except Exception:
                    continue
    def _load_acid_gate_map(self, csvp: Path):
        """
        FPL CSV에서 ACID/vehicle_id와 Gate 번호/이름을 추출해 acid→gate_key 매핑.
        - 컬럼명 후보(대소문자 무시):
          * ACID 후보: 'vehicle_id','vehicle','acid','id','flight_number'
          * Gate 후보: 'gate','gate_no','gate_id','dep_gate','gate_name'
          * Port 후보: 'origin','depart_vertiport','port','vertiport'
        - 포트명이 있으면 '<Port>_GATE#' 키도 함께 시도.
        """
        with open(csvp, newline="", encoding="utf-8-sig") as fp:
            rdr = csv.reader(fp)
            header = next(rdr, [])
            h = [c.strip().lower() for c in header]

            def _col(*cands):
                for i, name in enumerate(h):
                    if name in [c.lower() for c in cands]:
                        return i
                return None

            ix_id = _col("vehicle_id", "vehicle", "acid", "id", "flight_number")
            ix_gate = _col("gate", "gate_no", "gate_id", "dep_gate", "gate_name")
            ix_port = _col("origin", "depart_vertiport", "port", "vertiport")

            if ix_id is None or ix_gate is None:
                # 컬럼 추정 실패 → 스킵(스폰은 기본 위치 사용)
                return

            for row in rdr:
                try:
                    acid = str(row[ix_id]).strip()
                    gv = str(row[ix_gate]).strip()
                    if not acid or not gv:
                        continue
                    # "GATE 1" → "GATE1"
                    # 숫자만 있는 경우에도 "GATE#"로 만듦
                    if re.fullmatch(r"\d+", gv):
                        gate_key = f"GATE{gv}"
                    else:
                        gate_key = gv
                    gate_key = _norm_gate_key(gate_key)

                    # 포트가 있으면 포트+게이트도 같이 보관
                    if ix_port is not None and row[ix_port]:
                        port = _norm_gate_key(str(row[ix_port]))
                        port_gate = _norm_gate_key(f"{port}_{gate_key}")
                        self.acid_to_gate[acid] = port_gate
                        # fallback: 순수 게이트 키도 같이 넣음
                        self.acid_to_gate.setdefault(acid, gate_key)
                    else:
                        self.acid_to_gate[acid] = gate_key
                except Exception:
                    continue

    def get_spawn_pose_for(self, acid: str):
        """acid에 대한 초기 스폰 pose(있으면) 반환. 없으면 None."""
        if not acid:
            return None
        key = self.acid_to_gate.get(acid)
        if not key:
            return None
        # 우선 port+gate로 찾고, 없으면 순수 gate로 찾기
        cand = [key, _norm_gate_key(re.sub(r"^.*(_GATE\d+)$", r"\1", key))]
        for k in cand:
            if k in self.gate_pos:
                x, y, z, yaw_deg = self.gate_pos[k]
                # AirSim Pose (NED → Vector3r(N,E,D))
                pos = airsim.Vector3r(x, y, z)
                quat = airsim.to_quaternion(0.0, 0.0, math.radians(yaw_deg))
                return airsim.Pose(pos, quat)
        return None


# ───────────────────────────────────────────────────────────────
# 3) AirSim 액터 상태/관리 (원본 유지 + 스폰 개선/LLA 변환 지원)
#    원본 구현: 위치 스무딩/teleport=False 등 유지. :contentReference[oaicite:5]{index=5}
# ───────────────────────────────────────────────────────────────
def sanitize_id(s):
    s2 = re.sub(r"[^A-Za-z0-9_]", "_", str(s))
    if not s2:
        s2 = "AC"
    if s2[0].isdigit():
        s2 = "AC_" + s2
    return s2[:60]

def angle_wrap_pi(x):
    return (x + math.pi) % (2*math.pi) - math.pi

def is_blueprint_ref(asset_name):
    return ("Blueprint'" in asset_name) or asset_name.endswith("_C")

class ActorState(object):
    def __init__(self, name, last_pos, last_yaw, last_t):
        self.name = name
        self.last_pos = last_pos   # (x,y,z) = (N,E,D)
        self.last_yaw = last_yaw   # rad
        self.last_t   = last_t
        self.speed_mps   = 0.0
        self.yawrate_rps = 0.0
        self.roll        = 0.0

class AirSimFleetBridge(object):
    def __init__(self, client,
                 lla_conv: Optional[LLAConverter] = None,
                 spawner: Optional[GateSpawnResolver] = None):
        self.client = client
        self.id_to_actor = {}     # acid -> ActorState
        self.used_names  = set()
        self.lla = lla_conv
        self.spawner = spawner
        self.dref_by_acid = {}   # ← 각 ACID의 기준 지면(Down+) 저장

        try:
            self._assets = self.client.simListAssets()
        except Exception:
            self._assets = []

    def _pick_asset(self):
        for cand in STATICMESH_CANDIDATES:
            for a in self._assets:
                if isinstance(a, str) and (cand in a):
                    return cand, False
        if STATICMESH_CANDIDATES:
            return STATICMESH_CANDIDATES[0], False
        if FALLBACK_TO_BP and BLUEPRINT_CANDIDATES:
            return BLUEPRINT_CANDIDATES[0], True
        return "KP2StaticMesh", False

    def _unique_name(self, base):
        name = base; i = 2
        while name in self.used_names:
            name = "%s_%d" % (base, i); i += 1
        self.used_names.add(name)
        return name

    def ensure_actor(self, acid, init_pose):
        if acid in self.id_to_actor:
            return self.id_to_actor[acid]

        base = "AC_%s" % sanitize_id(acid)
        name = self._unique_name(base)
        asset, bp = self._pick_asset()
        try:
            spawned_name = self.client.simSpawnObject(
                object_name=name,
                asset_name=asset,
                pose=init_pose,
                scale=airsim.Vector3r(1.0,1.0,1.0),
                physics_enabled=PHYSICS_ENABLED,
                is_blueprint=bp
            )
            name = spawned_name
        except Exception as e:
            print("[ERR] spawn failed for %s with %s → %s" % (name, asset, e))
            raise

        st = ActorState(
            name=name,
            last_pos=(init_pose.position.x_val,
                      init_pose.position.y_val,
                      init_pose.position.z_val),
            last_yaw=0.0,
            last_t=time.perf_counter()
        )
        self.id_to_actor[acid] = st
        print("[OK] spawned %s → %s  (asset=%s, blueprint=%s, physics=%s)"
              % (acid, name, asset, bp, PHYSICS_ENABLED))
        return st

    def update_actor(self, acid, posNEZ_or_LLA: Dict, now_t: float, heading_deg: float, use_bank=False):
        """
        posNEZ_or_LLA:
          - {x,y,z}   : N,E,D [m] (기존 SitlSim 출력)
          - {lon,lat,alt_m}: LLA → NED 변환 사용
        """
        is_lla = ("lon" in posNEZ_or_LLA) and ("lat" in posNEZ_or_LLA)
        alt_agl = float(posNEZ_or_LLA.get("alt_m", 0.0)) if is_lla else None
        # 0) 좌표 해석 (NED m)
        if "x" in posNEZ_or_LLA:
            n = float(posNEZ_or_LLA["x"]) + WORLD_OFFSET_N
            e = float(posNEZ_or_LLA["y"]) + WORLD_OFFSET_E
            d = float(posNEZ_or_LLA["z"]) + WORLD_OFFSET_D
        elif is_lla and self.lla:
            lat = float(posNEZ_or_LLA["lat"])
            lon = float(posNEZ_or_LLA["lon"])
            # XY는 LLA→UE 기반, D는 '기준 지면 D_ref - AGL'
            n, e, _ = self.lla.lla_to_ned(lat, lon, 0.0)
            n += WORLD_OFFSET_N; e += WORLD_OFFSET_E
            if acid in self.dref_by_acid:
                d = self.dref_by_acid[acid] - float(alt_agl) + WORLD_OFFSET_D
            else:
                # 임시 fallback: 지면고를 모르면 alt=0 기준 D0에서 AGL을 빼서 사용
                d0 = self.lla.lla_to_ned(lat, lon, 0.0)[2]
                d = d0 - float(alt_agl) + WORLD_OFFSET_D            
        else:
            # 정보 부족 → 스킵
            return

        # 1) 스폰(최초) — Gate 지정이 있으면 Gate pose 우선
        st = self.id_to_actor.get(acid)
        if not st:
            # Gate 스폰 우선
            pose0 = None
            if self.spawner:
                pose0 = self.spawner.get_spawn_pose_for(acid)
            if pose0 is None:
                yaw0 = math.radians(heading_deg + YAW_BIAS_DEG)
                pose0 = airsim.Pose(airsim.Vector3r(n, e, d),
                                    airsim.to_quaternion(0.0, 0.0, yaw0))
            st = self.ensure_actor(acid, pose0)
            st.last_yaw = math.radians(heading_deg + YAW_BIAS_DEG)
            # 스폰 직후 D_ref 고정 (Gate 스폰이면 그 Z, 아니면 현재 Z+AGL)
            if acid not in self.dref_by_acid:
                if pose0 is not None:
                    self.dref_by_acid[acid] = pose0.position.z_val
                else:
                    # LLA로 스폰했으면 현재 D + AGL을 더해 기준 지면으로 설정
                    self.dref_by_acid[acid] = d + (float(alt_agl) if alt_agl is not None else 0.0)
            return

        # 2) 업데이트(원본 스무딩 로직 유지)
        dt_s = max(1e-6, now_t - st.last_t)
        x0,y0,z0 = st.last_pos
        dx, dy, dz = (n-x0), (e-y0), (d-z0)
        spd = math.sqrt(dx*dx + dy*dy + dz*dz) / dt_s

        yaw_tgt = math.radians(heading_deg + YAW_BIAS_DEG)
        dyaw = angle_wrap_pi(yaw_tgt - st.last_yaw)
        alpha_yaw = max(0.0, min(1.0, dt_s / max(YAW_TAU, 1e-3)))
        yaw_cmd   = st.last_yaw + alpha_yaw * dyaw
        yaw_rate  = angle_wrap_pi(yaw_cmd - st.last_yaw) / dt_s
        if abs(yaw_rate) > YAW_RATE_MAX:
            yaw_rate = math.copysign(YAW_RATE_MAX, yaw_rate)
            yaw_cmd  = angle_wrap_pi(st.last_yaw + yaw_rate * dt_s)

        roll = st.roll
        if use_bank and BANK_ENABLE:
            phi_tgt = math.atan(max(0.0, spd) * abs(yaw_rate) / G)
            if yaw_rate < 0:  # 우회전 음수
                phi_tgt = -phi_tgt
            phi_tgt = max(-math.radians(BANK_MAX_DEG),
                          min(math.radians(BANK_MAX_DEG), phi_tgt))
            a_bank  = max(0.0, min(1.0, dt_s / max(BANK_TAU, 1e-3)))
            roll    = roll + a_bank*(phi_tgt - roll)

        pose = airsim.Pose(airsim.Vector3r(n, e, d),
                           airsim.to_quaternion(0.0, roll, yaw_cmd))
        self.client.simSetObjectPose(st.name, pose, teleport=False)

        st.last_pos    = (n,e,d)
        st.last_yaw    = yaw_cmd
        st.last_t      = now_t
        st.speed_mps   = spd
        st.yawrate_rps = yaw_rate
        st.roll        = roll

    def prune_actors(self, alive_ids):
        for acid in list(self.id_to_actor.keys()):
            if acid not in alive_ids:
                try:
                    self.client.simDestroyObject(self.id_to_actor[acid].name)
                except Exception:
                    pass
                self.used_names.discard(self.id_to_actor[acid].name)
                self.id_to_actor.pop(acid, None)


# ───────────────────────────────────────────────────────────────
# 4) 메인 루프 (원본 유지 + 옵션 추가)
# ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpl-folder", default=FPL_FOLDER_DEFAULT)
    ap.add_argument("--start", default=SIM_START_HHMM)
    ap.add_argument("--speed", type=float, default=SIM_SPEED)
    ap.add_argument("--hz", type=float, default=UPDATE_HZ)
    ap.add_argument("--gate-json", type=str, default=None,
                    help="Gate 좌표 리소스 JSON")
    ap.add_argument("--pre-spawn-min", type=float, default=1.0,
                    help="STD 몇 분 전 미리 스폰")
    ap.add_argument("--gate-csv", type=str, default=None,
                   help="resources_vp.csv (pt_lat_deg/pt_lon_deg, X_m/Y_m/Z_m로 XY/Z 캘리브레이션)")
    ap.add_argument("--alt-bias-m", type=float, default=None,
                    help="Z(Down+) 상수 보정값 수동 지정(양수=더 아래). gate-csv 없을 때 유용")
    args = ap.parse_args()

    # SITL
    sim = SitlSim(args.fpl_folder, dt_sim=1.0, pre_spawn_min=args.pre_spawn_min)
    sim.start(args.start, sim_speed=args.speed)

    # AirSim
    client = airsim.VehicleClient()
    client.confirmConnection()

    # 변환기/스포너
    lla_conv = LLAConverter()
    if args.gate_csv:
        try:
           lla_conv.calibrate_from_vp_csv(args.gate_csv)
           print(f"[CAL] XY refine & Z-bias from {args.gate_csv}  "
                 f"(d_bias_m={lla_conv.d_bias_m:.3f}, s={lla_conv._xy_s:.8f})")
        except Exception as e:
           print(f"[WARN] gate-csv calibration failed: {e}")
    if args.alt_bias_m is not None:
       lla_conv.d_bias_m = float(args.alt_bias_m)
       print(f"[CAL] Z-bias override: d_bias_m={lla_conv.d_bias_m:.3f} m")

    spawner = GateSpawnResolver(
        gate_json=Path(args.gate_json) if args.gate_json else None,
        fpl_csv=Path(getattr(args, "fpl_csv", "")) if getattr(args, "fpl_csv", None) else None
    )
    bridge = AirSimFleetBridge(client, lla_conv=lla_conv, spawner=spawner)

    target_dt = 1.0 / float(args.hz)
    t_prev = time.perf_counter()

    print("[RUN] SITL→AirSim bridge  (speed×%.1f, %.0f Hz)" % (args.speed, args.hz))
    while True:
        t_now = time.perf_counter()
        real_dt = t_now - t_prev
        if real_dt < target_dt:
            time.sleep(target_dt - real_dt)
            t_now = time.perf_counter()
            real_dt = t_now - t_prev
        t_prev = t_now

        # 1) SITL 한 틱
        sim.step(real_dt)

        # 2) 스냅샷 가져오기
        snap = sim.snapshot()  # {acid: {...}}
        alive_ids = set(snap.keys())

        # 3) 각 기체 업데이트
        for acid, pkt in snap.items():
            hdg = float(pkt.get("heading_deg", 0.0))
            atd = str(pkt.get("atd", "-"))
            pre_departure = (atd == "-")  # STD 이전
            has_lla = ("lon" in pkt) and ("lat" in pkt)
            alt_agl = float(pkt.get("alt_m", pkt.get("z", 0.0)))  # SITL AGL(m)

            # ① STD 이전: 게이트 스폰만 하고 이동은 막는다
            if pre_departure:
                if acid not in bridge.id_to_actor:
                    if has_lla:
                        bridge.update_actor(acid, {"lon":pkt["lon"],"lat":pkt["lat"],"alt_m":alt_agl}, t_now, hdg, use_bank=False)
                    elif all(k in pkt for k in ("x","y","z")):
                        zned = -float(pkt["z"])
                        bridge.update_actor(acid, {"x":pkt["x"],"y":pkt["y"],"z":zned}, t_now, hdg, use_bank=False)
                continue 

            # ② STD 이후: 정상 업데이트
            if has_lla:
                bridge.update_actor(acid, {"lon":pkt["lon"],"lat":pkt["lat"],"alt_m":alt_agl}, t_now, hdg, use_bank=False)
            elif all(k in pkt for k in ("x","y","z")):
                zned = -float(pkt["z"])
                bridge.update_actor(acid, {"x":pkt["x"],"y":pkt["y"],"z":zned}, t_now, hdg, use_bank=False)
            else:
                continue

        # 4) 종료된 액터 제거
        bridge.prune_actors(alive_ids)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] user interrupt")
