# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import pandas as pd

def run_js(webview, code: str):
    if webview is None: return
    try: webview.page().runJavaScript(code)
    except Exception: pass

def update_noise(webview, val_map: dict, color_map: dict):
    vm = json.dumps(val_map, ensure_ascii=False)
    cm = json.dumps(color_map, ensure_ascii=False)
    run_js(webview, f"try {{ window.JY.updateNoise({vm}, {cm}); }} catch(e) {{ console.error(e); }}")

def set_ranks(webview, rank_map: dict):
    rm = json.dumps({str(int(k)): int(v) for k, v in rank_map.items()}, ensure_ascii=False)
    run_js(webview, f"try {{ window.JY.setRanks({rm}); }} catch(e) {{ console.error(e); }}")

def clear_top(webview):
    run_js(webview, "try { window.JY.clearRanks(); } catch(e) {}")

def highlight_gids(webview, gids, pan: bool=True):
    arr = [str(int(g)) for g in gids if pd.notna(g)]
    arr_json = json.dumps(arr, ensure_ascii=False)
    run_js(webview, f"try {{ window.JY.setHighlight({arr_json}, {str(bool(pan)).lower()}); }} catch(e) {{ console.error(e); }}")

def clear_highlight(webview):
    run_js(webview, "try { window.JY.clearHighlight(); } catch(e) {}")

def build_js_api(map_name, grid_name, noise_name):
    return f'''
        (function(){{
          var m = {map_name};
          var grid = {grid_name};
          var noise = {noise_name};
          if (!window.JY) window.JY = {{ }};
          window.JY.grid = grid;
          window.JY.noise = noise;
          window.JY.mode = 'pop';
          window.JY.noiseVal = {{}};
          window.JY.noiseColor = {{}};
          window.JY.edgeOn = false;
          var css = ".jy-rank-badge{{display:inline-block;min-width:24px;height:24px;line-height:24px;padding:0 6px;border-radius:12px;background:#ff6d00;color:#ffffff;font-weight:700;font-size:12px;text-align:center;box-shadow:0 0 0 2px rgba(0,0,0,.25);}}";
          if (!document.getElementById('jy-style')) {{
            var st = document.createElement('style'); st.id='jy-style'; st.innerHTML = css; document.head.appendChild(st);
          }}
          window.JY.rankGroup = L.layerGroup().addTo(m);
          function apply(){{
            var showPop   = (window.JY.mode !== 'noise');
            var showNoise = (window.JY.mode !== 'pop');
            grid.eachLayer(function(ly){{
              var col = (ly.feature && ly.feature.properties && ly.feature.properties.__pop_color__) || '#ECEFF1';
              ly.setStyle({{ fillColor: showPop ? col : '#ECEFF1', color:'#999', weight: showPop ? 0.2 : 0, fillOpacity: showPop ? 0.55 : 0.0 }});
            }});
            noise.eachLayer(function(ly){{
              var gid = ly.feature && ly.feature.properties && ly.feature.properties.gid;
              var col = (gid != null && window.JY.noiseColor[String(gid)]) || '#FF5722';
              var op  = (gid != null && window.JY.noiseVal[String(gid)] >= 50) ? 0.55 : 0.0;
              if (!showNoise) op = 0.0;
              ly.setStyle({{ fillColor: col, color: col, weight: 0, fillOpacity: op }});
            }});
          }}
          window.JY.setMode = function(mode){{ window.JY.mode = mode || 'pop'; apply(); }};
          window.JY.setEdge = function(on){{
            window.JY.edgeOn = !!on;
            grid.eachLayer(function(ly){{ ly.setStyle({{ weight: on ? 0.6 : 0.2, color: on ? '#666' : '#999' }}); }});
          }};
          window.JY.updateNoise = function(valMap, colorMap){{ window.JY.noiseVal = valMap || {{}}; window.JY.noiseColor = colorMap || {{}}; apply(); }};
          function clearRanks(){{ window.JY.rankGroup.clearLayers(); noise.eachLayer(function(ly){{ ly.setStyle({{ weight:0 }}); }}); }}
          function setRanks(rankMap){{
            clearRanks(); if (!rankMap) return;
            var latlng = [];
            noise.eachLayer(function(ly){{
              var gid = ly.feature && ly.feature.properties && ly.feature.properties.gid;
              if (!gid) return;
              var r = rankMap[String(gid)]; if (!r) return;
              var c = ly.getBounds().getCenter(); latlng.push(c);
              var html = "<div class=\\"jy-rank-badge\\">"+String(r)+"</div>";
              L.marker(c, {{ icon: L.divIcon({{className:'', html: html}}) }}).addTo(window.JY.rankGroup);
              ly.setStyle({{ color:'#ff6d00', weight:1.2 }});
            }});
            if (latlng.length>0){{ try {{ var b = L.latLngBounds(latlng); setTimeout(function(){{ m.fitBounds(b.pad(0.20)); }}, 50); }} catch(e){{}} }}
          }}
          window.JY.setRanks = setRanks; window.JY.clearRanks = clearRanks;
          var hlGroup = L.layerGroup().addTo(m);
          function clearHighlight(){{ hlGroup.clearLayers(); }}
          function setHighlight(gids, pan){{
            clearHighlight(); if (!gids || gids.length===0) return;
            var latlng = [];
            noise.eachLayer(function(ly){{
              var gid = ly.feature && ly.feature.properties && ly.feature.properties.gid;
              if (!gid) return;
              if (gids.indexOf(String(gid))>=0){{
                var b = ly.getBounds(); latlng.push(b.getCenter());
                L.rectangle(b, {{color:'#ff9800', weight:2, fillOpacity:0}}).addTo(hlGroup);
              }}
            }});
            if (pan && latlng.length>0){{ try {{ var bb = L.latLngBounds(latlng); setTimeout(function(){{m.fitBounds(bb.pad(0.30), {{ maxZoom: m.getZoom() }}); }}, 50); }} catch(e){{}} }}
          }}
          window.JY.setHighlight = setHighlight; window.JY.clearHighlight = clearHighlight;
        }})();
    '''

def update_traffic(webview, val_map: dict, color_map: dict):
    vm = json.dumps(val_map, ensure_ascii=False)
    cm = json.dumps(color_map, ensure_ascii=False)
    run_js(webview, f"try {{ if(window.JY) window.JY.updateTraffic({vm}, {cm}); }} catch(e) {{ console.error(e); }}")

def build_js_api_traffic(map_name: str, lanes_name: str) -> str:
    """lane_id별 스타일 갱신 (Noise의 window.JY 패턴 유지)"""
    return f"""
    (function(){{
      var m = {map_name};
      var lanes = {lanes_name};
      if (!window.JY) window.JY = {{}};
      window.JY.lanes = lanes;
      window.JY.laneVal = {{}};
      window.JY.laneColor = {{}};

      function apply(){{
        if (!window.JY || !window.JY.lanes) return;
        lanes.eachLayer(function(ly){{
          var id = ly.feature && ly.feature.properties && ly.feature.properties.lane_id;
          if (!id) return;
          var col = window.JY.laneColor[String(id)] || '#1E88E5';
          var v   = window.JY.laneVal[String(id)] || 0;
          var op  = v > 0 ? 0.80 : 0.15;
          ly.setStyle({{ color: col, weight: 3.0, opacity: op }});
        }});
      }}
      window.JY.updateTraffic = function(valMap, colorMap){{
        window.JY.laneVal = valMap || {{}};
        window.JY.laneColor = colorMap || {{}};
        apply();
      }};
    }})();
    """