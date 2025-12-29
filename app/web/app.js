(() => {
  const body = document.body;
  const demConfig = {
    tileUrl: "/dem/{z}/{x}/{y}.png",
    tileSize: 256,
    maxZoom: 12,
    encoding: "terrarium",
    exaggeration: 1.0,
    pitchThreshold: 2,
  };
  const viewConfig = {
    maxZoomBuffer: 2,
    maxPitch: 85,
  };
  const dataConfig = {
    vertiportCsv: body.dataset.vertiportCsv || "/data/vertiport_default.csv",
    waypointCsv: body.dataset.waypointCsv || "/data/waypoint_default.csv",
  };
  const config = {
    tileUrl: body.dataset.tileUrl,
    minZoom: Number(body.dataset.minZoom),
    maxZoom: Number(body.dataset.maxZoom),
    center: [Number(body.dataset.centerLon), Number(body.dataset.centerLat)],
    startZoom: Number(body.dataset.startZoom),
    bounds: body.dataset.bounds ? JSON.parse(body.dataset.bounds) : null,
    dem: demConfig,
    view: viewConfig,
    data: dataConfig,
  };

  const parseCsvRows = (text) => {
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;

    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      if (inQuotes) {
        if (char === "\"") {
          if (text[i + 1] === "\"") {
            field += "\"";
            i += 1;
          } else {
            inQuotes = false;
          }
        } else {
          field += char;
        }
        continue;
      }

      if (char === "\"") {
        inQuotes = true;
      } else if (char === ",") {
        row.push(field);
        field = "";
      } else if (char === "\n") {
        row.push(field);
        rows.push(row);
        row = [];
        field = "";
      } else if (char !== "\r") {
        field += char;
      }
    }

    if (field.length > 0 || row.length > 0) {
      row.push(field);
      rows.push(row);
    }
    return rows;
  };

  const normalizeRows = (rows) =>
    rows.filter((row) => row.some((cell) => cell.trim().length > 0));

  const readCell = (row, index) => (row[index] ? row[index].trim() : "");
  const BASE_MAP_PALETTES = {
    light: {
      background: "#eef2f3",
      landcover: "#dfe8d8",
      landuse: "#e8e6d8",
      park: "#cfe8c5",
      water: "#a8c8e6",
      waterway: "#90b7dd",
      boundary: "#9a9a9a",
      transportation: "#c2b59b",
      building: "#d0c7c2",
    },
    dark: {
      background: "#0f1820",
      landcover: "#182522",
      landuse: "#1b2320",
      park: "#1d3024",
      water: "#142a3e",
      waterway: "#1f425e",
      boundary: "#5e6872",
      transportation: "#4a453a",
      building: "#2f2c2a",
    },
  };
  const FT_TO_M = 0.3048;
  const parseAltitudeMeters = (value) => {
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed)) {
      return null;
    }
    return parsed * FT_TO_M;
  };
  const formatAltitudeMeters = (value) => {
    const meters = parseAltitudeMeters(value);
    if (meters === null) {
      return "";
    }
    return meters.toFixed(1);
  };
  const hexToRgba = (hex, alpha = 1) => {
    const cleaned = hex.replace("#", "");
    const full =
      cleaned.length === 3
        ? cleaned
            .split("")
            .map((char) => char + char)
            .join("")
        : cleaned;
    const value = Number.parseInt(full, 16);
    if (!Number.isFinite(value)) {
      return new Float32Array([1, 1, 1, alpha]);
    }
    const r = ((value >> 16) & 255) / 255;
    const g = ((value >> 8) & 255) / 255;
    const b = (value & 255) / 255;
    return new Float32Array([r, g, b, alpha]);
  };

  class CenterControl {
    constructor(onClick) {
      this._onClick = onClick;
      this._container = null;
      this._button = null;
      this._handleClick = () => {
        if (this._onClick) {
          this._onClick();
        }
      };
    }

    onAdd(map) {
      this._map = map;
      const container = document.createElement("div");
      container.className = "maplibregl-ctrl maplibregl-ctrl-group center-control";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "center-control-btn";
      button.title = "Reset view";
      const dot = document.createElement("span");
      dot.className = "center-control-dot";
      button.appendChild(dot);
      button.addEventListener("click", this._handleClick);
      container.appendChild(button);
      this._container = container;
      this._button = button;
      return container;
    }

    onRemove() {
      if (this._button) {
        this._button.removeEventListener("click", this._handleClick);
      }
      if (this._container && this._container.parentNode) {
        this._container.parentNode.removeChild(this._container);
      }
      this._map = undefined;
    }
  }

  class MapApp {
    constructor(config) {
      this.config = config;
      this.map = null;
      this.mapContainer = document.getElementById("map");
      this.themeButtons = Array.from(document.querySelectorAll(".theme-card"));
      this.actionButtons = Array.from(document.querySelectorAll(".ui-btn"));
      this.closeButtons = Array.from(document.querySelectorAll(".panel-close"));
      this.panels = {
        vertiport: document.getElementById("vertiport-panel"),
        corridor: document.getElementById("corridor-panel"),
        plan: document.getElementById("plan-panel"),
        settings: document.getElementById("settings-panel"),
      };
      this.tableBodies = {
        vertiport: document.getElementById("vertiport-table-body"),
        corridor: document.getElementById("corridor-table-body"),
      };
      this.currentTheme = "light";
      this.terrainEnabled = false;
      this.homeView = null;
    }

    init() {
      this.map = this.createMap();
      this.bindThemeButtons();
      this.bindActionButtons();
      this.bindCloseButtons();
      this.applyTheme(this.currentTheme);
      this.loadDefaultTables();
      this.setupScaleObserver();
      this.setupTerrainToggle();
    }

    createMap() {
      const style = this.buildStyle();
      const map = new maplibregl.Map({
        container: "map",
        style: style,
        center: this.config.center,
        zoom: this.config.startZoom,
        minZoom: this.config.minZoom,
        maxZoom: this.config.maxZoom + this.config.view.maxZoomBuffer,
        maxPitch: this.config.view.maxPitch,
        attributionControl: false,
      });

      map.addControl(new CenterControl(() => this.resetView()), "bottom-right");
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

      map.once("load", () => {
        this.updateMapTheme(this.currentTheme);
        this.loadCorridorOverlay();
        if (this.config.bounds) {
          map.fitBounds(this.config.bounds, { padding: 20, duration: 0 });
          map.once("idle", () => this.captureHomeView());
        } else {
          this.captureHomeView();
        }
      });
      return map;
    }

    buildStyle() {
      return {
        version: 8,
        sources: {
          mbtiles: {
            type: "vector",
            tiles: [this.config.tileUrl],
            minzoom: this.config.minZoom,
            maxzoom: this.config.maxZoom,
          },
          dem: {
            type: "raster-dem",
            tiles: [this.config.dem.tileUrl],
            tileSize: this.config.dem.tileSize,
            maxzoom: this.config.dem.maxZoom,
            encoding: this.config.dem.encoding,
          },
        },
        layers: [
          {
            id: "background",
            type: "background",
            paint: { "background-color": "#eef2f3" },
          },
          {
            id: "landcover",
            type: "fill",
            source: "mbtiles",
            "source-layer": "landcover",
            paint: { "fill-color": "#dfe8d8", "fill-opacity": 0.7 },
          },
          {
            id: "landuse",
            type: "fill",
            source: "mbtiles",
            "source-layer": "landuse",
            paint: { "fill-color": "#e8e6d8", "fill-opacity": 0.7 },
          },
          {
            id: "park",
            type: "fill",
            source: "mbtiles",
            "source-layer": "park",
            paint: { "fill-color": "#cfe8c5", "fill-opacity": 0.85 },
          },
          {
            id: "water",
            type: "fill",
            source: "mbtiles",
            "source-layer": "water",
            paint: { "fill-color": "#a8c8e6" },
          },
          {
            id: "waterway",
            type: "line",
            source: "mbtiles",
            "source-layer": "waterway",
            paint: { "line-color": "#90b7dd", "line-width": 1 },
          },
          {
            id: "boundary",
            type: "line",
            source: "mbtiles",
            "source-layer": "boundary",
            paint: { "line-color": "#9a9a9a", "line-width": 1, "line-dasharray": [2, 2] },
          },
          {
            id: "transportation",
            type: "line",
            source: "mbtiles",
            "source-layer": "transportation",
            paint: { "line-color": "#c2b59b", "line-width": 1 },
          },
          {
            id: "building",
            type: "fill",
            source: "mbtiles",
            "source-layer": "building",
            minzoom: 13,
            paint: { "fill-color": "#d0c7c2", "fill-opacity": 0.6 },
          },
        ],
      };
    }

    bindThemeButtons() {
      this.themeButtons.forEach((button) => {
        button.addEventListener("click", () => {
          const theme = this.getThemeFromButton(button);
          this.setTheme(theme);
        });
      });
    }

    getThemeFromButton(button) {
      const label = button.querySelector(".label");
      if (label) {
        const text = label.textContent.trim().toLowerCase();
        if (text === "dark") {
          return "dark";
        }
        if (text === "white" || text === "light") {
          return "light";
        }
      }
      return button.dataset.theme || "light";
    }

    bindActionButtons() {
      this.actionButtons.forEach((button) => {
        const action = button.dataset.action;
        if (!action) {
          return;
        }
        button.addEventListener("click", () => {
          this.handleAction(action);
        });
      });
    }

    bindCloseButtons() {
      this.closeButtons.forEach((button) => {
        const action = button.dataset.action;
        if (!action || !action.startsWith("close-")) {
          return;
        }
        const panelName = action.replace("close-", "");
        button.addEventListener("click", () => {
          this.hidePanel(panelName);
        });
      });
    }

    async loadDefaultTables() {
      await Promise.all([
        this.loadVertiportTable(this.config.data.vertiportCsv),
        this.loadCorridorTable(this.config.data.waypointCsv),
      ]);
    }

    async loadVertiportTable(url) {
      const body = this.tableBodies.vertiport;
      if (!body) {
        return;
      }
      try {
        const rows = await this.fetchCsvRows(url);
        this.populateVertiportTable(body, rows);
      } catch (error) {
        console.warn("Failed to load vertiport data.", error);
      }
    }

    async loadCorridorTable(url) {
      const body = this.tableBodies.corridor;
      if (!body) {
        return;
      }
      try {
        const rows = await this.fetchCsvRows(url);
        this.populateCorridorTable(body, rows);
      } catch (error) {
        console.warn("Failed to load corridor data.", error);
      }
    }

    async fetchCsvRows(url) {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }
      const text = await response.text();
      const rows = normalizeRows(parseCsvRows(text));
      if (rows.length <= 1) {
        return [];
      }
      return rows.slice(1);
    }

    populateVertiportTable(body, rows) {
      body.innerHTML = "";
      rows.forEach((row, index) => {
        const cells = [
          String(index + 1),
          readCell(row, 0),
          readCell(row, 3),
          readCell(row, 2),
          readCell(row, 1),
        ];
        body.appendChild(this.createTableRow(cells));
      });
    }

    populateCorridorTable(body, rows) {
      body.innerHTML = "";
      rows.forEach((row, index) => {
        const cells = [
          String(index + 1),
          readCell(row, 0),
          readCell(row, 2),
          readCell(row, 1),
          formatAltitudeMeters(readCell(row, 3)),
          readCell(row, 4),
        ];
        body.appendChild(this.createTableRow(cells));
      });
    }

    createTableRow(values) {
      const row = document.createElement("tr");
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      });
      return row;
    }

    handleAction(action) {
      switch (action) {
        case "vertiport":
          this.handleVertiportAction();
          break;
        case "corridor":
          this.handleCorridorAction();
          break;
        case "plan":
          this.handlePlanAction();
          break;
        case "settings":
          this.handleSettingsAction();
          break;
        default:
          break;
      }
    }

    handleVertiportAction() {
      this.togglePanel("vertiport");
    }

    handleCorridorAction() {
      this.togglePanel("corridor");
    }

    handlePlanAction() {
      this.togglePanel("plan");
    }

    handleSettingsAction() {
      this.togglePanel("settings");
    }

    captureHomeView() {
      if (!this.map) {
        return;
      }
      const center = this.map.getCenter();
      this.homeView = {
        center: [center.lng, center.lat],
        zoom: this.map.getZoom(),
        bearing: this.map.getBearing(),
        pitch: this.map.getPitch(),
      };
    }

    resetView() {
      if (!this.map || !this.homeView) {
        return;
      }
      this.map.easeTo({
        center: this.homeView.center,
        zoom: this.homeView.zoom,
        bearing: this.homeView.bearing,
        pitch: this.homeView.pitch,
        duration: 400,
      });
    }

    togglePanel(name) {
      const panel = this.panels[name];
      if (!panel) {
        return;
      }
      const isVisible = panel.classList.contains("is-visible");
      if (isVisible) {
        this.hidePanel(name);
      } else {
        this.showPanel(name);
      }
    }

    showPanel(name) {
      Object.entries(this.panels).forEach(([key, panel]) => {
        const isTarget = key === name;
        panel.classList.toggle("is-visible", isTarget);
        panel.setAttribute("aria-hidden", isTarget ? "false" : "true");
      });
    }

    hidePanel(name) {
      const panel = this.panels[name];
      if (!panel) {
        return;
      }
      panel.classList.remove("is-visible");
      panel.setAttribute("aria-hidden", "true");
    }

    setTheme(theme) {
      if (theme === this.currentTheme) {
        return;
      }
      this.currentTheme = theme;
      this.applyTheme(theme);
    }

    applyTheme(theme) {
      const isDark = theme === "dark";
      this.mapContainer.classList.toggle("theme-dark", isDark);
      this.themeButtons.forEach((button) => {
        button.classList.toggle("is-active", button.dataset.theme === theme);
      });
      this.updateMapTheme(theme);
      this.updateCorridorTheme(theme);
    }

    getCorridorColor(theme) {
      return theme === "dark" ? "#0efcff" : "#2c6dff";
    }

    updateMapTheme(theme) {
      if (!this.map || !this.map.isStyleLoaded()) {
        return;
      }
      const palette = BASE_MAP_PALETTES[theme] || BASE_MAP_PALETTES.light;
      this.map.setPaintProperty("background", "background-color", palette.background);
      this.map.setPaintProperty("landcover", "fill-color", palette.landcover);
      this.map.setPaintProperty("landuse", "fill-color", palette.landuse);
      this.map.setPaintProperty("park", "fill-color", palette.park);
      this.map.setPaintProperty("water", "fill-color", palette.water);
      this.map.setPaintProperty("waterway", "line-color", palette.waterway);
      this.map.setPaintProperty("boundary", "line-color", palette.boundary);
      this.map.setPaintProperty("transportation", "line-color", palette.transportation);
      this.map.setPaintProperty("building", "fill-color", palette.building);
    }

    updateCorridorTheme(theme) {
      if (!this.map) {
        return;
      }
      const color = this.getCorridorColor(theme);
      if (this.corridorLayer && this.corridorLayer.setColor) {
        this.corridorLayer.setColor(color);
        this.map.triggerRepaint();
      }
    }

    async loadCorridorOverlay() {
      if (!this.map) {
        return;
      }
      try {
        const rows = await this.fetchCsvRows(this.config.data.waypointCsv);
        const data = this.buildCorridorFeatures(rows);
        this.setCorridorLayer(data);
        this.updateCorridorTheme(this.currentTheme);
      } catch (error) {
        console.warn("Failed to load corridor overlay.", error);
      }
    }

    buildCorridorFeatures(rows) {
      const points = [];
      const lines = [];
      const lookup = new Map();
      rows.forEach((row) => {
        const name = readCell(row, 0);
        const lat = Number.parseFloat(readCell(row, 1));
        const lon = Number.parseFloat(readCell(row, 2));
        if (!name || !Number.isFinite(lat) || !Number.isFinite(lon)) {
          return;
        }
        const altitudeMeters = parseAltitudeMeters(readCell(row, 3));
        const coord = [lon, lat];
        const entry = { name, coord, altitude_m: altitudeMeters ?? 0 };
        lookup.set(name, entry);
        points.push(entry);
      });

      const seen = new Set();
      rows.forEach((row) => {
        const name = readCell(row, 0);
        const startEntry = lookup.get(name);
        if (!startEntry) {
          return;
        }
        const linkCell = readCell(row, 4);
        if (!linkCell) {
          return;
        }
        linkCell.split(",").forEach((raw) => {
          const target = raw.trim();
          if (!target) {
            return;
          }
          const endEntry = lookup.get(target);
          if (!endEntry) {
            return;
          }
          const key = [name, target].sort().join("|");
          if (seen.has(key)) {
            return;
          }
          seen.add(key);
          lines.push({
            from: name,
            to: target,
            start: startEntry,
            end: endEntry,
          });
        });
      });

      return { points, lines };
    }

    setCorridorLayer(data) {
      const map = this.map;
      if (!map) {
        return;
      }
      if (this.corridorLayer && map.getLayer(this.corridorLayer.id)) {
        return;
      }
      const layer = this.createCorridorLayer(data);
      this.corridorLayer = layer;
      map.addLayer(layer);
    }

    buildCorridorBuffers(data) {
      const pointPositions = [];
      const linePositions = [];
      data.points.forEach((entry) => {
        const merc = maplibregl.MercatorCoordinate.fromLngLat(
          entry.coord,
          entry.altitude_m ?? 0,
        );
        pointPositions.push(merc.x, merc.y, merc.z);
      });
      data.lines.forEach((line) => {
        const startAlt = line.start.altitude_m ?? 0;
        const endAlt = line.end.altitude_m ?? 0;
        const start = maplibregl.MercatorCoordinate.fromLngLat(line.start.coord, startAlt);
        const end = maplibregl.MercatorCoordinate.fromLngLat(line.end.coord, endAlt);
        linePositions.push(start.x, start.y, start.z, end.x, end.y, end.z);
      });
      return {
        pointPositions,
        linePositions,
      };
    }

    createCorridorLayer(data) {
      const buffers = this.buildCorridorBuffers(data);
      const color = this.getCorridorColor(this.currentTheme);
      const pointSize = 6 * (window.devicePixelRatio || 1);
      const layer = {
        id: "corridor-3d",
        type: "custom",
        renderingMode: "3d",
        _color: color,
        _pointSize: pointSize,
        setColor(nextColor) {
          this._color = nextColor;
        },
        onAdd(_map, gl) {
          this._gl = gl;
          const vertexSource = `
            attribute vec3 a_pos;
            uniform mat4 u_matrix;
            uniform float u_pointSize;
            void main() {
              gl_Position = u_matrix * vec4(a_pos, 1.0);
              gl_PointSize = u_pointSize;
            }
          `;
          const fragmentSource = `
            precision mediump float;
            uniform vec4 u_color;
            uniform float u_isPoint;
            void main() {
              if (u_isPoint > 0.5) {
                float dist = length(gl_PointCoord - vec2(0.5));
                if (dist > 0.5) {
                  discard;
                }
              }
              gl_FragColor = u_color;
            }
          `;
          const compile = (type, source) => {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            return shader;
          };
          const vertexShader = compile(gl.VERTEX_SHADER, vertexSource);
          const fragmentShader = compile(gl.FRAGMENT_SHADER, fragmentSource);
          const program = gl.createProgram();
          gl.attachShader(program, vertexShader);
          gl.attachShader(program, fragmentShader);
          gl.linkProgram(program);
          this._program = program;
          this._aPos = gl.getAttribLocation(program, "a_pos");
          this._uMatrix = gl.getUniformLocation(program, "u_matrix");
          this._uColor = gl.getUniformLocation(program, "u_color");
          this._uPointSize = gl.getUniformLocation(program, "u_pointSize");
          this._uIsPoint = gl.getUniformLocation(program, "u_isPoint");

          this._pointBuffer = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, this._pointBuffer);
          gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array(buffers.pointPositions),
            gl.STATIC_DRAW,
          );
          this._pointCount = buffers.pointPositions.length / 3;

          this._lineBuffer = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, this._lineBuffer);
          gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array(buffers.linePositions),
            gl.STATIC_DRAW,
          );
          this._lineCount = buffers.linePositions.length / 3;
        },
        render(gl, matrix) {
          if (!this._program) {
            return;
          }
          gl.useProgram(this._program);
          gl.uniformMatrix4fv(this._uMatrix, false, matrix);
          gl.uniform1f(this._uPointSize, this._pointSize);
          gl.enableVertexAttribArray(this._aPos);
          gl.enable(gl.BLEND);
          gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

          const color = hexToRgba(this._color, 0.95);
          gl.uniform4fv(this._uColor, color);

          if (this._lineCount > 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this._lineBuffer);
            gl.vertexAttribPointer(this._aPos, 3, gl.FLOAT, false, 0, 0);
            gl.uniform1f(this._uIsPoint, 0);
            gl.lineWidth(1);
            gl.drawArrays(gl.LINES, 0, this._lineCount);
          }

          if (this._pointCount > 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this._pointBuffer);
            gl.vertexAttribPointer(this._aPos, 3, gl.FLOAT, false, 0, 0);
            gl.uniform1f(this._uIsPoint, 1);
            gl.drawArrays(gl.POINTS, 0, this._pointCount);
          }
        },
      };
      return layer;
    }
    

    setupTerrainToggle() {
      const updateTerrain = () => {
        const shouldEnable = this.map.getPitch() >= this.config.dem.pitchThreshold;
        if (shouldEnable === this.terrainEnabled) {
          return;
        }
        if (shouldEnable) {
          if (this.map.getSource("dem")) {
            this.map.setTerrain({
              source: "dem",
              exaggeration: this.config.dem.exaggeration,
            });
          }
        } else {
          this.map.setTerrain(null);
        }
        this.terrainEnabled = shouldEnable;
      };

      this.map.on("load", () => {
        updateTerrain();
        this.map.on("pitch", updateTerrain);
      });
    }

    setupScaleObserver() {
      const root = document.documentElement;
      const baseWidth = 1200;
      const baseHeight = 900;

      const getViewportSize = () => {
        if (window.visualViewport) {
          return { width: window.visualViewport.width, height: window.visualViewport.height };
        }
        return { width: window.innerWidth, height: window.innerHeight };
      };

      const updateScale = () => {
        const { width, height } = getViewportSize();
        if (!width || !height) {
          return;
        }
        const scale = Math.min(width / baseWidth, height / baseHeight);
        const clamped = Math.max(0.75, Math.min(scale, 1.25));
        root.style.setProperty("--ui-scale", clamped.toFixed(3));
        if (this.map) {
          requestAnimationFrame(() => this.map.resize());
        }
      };

      updateScale();
      window.addEventListener("resize", updateScale);
      if (window.visualViewport) {
        window.visualViewport.addEventListener("resize", updateScale);
      }

      if (window.ResizeObserver) {
        const observer = new ResizeObserver(updateScale);
        observer.observe(document.documentElement);
      }
    }
  }

  const app = new MapApp(config);
  app.init();
} )();
