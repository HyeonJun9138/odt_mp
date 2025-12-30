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
  const getDataRows = (text) => {
    const rows = normalizeRows(parseCsvRows(text));
    return rows.length > 1 ? rows.slice(1) : [];
  };
  const getFileName = (value) => {
    if (!value) {
      return "";
    }
    const parts = value.split(/[\\/]/);
    return parts[parts.length - 1];
  };

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
  const HOVER_OUTLINE_COLOR = "#ffe600";
  const MAP_SIZE_SCALE = 1.2;
  const VERTIPORT_ICON_ID = "vertiport-icon";
  const VERTIPORT_LINE_COLOR = "#2ecc71";

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
      this.playbackPanel = document.getElementById("playback-panel");
      this.playbackSpeedButton = null;
      this.playbackSpeedLabel = null;
      this.playbackSpeeds = [1, 2, 4, 8, 16];
      this.playbackSpeedIndex = 0;
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
      this.fileControls = {
        vertiport: {
          nameInput: document.getElementById("vertiport-file-name"),
          openButton: document.querySelector('[data-action="open-vertiport"]'),
          resetButton: document.querySelector('[data-action="reset-vertiport"]'),
        },
        corridor: {
          nameInput: document.getElementById("corridor-file-name"),
          openButton: document.querySelector('[data-action="open-corridor"]'),
          resetButton: document.querySelector('[data-action="reset-corridor"]'),
        },
      };
      this.filePickers = {};
      this.defaultFiles = {
        vertiport: {
          url: this.config.data.vertiportCsv,
          name: getFileName(this.config.data.vertiportCsv),
        },
        corridor: {
          url: this.config.data.waypointCsv,
          name: getFileName(this.config.data.waypointCsv),
        },
      };
      this.currentTheme = "light";
      this.terrainEnabled = false;
      this.homeView = null;
      this.corridorLayer = null;
      this.corridorHitData = null;
      this.corridorHover = null;
      this.corridorHoverLabel = null;
      this.pendingCorridorRows = null;
      this.lastCorridorRows = null;
      this.corridorEnsured = false;
      this.corridorPointLookup = new Map();
      this.vertiportHover = null;
      this.vertiportHoverId = null;
      this.vertiportLinkHover = null;
      this.vertiportLinkHoverId = null;
      this.vertiportHoverLabel = null;
      this.vertiportHoverName = null;
      this.vertiportLabels = [];
      this.lastVertiportRows = null;
      this.pendingVertiportRows = null;
      this.vertiportIconPromise = null;
    }

    init() {
      this.map = this.createMap();
      this.normalizeThemeButtons();
      this.bindThemeButtons();
      this.bindActionButtons();
      this.bindCloseButtons();
      this.bindFileControls();
      this.bindPlaybackControls();
      this.syncFileNames();
      this.applyTheme(this.currentTheme);
      this.loadDefaultTables();
      this.setupScaleObserver();
      this.setupTerrainToggle();
      this.setupCorridorHover();
      this.setupVertiportHover();
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

      map.once("load", async () => {
        this.updateMapTheme(this.currentTheme);
        await this.ensureVertiportIcon();
        if (this.pendingVertiportRows) {
          const rows = this.pendingVertiportRows;
          this.pendingVertiportRows = null;
          this.updateVertiportOverlayFromRows(rows);
        } else {
          this.loadVertiportOverlay();
        }
        if (this.pendingCorridorRows) {
          const rows = this.pendingCorridorRows;
          this.pendingCorridorRows = null;
          this.updateCorridorOverlayFromRows(rows);
        } else {
          this.loadCorridorOverlay();
        }
        if (this.config.bounds) {
          map.fitBounds(this.config.bounds, { padding: 20, duration: 0 });
          map.once("idle", () => {
            this.captureHomeView();
            this.ensureCorridorReady();
          });
        } else {
          this.captureHomeView();
          map.once("idle", () => this.ensureCorridorReady());
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
          const theme = button.dataset.theme || "light";
          this.setTheme(theme);
        });
      });
    }

    normalizeThemeButtons() {
      this.themeButtons.forEach((button) => {
        const label = button.querySelector(".label");
        if (!label) {
          return;
        }
        const text = label.textContent.trim().toLowerCase();
        if (text === "dark") {
          button.dataset.theme = "dark";
        } else if (text === "white" || text === "light") {
          button.dataset.theme = "light";
        }
      });
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

    bindPlaybackControls() {
      if (!this.playbackPanel) {
        return;
      }
      this.playbackSpeedButton = this.playbackPanel.querySelector(".playback-btn-speed");
      this.playbackSpeedLabel = this.playbackPanel.querySelector(".playback-speed-text");
      this.updatePlaybackSpeedLabel();
      if (this.playbackSpeedButton) {
        this.playbackSpeedButton.addEventListener("click", () => {
          this.playbackSpeedIndex =
            (this.playbackSpeedIndex + 1) % this.playbackSpeeds.length;
          this.updatePlaybackSpeedLabel();
        });
      }
    }

    updatePlaybackSpeedLabel() {
      if (!this.playbackSpeedLabel) {
        return;
      }
      const speed = this.playbackSpeeds[this.playbackSpeedIndex] || this.playbackSpeeds[0];
      this.playbackSpeedLabel.textContent = `${speed}x`;
      if (this.playbackSpeedButton) {
        this.playbackSpeedButton.title = `Speed ${speed}x`;
      }
    }

    bindFileControls() {
      this.filePickers.vertiport = this.createFilePicker((text, fileName) => {
        const rows = getDataRows(text);
        if (this.tableBodies.vertiport) {
          this.populateVertiportTable(this.tableBodies.vertiport, rows);
        }
        this.updateVertiportOverlayFromRows(rows);
        this.updatePanelFileName("vertiport", fileName);
      });
      this.filePickers.corridor = this.createFilePicker((text, fileName) => {
        const rows = getDataRows(text);
        if (this.tableBodies.corridor) {
          this.populateCorridorTable(this.tableBodies.corridor, rows);
        }
        this.updateCorridorOverlayFromRows(rows);
        this.updatePanelFileName("corridor", fileName);
      });

      const vertiport = this.fileControls.vertiport;
      if (vertiport && vertiport.openButton) {
        vertiport.openButton.addEventListener("click", () => {
          this.filePickers.vertiport.click();
        });
      }
      if (vertiport && vertiport.resetButton) {
        vertiport.resetButton.addEventListener("click", () => {
          this.resetVertiport();
        });
      }

      const corridor = this.fileControls.corridor;
      if (corridor && corridor.openButton) {
        corridor.openButton.addEventListener("click", () => {
          this.filePickers.corridor.click();
        });
      }
      if (corridor && corridor.resetButton) {
        corridor.resetButton.addEventListener("click", () => {
          this.resetCorridor();
        });
      }
    }

    syncFileNames() {
      Object.entries(this.defaultFiles).forEach(([key, info]) => {
        this.updatePanelFileName(key, info.name);
      });
    }

    updatePanelFileName(key, name) {
      const control = this.fileControls[key];
      if (!control || !control.nameInput) {
        return;
      }
      control.nameInput.value = name || "";
    }

    createFilePicker(onSelect) {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".csv,text/csv";
      input.style.display = "none";
      input.addEventListener("change", () => {
        const file = input.files && input.files[0];
        if (!file) {
          return;
        }
        const reader = new FileReader();
        reader.onload = () => {
          const text = typeof reader.result === "string" ? reader.result : "";
          onSelect(text, file.name);
        };
        reader.onerror = () => {
          console.warn("Failed to read csv file.");
        };
        reader.readAsText(file);
        input.value = "";
      });
      document.body.appendChild(input);
      return input;
    }

    resetVertiport() {
      const info = this.defaultFiles.vertiport;
      if (info) {
        this.updatePanelFileName("vertiport", info.name);
        this.loadVertiportTable(info.url);
      }
    }

    resetCorridor() {
      const info = this.defaultFiles.corridor;
      if (info) {
        this.updatePanelFileName("corridor", info.name);
        this.loadCorridorData(info.url);
      }
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
        this.updateVertiportOverlayFromRows(rows);
      } catch (error) {
        console.warn("Failed to load vertiport data.", error);
      }
    }

    async loadCorridorTable(url) {
      await this.loadCorridorData(url);
    }

    async loadCorridorData(url) {
      const body = this.tableBodies.corridor;
      if (!body) {
        return;
      }
      try {
        const rows = await this.fetchCsvRows(url);
        this.populateCorridorTable(body, rows);
        this.updateCorridorOverlayFromRows(rows);
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
      return getDataRows(text);
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
        case "playback":
          this.togglePlaybackPanel();
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

    togglePlaybackPanel() {
      if (!this.playbackPanel) {
        return;
      }
      const isOpen = this.playbackPanel.classList.contains("is-open");
      this.playbackPanel.classList.toggle("is-open", !isOpen);
      this.playbackPanel.setAttribute("aria-hidden", isOpen ? "true" : "false");
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

    async loadVertiportOverlay() {
      if (!this.map) {
        return;
      }
      try {
        const rows = await this.fetchCsvRows(this.config.data.vertiportCsv);
        this.updateVertiportOverlayFromRows(rows);
      } catch (error) {
        console.warn("Failed to load vertiport overlay.", error);
      }
    }

    ensureVertiportIcon() {
      if (!this.map) {
        return Promise.resolve(false);
      }
      if (this.vertiportIconPromise) {
        return this.vertiportIconPromise;
      }
      if (this.map.hasImage(VERTIPORT_ICON_ID)) {
        this.vertiportIconPromise = Promise.resolve(true);
        return this.vertiportIconPromise;
      }
      this.vertiportIconPromise = new Promise((resolve) => {
        this.map.loadImage("/resources/v_sign.png", (error, image) => {
          if (!error && image && !this.map.hasImage(VERTIPORT_ICON_ID)) {
            this.map.addImage(VERTIPORT_ICON_ID, image);
          }
          if (!error) {
            this.addVertiportIconLayer();
          }
          resolve(!error);
        });
      });
      return this.vertiportIconPromise;
    }

    updateVertiportOverlayFromRows(rows) {
      this.lastVertiportRows = rows;
      this.clearVertiportHover();
      this.setVertiportHoverFilter(null);
      if (!this.map || !this.map.isStyleLoaded()) {
        this.pendingVertiportRows = rows;
        return;
      }
      this.pendingVertiportRows = null;
      const data = this.buildVertiportFeatures(rows);
      const applyLayers = () => {
        this.setVertiportLayers(data);
        this.setVertiportLabels(data.points);
        this.map.triggerRepaint();
      };
      if (this.map.hasImage(VERTIPORT_ICON_ID)) {
        applyLayers();
        return;
      }
      this.ensureVertiportIcon().then(applyLayers);
    }

    buildVertiportFeatures(rows) {
      const points = [];
      const lines = [];
      const lookup = new Map();
      const corridorLookup = this.corridorPointLookup || new Map();

      rows.forEach((row) => {
        const name = readCell(row, 0);
        const lat = Number.parseFloat(readCell(row, 2));
        const lon = Number.parseFloat(readCell(row, 3));
        if (!name || !Number.isFinite(lat) || !Number.isFinite(lon)) {
          return;
        }
        const entry = { id: name, name, coord: [lon, lat], className: readCell(row, 1) };
        lookup.set(name, entry);
        points.push(entry);
      });

      const seen = new Set();
      let lineIndex = 0;
      rows.forEach((row) => {
        const name = readCell(row, 0);
        const startEntry = lookup.get(name);
        if (!startEntry) {
          return;
        }
        const linkCell = readCell(row, Math.max(0, row.length - 1));
        if (!linkCell) {
          return;
        }
        linkCell.split(",").forEach((raw) => {
          const target = raw.trim();
          if (!target) {
            return;
          }
          const endEntry = corridorLookup.get(target) || lookup.get(target);
          if (!endEntry) {
            return;
          }
          const key = [name, target].sort().join("|");
          if (seen.has(key)) {
            return;
          }
          seen.add(key);
          lines.push({
            id: lineIndex,
            from: name,
            to: target,
            start: startEntry,
            end: endEntry,
          });
          lineIndex += 1;
        });
      });

      return { points, lines };
    }

    buildVertiportPointGeoJson(points) {
      return {
        type: "FeatureCollection",
        features: points.map((entry) => ({
          type: "Feature",
          id: entry.id,
          geometry: {
            type: "Point",
            coordinates: entry.coord,
          },
          properties: {
            name: entry.name,
            class: entry.className || "",
          },
        })),
      };
    }

    buildVertiportLineGeoJson(lines) {
      return {
        type: "FeatureCollection",
        features: lines.map((line) => ({
          type: "Feature",
          id: line.id,
          geometry: {
            type: "LineString",
            coordinates: [line.start.coord, line.end.coord],
          },
          properties: {
            from: line.from,
            to: line.to,
            name: `${line.from} - ${line.to}`,
          },
        })),
      };
    }

    setVertiportLayers(data) {
      const map = this.map;
      if (!map) {
        return;
      }
      const pointSourceId = "vertiport-points";
      const lineSourceId = "vertiport-links";
      const pointData = this.buildVertiportPointGeoJson(data.points);
      const lineData = this.buildVertiportLineGeoJson(data.lines);

      if (map.getSource(pointSourceId)) {
        map.getSource(pointSourceId).setData(pointData);
      } else {
        map.addSource(pointSourceId, { type: "geojson", data: pointData });
      }

      if (map.getSource(lineSourceId)) {
        map.getSource(lineSourceId).setData(lineData);
      } else {
        map.addSource(lineSourceId, { type: "geojson", data: lineData });
      }

      const beforeId = map.getLayer("corridor-3d") ? "corridor-3d" : undefined;
      if (!map.getLayer("vertiport-links-line")) {
        map.addLayer(
          {
            id: "vertiport-links-line",
            type: "line",
            source: lineSourceId,
            layout: {
              "line-join": "round",
              "line-cap": "round",
            },
            paint: {
              "line-color": [
                "case",
                ["boolean", ["feature-state", "hover"], false],
                "#8dffbb",
                VERTIPORT_LINE_COLOR,
              ],
              "line-width": [
                "case",
                ["boolean", ["feature-state", "hover"], false],
                2 * MAP_SIZE_SCALE,
                1 * MAP_SIZE_SCALE,
              ],
              "line-dasharray": [1.5, 1.5],
            },
          },
          beforeId,
        );
      }

      if (!map.getLayer("vertiport-links-hit")) {
        map.addLayer(
          {
            id: "vertiport-links-hit",
            type: "line",
            source: lineSourceId,
            layout: {
              "line-join": "round",
              "line-cap": "round",
            },
            paint: {
              "line-color": "#000000",
              "line-opacity": 0.01,
              "line-width": 10 * MAP_SIZE_SCALE,
            },
          },
          beforeId,
        );
      }

      if (!map.getLayer("vertiport-circle")) {
        map.addLayer(
          {
            id: "vertiport-circle",
            type: "circle",
            source: pointSourceId,
            paint: {
              "circle-radius": [
                "case",
                ["boolean", ["feature-state", "hover"], false],
                10 * MAP_SIZE_SCALE,
                8 * MAP_SIZE_SCALE,
              ],
              "circle-color": [
                "case",
                ["boolean", ["feature-state", "hover"], false],
                "#f2fff8",
                "#ffffff",
              ],
              "circle-stroke-color": [
                "case",
                ["boolean", ["feature-state", "hover"], false],
                "#8dffbb",
                "#1a1a1a",
              ],
              "circle-stroke-width": [
                "case",
                ["boolean", ["feature-state", "hover"], false],
                3 * MAP_SIZE_SCALE,
                1 * MAP_SIZE_SCALE,
              ],
            },
          },
          beforeId,
        );
      }

      if (map.hasImage(VERTIPORT_ICON_ID)) {
        this.addVertiportIconLayer(beforeId);
      }
      this.addVertiportHoverRing(beforeId);
    }

    addVertiportHoverRing(beforeId) {
      const map = this.map;
      if (!map || !map.getSource("vertiport-points")) {
        return;
      }
      if (map.getLayer("vertiport-hover-ring")) {
        return;
      }
      map.addLayer(
        {
          id: "vertiport-hover-ring",
          type: "circle",
          source: "vertiport-points",
          filter: ["==", ["get", "name"], ""],
          paint: {
            "circle-radius": 12 * MAP_SIZE_SCALE,
            "circle-color": "#000000",
            "circle-opacity": 0,
            "circle-stroke-color": "#8dffbb",
            "circle-stroke-opacity": 0.95,
            "circle-stroke-width": 3.5 * MAP_SIZE_SCALE,
          },
        },
        beforeId,
      );
    }

    setVertiportHoverFilter(name) {
      if (!this.map || !this.map.getLayer("vertiport-hover-ring")) {
        return;
      }
      if (this.vertiportHoverName === name) {
        return;
      }
      this.vertiportHoverName = name || "";
      this.map.setFilter("vertiport-hover-ring", [
        "==",
        ["get", "name"],
        this.vertiportHoverName,
      ]);
    }

    clearVertiportLabels() {
      this.vertiportLabels.forEach((marker) => marker.remove());
      this.vertiportLabels = [];
    }

    setVertiportLabels(points) {
      if (!this.map) {
        return;
      }
      this.clearVertiportLabels();
      points.forEach((entry) => {
        const label = document.createElement("div");
        label.className = "vertiport-static-label";
        label.textContent = entry.name;
        const marker = new maplibregl.Marker({
          element: label,
          anchor: "top",
          offset: [0, 8 * MAP_SIZE_SCALE],
        })
          .setLngLat(entry.coord)
          .addTo(this.map);
        this.vertiportLabels.push(marker);
      });
    }

    addVertiportIconLayer(beforeId) {
      const map = this.map;
      if (!map || !map.getSource("vertiport-points")) {
        return;
      }
      if (map.getLayer("vertiport-icon")) {
        return;
      }
      map.addLayer(
        {
          id: "vertiport-icon",
          type: "symbol",
          source: "vertiport-points",
          layout: {
            "icon-image": VERTIPORT_ICON_ID,
            "icon-size": 0.04 * MAP_SIZE_SCALE,
            "icon-allow-overlap": true,
            "icon-anchor": "center",
            "icon-rotation-alignment": "viewport",
            "icon-pitch-alignment": "viewport",
          },
        },
        beforeId,
      );
    }

    async loadCorridorOverlay() {
      if (!this.map) {
        return;
      }
      try {
        const rows = await this.fetchCsvRows(this.config.data.waypointCsv);
        this.updateCorridorOverlayFromRows(rows);
      } catch (error) {
        console.warn("Failed to load corridor overlay.", error);
      }
    }

    ensureCorridorReady() {
      if (!this.map || this.corridorEnsured) {
        return;
      }
      this.corridorEnsured = true;
      if (this.pendingCorridorRows) {
        const rows = this.pendingCorridorRows;
        this.pendingCorridorRows = null;
        this.updateCorridorOverlayFromRows(rows);
        return;
      }
      if (this.lastCorridorRows) {
        this.updateCorridorOverlayFromRows(this.lastCorridorRows);
        return;
      }
      const info = this.defaultFiles.corridor;
      if (info && info.url) {
        this.loadCorridorData(info.url);
      }
    }

    updateCorridorOverlayFromRows(rows) {
      this.lastCorridorRows = rows;
      if (!this.map || !this.map.isStyleLoaded()) {
        this.pendingCorridorRows = rows;
        return;
      }
      this.pendingCorridorRows = null;
      const data = this.buildCorridorFeatures(rows);
      this.corridorPointLookup = new Map(data.points.map((entry) => [entry.name, entry]));
      this.corridorHitData = this.buildCorridorHitData(data);
      this.clearCorridorHover();
      const hasLayer =
        this.corridorLayer && this.map.getLayer && this.map.getLayer(this.corridorLayer.id);
      if (hasLayer && this.corridorLayer.updateBuffers) {
        const buffers = this.buildCorridorBuffers(data);
        this.corridorLayer.updateBuffers(buffers);
      } else {
        this.setCorridorLayer(data);
      }
      this.updateCorridorTheme(this.currentTheme);
      this.map.triggerRepaint();
      requestAnimationFrame(() => {
        if (this.map) {
          this.map.triggerRepaint();
        }
      });
      if (this.lastVertiportRows) {
        this.updateVertiportOverlayFromRows(this.lastVertiportRows);
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

    buildCorridorHitData(data) {
      const points = data.points.map((entry, index) => {
        const altitude = entry.altitude_m ?? 0;
        const mercator = maplibregl.MercatorCoordinate.fromLngLat(entry.coord, altitude);
        return {
          index,
          name: entry.name,
          coord: entry.coord,
          altitude_m: altitude,
          mercator,
        };
      });
      const lines = data.lines.map((line, index) => {
        const startAlt = line.start.altitude_m ?? 0;
        const endAlt = line.end.altitude_m ?? 0;
        const mercStart = maplibregl.MercatorCoordinate.fromLngLat(line.start.coord, startAlt);
        const mercEnd = maplibregl.MercatorCoordinate.fromLngLat(line.end.coord, endAlt);
        return {
          index,
          name: `${line.from} - ${line.to}`,
          start: line.start,
          end: line.end,
          mercStart,
          mercEnd,
        };
      });
      return { points, lines };
    }

    projectMercatorToScreen(mercator, matrix) {
      if (!this.map || !mercator || !matrix) {
        return null;
      }
      const x = mercator.x;
      const y = mercator.y;
      const z = mercator.z;
      const w = 1;
      const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12] * w;
      const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13] * w;
      const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15] * w;
      if (!Number.isFinite(clipW) || clipW === 0) {
        return null;
      }
      const ndcX = clipX / clipW;
      const ndcY = clipY / clipW;
      const canvas = this.map.getCanvas();
      const pixelRatio =
        canvas.clientWidth > 0 ? canvas.width / canvas.clientWidth : window.devicePixelRatio || 1;
      const width = canvas.width / pixelRatio;
      const height = canvas.height / pixelRatio;
      return {
        x: (ndcX + 1) * 0.5 * width,
        y: (1 - ndcY) * 0.5 * height,
      };
    }

    distanceToSegment(point, start, end) {
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      if (dx === 0 && dy === 0) {
        return Math.hypot(point.x - start.x, point.y - start.y);
      }
      const t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy);
      const clamped = Math.max(0, Math.min(1, t));
      const projX = start.x + clamped * dx;
      const projY = start.y + clamped * dy;
      return Math.hypot(point.x - projX, point.y - projY);
    }

    findCorridorHoverTarget(pointer, matrix) {
      if (!this.corridorHitData) {
        return null;
      }
      const pointThreshold = 10 * MAP_SIZE_SCALE;
      const lineThreshold = 8 * MAP_SIZE_SCALE;
      let bestPoint = null;
      let bestPointDist = Infinity;
      this.corridorHitData.points.forEach((entry) => {
        const screen = this.projectMercatorToScreen(entry.mercator, matrix);
        if (!screen) {
          return;
        }
        const dist = Math.hypot(pointer.x - screen.x, pointer.y - screen.y);
        if (dist <= pointThreshold && dist < bestPointDist) {
          bestPointDist = dist;
          bestPoint = {
            type: "point",
            index: entry.index,
            name: entry.name,
          };
        }
      });
      if (bestPoint) {
        return bestPoint;
      }
      let bestLine = null;
      let bestLineDist = Infinity;
      this.corridorHitData.lines.forEach((entry) => {
        const start = this.projectMercatorToScreen(entry.mercStart, matrix);
        const end = this.projectMercatorToScreen(entry.mercEnd, matrix);
        if (!start || !end) {
          return;
        }
        const dist = this.distanceToSegment(pointer, start, end);
        if (dist <= lineThreshold && dist < bestLineDist) {
          bestLineDist = dist;
          bestLine = {
            type: "line",
            index: entry.index,
            name: entry.name,
          };
        }
      });
      return bestLine;
    }

    createCorridorLayer(data) {
      const buffers = this.buildCorridorBuffers(data);
      const color = this.getCorridorColor(this.currentTheme);
      const pointSize = 6 * MAP_SIZE_SCALE * (window.devicePixelRatio || 1);
      const layer = {
        id: "corridor-3d",
        type: "custom",
        renderingMode: "3d",
        _color: color,
        _pointSize: pointSize,
        _hoverPoint: -1,
        _hoverLine: -1,
        _highlightColor: HOVER_OUTLINE_COLOR,
        _hoverPointSize: 6 * MAP_SIZE_SCALE * (window.devicePixelRatio || 1),
        _ringWidth: 0.12,
        setColor(nextColor) {
          this._color = nextColor;
        },
        updateBuffers(nextBuffers) {
          this._pendingBuffers = nextBuffers;
          if (!this._gl || !this._pointBuffer || !this._lineBuffer) {
            return;
          }
          const gl = this._gl;
          const pointData = new Float32Array(nextBuffers.pointPositions);
          gl.bindBuffer(gl.ARRAY_BUFFER, this._pointBuffer);
          gl.bufferData(gl.ARRAY_BUFFER, pointData, gl.STATIC_DRAW);
          this._pointCount = pointData.length / 3;

          const lineData = new Float32Array(nextBuffers.linePositions);
          gl.bindBuffer(gl.ARRAY_BUFFER, this._lineBuffer);
          gl.bufferData(gl.ARRAY_BUFFER, lineData, gl.STATIC_DRAW);
          this._lineCount = lineData.length / 3;
          this._pendingBuffers = null;
        },
        setHover(hover) {
          this._hoverPoint = hover && hover.type === "point" ? hover.index : -1;
          this._hoverLine = hover && hover.type === "line" ? hover.index : -1;
        },
        clearHover() {
          this._hoverPoint = -1;
          this._hoverLine = -1;
        },
        getMatrix() {
          return this._lastMatrix;
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
            uniform float u_ring;
            uniform float u_ringWidth;
            void main() {
              if (u_isPoint > 0.5) {
                float dist = length(gl_PointCoord - vec2(0.5));
                if (dist > 0.5) {
                  discard;
                }
                if (u_ring > 0.5) {
                  if (dist < (0.5 - u_ringWidth)) {
                    discard;
                  }
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
          this._uRing = gl.getUniformLocation(program, "u_ring");
          this._uRingWidth = gl.getUniformLocation(program, "u_ringWidth");

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

          if (this._pendingBuffers) {
            this.updateBuffers(this._pendingBuffers);
          }
        },
        render(gl, matrix) {
          if (!this._program) {
            return;
          }
          this._lastMatrix = matrix;
          gl.useProgram(this._program);
          gl.uniformMatrix4fv(this._uMatrix, false, matrix);
          gl.uniform1f(this._uPointSize, this._pointSize);
          gl.uniform1f(this._uRing, 0);
          gl.uniform1f(this._uRingWidth, this._ringWidth);
          gl.enableVertexAttribArray(this._aPos);
          gl.enable(gl.BLEND);
          gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

          const color = hexToRgba(this._color, 0.95);
          gl.uniform4fv(this._uColor, color);

          if (this._lineCount > 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this._lineBuffer);
            gl.vertexAttribPointer(this._aPos, 3, gl.FLOAT, false, 0, 0);
            gl.uniform1f(this._uIsPoint, 0);
            gl.lineWidth(1 * MAP_SIZE_SCALE);
            gl.drawArrays(gl.LINES, 0, this._lineCount);
          }

          if (this._pointCount > 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this._pointBuffer);
            gl.vertexAttribPointer(this._aPos, 3, gl.FLOAT, false, 0, 0);
            gl.uniform1f(this._uIsPoint, 1);
            gl.drawArrays(gl.POINTS, 0, this._pointCount);
          }

          const highlightColor = hexToRgba(this._highlightColor, 0.95);
          if (this._hoverLine > -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this._lineBuffer);
            gl.vertexAttribPointer(this._aPos, 3, gl.FLOAT, false, 0, 0);
            gl.uniform4fv(this._uColor, highlightColor);
            gl.uniform1f(this._uIsPoint, 0);
            gl.uniform1f(this._uRing, 0);
            gl.lineWidth(2 * MAP_SIZE_SCALE);
            gl.drawArrays(gl.LINES, this._hoverLine * 2, 2);
          }

          if (this._hoverPoint > -1) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this._pointBuffer);
            gl.vertexAttribPointer(this._aPos, 3, gl.FLOAT, false, 0, 0);
            gl.uniform4fv(this._uColor, highlightColor);
            gl.uniform1f(this._uIsPoint, 1);
            gl.uniform1f(this._uRing, 1);
            gl.uniform1f(this._uPointSize, this._pointSize + this._hoverPointSize);
            gl.drawArrays(gl.POINTS, this._hoverPoint, 1);
          }
        },
      };
      return layer;
    }

    setupCorridorHover() {
      if (!this.map) {
        return;
      }
      const label = document.createElement("div");
      label.className = "corridor-hover-label";
      this.map.getContainer().appendChild(label);
      this.corridorHoverLabel = label;

      this.map.on("mousemove", (event) => this.handleCorridorHover(event));
      this.map.on("mouseleave", () => this.clearCorridorHover());
      this.map.on("movestart", () => this.clearCorridorHover());
      this.map.on("dragstart", () => this.clearCorridorHover());
      this.map.on("zoomstart", () => this.clearCorridorHover());
      this.map.on("pitchstart", () => this.clearCorridorHover());
      this.map.on("rotatestart", () => this.clearCorridorHover());
    }

    handleCorridorHover(event) {
      if (!this.map || !this.corridorLayer || !this.corridorLayer.getMatrix) {
        return;
      }
      const matrix = this.corridorLayer.getMatrix();
      if (!matrix) {
        return;
      }
      const target = this.findCorridorHoverTarget(event.point, matrix);
      if (!target) {
        this.clearCorridorHover();
        return;
      }

      const isSame =
        this.corridorHover &&
        this.corridorHover.type === target.type &&
        this.corridorHover.index === target.index;
      this.corridorHover = target;
      if (this.corridorLayer.setHover) {
        this.corridorLayer.setHover(target);
      }
      if (!isSame) {
        this.map.triggerRepaint();
      }
      this.showCorridorLabel(`C : ${target.name}`, event.point);
      this.map.getCanvas().style.cursor = "pointer";
    }

    showCorridorLabel(text, point) {
      if (!this.corridorHoverLabel) {
        return;
      }
      this.corridorHoverLabel.textContent = text;
      this.corridorHoverLabel.style.left = `${point.x}px`;
      this.corridorHoverLabel.style.top = `${point.y}px`;
      this.corridorHoverLabel.classList.add("is-visible");
    }

    clearCorridorHover() {
      if (!this.map) {
        return;
      }
      const hadHover = Boolean(this.corridorHover);
      this.corridorHover = null;
      if (this.corridorLayer && this.corridorLayer.clearHover) {
        this.corridorLayer.clearHover();
      }
      if (hadHover) {
        this.map.triggerRepaint();
      }
      if (this.corridorHoverLabel) {
        this.corridorHoverLabel.classList.remove("is-visible");
      }
      this.map.getCanvas().style.cursor = "";
    }

    setupVertiportHover() {
      if (!this.map) {
        return;
      }
      const label = document.createElement("div");
      label.className = "vertiport-hover-label";
      this.map.getContainer().appendChild(label);
      this.vertiportHoverLabel = label;

      this.map.on("mousemove", (event) => this.handleVertiportHover(event));
      this.map.on("mouseleave", () => this.clearVertiportHover());
      this.map.on("movestart", () => this.clearVertiportHover());
      this.map.on("dragstart", () => this.clearVertiportHover());
      this.map.on("zoomstart", () => this.clearVertiportHover());
      this.map.on("pitchstart", () => this.clearVertiportHover());
      this.map.on("rotatestart", () => this.clearVertiportHover());
    }

    handleVertiportHover(event) {
      if (!this.map) {
        return;
      }
      if (this.corridorHover) {
        this.clearVertiportHover();
        return;
      }
      const pointLayers = [];
      if (this.map.getLayer("vertiport-icon")) {
        pointLayers.push("vertiport-icon");
      }
      if (this.map.getLayer("vertiport-circle")) {
        pointLayers.push("vertiport-circle");
      }
      if (pointLayers.length) {
        const features = this.map.queryRenderedFeatures(event.point, {
          layers: pointLayers,
        });
        if (features.length) {
          const feature = features[0];
          const name = feature && feature.properties ? String(feature.properties.name || "") : "";
          if (name) {
            const id = feature.id != null ? feature.id : name;
            if (this.vertiportLinkHoverId != null) {
              this.setVertiportLinkHoverState(null);
            }
            if (this.vertiportHoverId !== id) {
              this.setVertiportHoverState(id);
            }
            this.setVertiportHoverFilter(name);
            this.vertiportHover = { id, name };
            this.vertiportLinkHover = null;
            this.showVertiportLabel(`V : ${name}`, event.point);
            this.map.getCanvas().style.cursor = "pointer";
            return;
          }
        }
      }

      if (this.map.getLayer("vertiport-links-line")) {
        const features = this.map.queryRenderedFeatures(event.point, {
          layers: this.map.getLayer("vertiport-links-hit")
            ? ["vertiport-links-hit"]
            : ["vertiport-links-line"],
        });
        if (features.length) {
          const feature = features[0];
          const name = feature && feature.properties ? String(feature.properties.name || "") : "";
          if (name) {
            const id = feature.id != null ? feature.id : null;
            if (this.vertiportHoverId != null) {
              this.setVertiportHoverState(null);
            }
            if (id != null && this.vertiportLinkHoverId !== id) {
              this.setVertiportLinkHoverState(id);
            }
            this.setVertiportHoverFilter(null);
            this.vertiportHover = null;
            this.vertiportLinkHover = { id, name };
            this.showVertiportLabel(`VL : ${name}`, event.point);
            this.map.getCanvas().style.cursor = "pointer";
            return;
          }
        }
      }

      this.clearVertiportHover();
    }

    setVertiportHoverState(nextId) {
      if (!this.map) {
        return;
      }
      const sourceId = "vertiport-points";
      if (this.map.isStyleLoaded() && this.map.getSource(sourceId) && this.vertiportHoverId != null) {
        this.map.setFeatureState(
          { source: sourceId, id: this.vertiportHoverId },
          { hover: false },
        );
      }
      if (this.map.isStyleLoaded() && this.map.getSource(sourceId) && nextId != null) {
        this.map.setFeatureState({ source: sourceId, id: nextId }, { hover: true });
      }
      this.vertiportHoverId = nextId;
    }

    setVertiportLinkHoverState(nextId) {
      if (!this.map) {
        return;
      }
      const sourceId = "vertiport-links";
      if (
        this.map.isStyleLoaded() &&
        this.map.getSource(sourceId) &&
        this.vertiportLinkHoverId != null
      ) {
        this.map.setFeatureState(
          { source: sourceId, id: this.vertiportLinkHoverId },
          { hover: false },
        );
      }
      if (this.map.isStyleLoaded() && this.map.getSource(sourceId) && nextId != null) {
        this.map.setFeatureState({ source: sourceId, id: nextId }, { hover: true });
      }
      this.vertiportLinkHoverId = nextId;
    }

    showVertiportLabel(text, point) {
      if (!this.vertiportHoverLabel) {
        return;
      }
      this.vertiportHoverLabel.textContent = text;
      this.vertiportHoverLabel.style.left = `${point.x}px`;
      this.vertiportHoverLabel.style.top = `${point.y}px`;
      this.vertiportHoverLabel.classList.add("is-visible");
    }

    clearVertiportHover() {
      if (!this.map) {
        return;
      }
      if (
        this.vertiportHoverId != null &&
        this.map.isStyleLoaded() &&
        this.map.getSource("vertiport-points")
      ) {
        this.map.setFeatureState(
          { source: "vertiport-points", id: this.vertiportHoverId },
          { hover: false },
        );
      }
      if (
        this.vertiportLinkHoverId != null &&
        this.map.isStyleLoaded() &&
        this.map.getSource("vertiport-links")
      ) {
        this.map.setFeatureState(
          { source: "vertiport-links", id: this.vertiportLinkHoverId },
          { hover: false },
        );
      }
      this.vertiportHover = null;
      this.vertiportHoverId = null;
      this.vertiportLinkHover = null;
      this.vertiportLinkHoverId = null;
      this.setVertiportHoverFilter(null);
      if (this.vertiportHoverLabel) {
        this.vertiportHoverLabel.classList.remove("is-visible");
      }
      if (!this.corridorHover) {
        this.map.getCanvas().style.cursor = "";
      }
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
