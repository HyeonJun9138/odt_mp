(() => {
  const body = document.body;
  const config = {
    tileUrl: body.dataset.tileUrl,
    minZoom: Number(body.dataset.minZoom),
    maxZoom: Number(body.dataset.maxZoom),
    center: [Number(body.dataset.centerLon), Number(body.dataset.centerLat)],
    startZoom: Number(body.dataset.startZoom),
    bounds: body.dataset.bounds ? JSON.parse(body.dataset.bounds) : null,
  };

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
      };
      this.currentTheme = "light";
    }

    init() {
      this.map = this.createMap();
      this.bindThemeButtons();
      this.bindActionButtons();
      this.bindCloseButtons();
      this.applyTheme(this.currentTheme);
      this.setupScaleObserver();
    }

    createMap() {
      const style = this.buildStyle();
      const map = new maplibregl.Map({
        container: "map",
        style: style,
        center: this.config.center,
        zoom: this.config.startZoom,
        minZoom: this.config.minZoom,
        maxZoom: this.config.maxZoom,
        attributionControl: false,
      });

      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

      if (this.config.bounds) {
        map.once("load", () => {
          map.fitBounds(this.config.bounds, { padding: 20, duration: 0 });
        });
      }
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

    handleAction(action) {
      if (action === "vertiport") {
        this.togglePanel("vertiport");
      }
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
    }

    setupScaleObserver() {
      const root = document.documentElement;
      const app = document.getElementById("app");
      const baseWidth = 1200;
      const baseHeight = 900;

      const updateScale = () => {
        const rect = app.getBoundingClientRect();
        const scale = Math.min(rect.width / baseWidth, rect.height / baseHeight);
        const clamped = Math.max(0.75, Math.min(scale, 1.1));
        root.style.setProperty("--ui-scale", clamped.toFixed(3));
        if (this.map) {
          this.map.resize();
        }
      };

      updateScale();
      window.addEventListener("resize", updateScale);

      if (window.ResizeObserver) {
        const observer = new ResizeObserver(updateScale);
        observer.observe(app);
      }
    }
  }

  const app = new MapApp(config);
  app.init();
})();
