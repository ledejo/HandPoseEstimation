import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { Save, ScanLine, RotateCcw, Loader2, Video } from 'lucide-react';

import { API_URL, DEFAULT_CONFIG } from '../../api/config';

// -------------------------------------------------------------------------
// CONFIG TAB KOMPONENTE
// -------------------------------------------------------------------------

/**
 * Parse-Helfer für ganzzahlige Eingaben mit Fallback.
 *
 * @param {unknown} value - Zu parsende Eingabe.
 * @param {number} fallback - Rückgabewert bei ungültiger Eingabe.
 * @returns {number} Geparste Zahl oder Fallback.
 */
function parseIntegerOrFallback(value, fallback) {
  const parsedValue = Number.parseInt(String(value), 10);
  return Number.isNaN(parsedValue) ? fallback : parsedValue;
}

/**
 * Rendert den Konfigurations-Tab inklusive Kamera-Ausrichtung.
 *
 * @param {Object} props - Komponenten-Parameter.
 * @param {(message: string) => void} props.addLog - Callback für Log-Einträge.
 * @param {Object} props.config - App-Konfiguration.
 * @param {Function} props.setConfig - Callback zum Aktualisieren der Config.
 * @param {Object} props.enabledModels - Konfiguration der aktivierten Modelle.
 * @returns {JSX.Element} Konfigurationsansicht.
 */
export default function ConfigTab({ addLog, config: initialConfig, setConfig: updateConfig, enabledModels }) {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [isAligning, setIsAligning] = useState(false);

  // Lokalen State mit initialConfig synchronisieren
  useEffect(() => {
    if (initialConfig) {
      setConfig(initialConfig);
    }
  }, [initialConfig]);

  const clusterCount = useMemo(
    () => Math.max(1, parseIntegerOrFallback(config.dbscan_clusters, 1)),
    [config.dbscan_clusters]
  );

  // -------------------------------------------------------------------------
  // EVENT HANDLERS
  // -------------------------------------------------------------------------

  /**
   * Speichert die Konfiguration und validiert Cluster-Namen.
   */
  const saveConfig = async () => {
    const clusterCountLocal = Math.max(1, parseIntegerOrFallback(config.dbscan_clusters, 1));
    const trimmedNames = config.cluster_names
      .split(',')
      .map((name) => name.trim())
      .slice(0, clusterCountLocal)
      .join(', ');

    const payload = { ...config, cluster_names: trimmedNames };

    try {
      await axios.post(`${API_URL}/config`, payload);
      setConfig(payload);
      updateConfig(payload); // Aktualisiere auch den globalen State
      addLog('Konfiguration gespeichert.');
    } catch {
      addLog('Fehler beim Speichern der Konfiguration.');
    }
  };

  /**
   * Aktualisiert einen einzelnen Cluster-Namen.
   */
  const handleClusterNameChange = (clusterIndex, value) => {
    const names = config.cluster_names.split(',').map((name) => name.trim());
    while (names.length <= clusterIndex) names.push('');
    names[clusterIndex] = value;
    setConfig({ ...config, cluster_names: names.join(', ') });
  };

  /**
   * Aktualisiert den Kamera-Typ.
   */
  const handleCameraTypeChange = (value) => {
    setConfig({ ...config, camera_type: parseInt(value, 10) });
  };

  /**
   * Startet die automatische Kamera-Ausrichtung via ArUco-Marker.
   * RESTful POST /api/camera/alignment
   */
  const handleAlignCamera = async () => {
    setIsAligning(true);
    addLog('Starte Ausrichtung...');
    try {
      const res = await axios.post(`${API_URL}/camera/alignment`);

      if (res.data.success) {
        addLog('Ausrichtung erfolgreich!');
      } else {
        addLog('Standardeinstellung wird verwendet.');
      }
    } catch {
      addLog('Standardeinstellung wird verwendet.');
    } finally {
      setIsAligning(false);
    }
  };

  /**
   * Setzt die Kamera-Ausrichtung zurück auf Standard.
   * RESTful DELETE /api/camera/reset-alignment
   */
  const handleResetAlignment = async () => {
    try {
      await axios.delete(`${API_URL}/camera/reset-alignment`);
      addLog('Standardeinstellung wird verwendet.');
    } catch {
      addLog('Fehler beim Zurücksetzen.');
    }
  };

  /**
   * Blendet den Fallback-Container ein, wenn der Preview-Stream nicht lädt.
   *
   * @param {React.SyntheticEvent<HTMLImageElement>} event - Fehler-Event des Bildes.
   */
  const handlePreviewError = (event) => {
    const imageElement = event.currentTarget;
    imageElement.style.display = 'none';

    const fallbackElement = imageElement.nextElementSibling;
    if (fallbackElement instanceof HTMLElement) {
      fallbackElement.style.display = 'flex';
    }
  };

  return (
    <div className="relative mx-auto w-full max-w-5xl rounded-xl bg-white shadow-2xl mb-10 p-10">
      <div className="mx-auto max-w-3xl">

        {/* Nur anzeigen wenn Kalibrierung aktiviert ist */}
        {enabledModels?.enable_calibration && (
          <div className="mb-12">
            <h3 className="mb-6 border-b pb-4 text-xl font-black text-[#32325d]">Setup & Ausrichtung</h3>
            <div className="rounded-lg border bg-gray-50 p-6 flex flex-col gap-6">
              <div className="relative w-full overflow-hidden rounded-lg bg-black flex items-center justify-center min-h-[300px] border-2 border-gray-200">
                <img
                  src={`${API_URL}/video/preview`}
                  alt="Kamera Live-Feed"
                  className="object-contain w-full h-full max-h-[450px]"
                  onError={handlePreviewError}
                />
                <div className="absolute inset-0 hidden flex-col items-center justify-center text-gray-500">
                  <Video size={48} className="mb-2 opacity-50" />
                  <p className="text-sm font-medium">Kamera nicht verfügbar</p>
                  <p className="text-xs">Bitte Backend prüfen oder Aufnahme starten.</p>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                <div>
                  <h4 className="text-sm font-bold text-gray-700">Brett automatisch ausrichten</h4>
                  <p className="text-xs text-gray-500 mt-1 max-w-sm">
                    Legen Sie das Brett mit den 4 Markern ins Bild und klicken Sie auf Ausrichten.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleResetAlignment}
                    className="flex items-center justify-center rounded-lg bg-gray-200 px-4 py-2.5 text-xs font-black uppercase text-gray-600 transition-all hover:bg-gray-300"
                    title="Zurücksetzen auf Standard"
                  >
                    <RotateCcw size={16} />
                  </button>
                  <button
                    onClick={handleAlignCamera}
                    disabled={isAligning}
                    className={`flex items-center gap-2 rounded-lg px-6 py-2.5 text-xs font-black uppercase text-white shadow-md transition-all ${
                      isAligning
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-[#5e72e4] hover:shadow-lg hover:-translate-y-0.5'
                    }`}
                  >
                    {isAligning ? <Loader2 size={16} className="animate-spin" /> : <ScanLine size={16} />}
                    {isAligning ? 'Suche...' : 'Ausrichten'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        <h3 className="mb-8 border-b pb-4 text-xl font-black text-[#32325d]">Prozess-Konfiguration</h3>

        <div className="mb-10 grid grid-cols-1 gap-4 md:grid-cols-2">
          {Array.from({ length: clusterCount }).map((_, i) => {
            const names = config.cluster_names.split(',').map((name) => name.trim());
            return (
              <div key={i}>
                <label className="text-[9px] font-bold uppercase text-gray-400">Box {i + 1}</label>
                <input
                  type="text"
                  className="w-full rounded-lg border bg-gray-50 p-2.5 text-sm outline-none focus:border-[#5e72e4]"
                  value={names[i] || ''}
                  onChange={(e) => handleClusterNameChange(i, e.target.value)}
                />
              </div>
            );
          })}
        </div>

        <div className="mb-10 space-y-2">
          <label className="text-[10px] font-black uppercase text-gray-400">Verwendete Kamera</label>
          <select
            className="w-full rounded-lg border bg-gray-50 p-3 outline-none focus:border-[#5e72e4]"
            value={config.camera_type ?? 1}
            onChange={(e) => handleCameraTypeChange(e.target.value)}
          >
            <option value="1">Externe Kamera</option>
            <option value="0">Interne Kamera</option>
          </select>
        </div>

        <button
          onClick={saveConfig}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-[#2dce89] px-12 py-3 text-xs font-black uppercase text-white shadow-lg transition-all hover:-translate-y-0.5 hover:shadow-2xl"
        >
          <Save size={16} /> Konfiguration speichern
        </button>
      </div>
    </div>
  );
}
