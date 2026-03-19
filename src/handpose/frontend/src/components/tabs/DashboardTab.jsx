import { useCallback, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { List, CheckCircle2, XCircle, RefreshCw, ImageIcon } from 'lucide-react';

import { API_URL, DEFAULT_CONFIG } from '../../api/config';

// -------------------------------------------------------------------------
// DASHBOARD TAB KOMPONENTE
// -------------------------------------------------------------------------

/**
 * Rendert den Dashboard-Tab mit KPIs, Teile-Status und Frame-Inspektor.
 *
 * @param {Object} props - Komponenten-Parameter.
 * @param {(message: string) => void} props.addLog - Callback für Log-Einträge.
 * @param {Object} props.enabledModels - Konfiguration der aktivierten ML-Modelle.
 * @param {Object} props.config - App-Konfiguration (cluster_names, etc.).
 * @returns {JSX.Element} Dashboard-Ansicht.
 */
export default function DashboardTab({ addLog, enabledModels, config }) {
  const [availableRuns, setAvailableRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [dashboardData, setDashboardData] = useState(null);
  const [frames, setFrames] = useState([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  const [isLoadingData, setIsLoadingData] = useState(false);

  // -------------------------------------------------------------------------
  // LIFECYCLE & DATA FETCHING
  // -------------------------------------------------------------------------

  const loadRunData = useCallback(
    async (runId) => {
      if (!runId) return;
      setSelectedRun(runId);
      setIsLoadingData(true);

      try {
        const metricsRes = await axios.get(`${API_URL}/results/${runId}`, {
          timeout: 10000,
        });
        setDashboardData(metricsRes.data);
      } catch {
        setDashboardData(null);
        addLog('KPI-Daten konnten nicht geladen werden.');
      } finally {
        setIsLoadingData(false);
      }

      setFrames([]);
      setFrameIndex(0);
    },
    [addLog]
  );

  const loadFramesForRun = useCallback(
    async (runId) => {
      if (!runId) return;
      try {
        const framesRes = await axios.get(`${API_URL}/frames/${runId}`);
        setFrames(framesRes.data || []);
        setFrameIndex(0);
      } catch {
        setFrames([]);
        addLog('Frames konnten nicht geladen werden.');
      }
    },
    [addLog]
  );

  const fetchRuns = useCallback(async () => {
    setIsLoadingRuns(true);
    try {
      const response = await axios.get(`${API_URL}/results`, {
        timeout: 10000,
      });
      setAvailableRuns(response.data);
      if (response.data.length > 0) {
        await loadRunData(response.data[0]);
      }
    } catch {
      addLog('Fehler beim Laden der Runs.');
    } finally {
      setIsLoadingRuns(false);
    }
  }, [addLog, loadRunData]);

  useEffect(() => {
    void fetchRuns();
  }, [fetchRuns]);

  // -------------------------------------------------------------------------
  // EVENT HANDLERS
  // -------------------------------------------------------------------------

  const handleExtractFrames = async () => {
    if (!selectedRun) return;
    setIsExtracting(true);
    try {
      await axios.post(`${API_URL}/recordings/${selectedRun}/frames`);
      addLog(`Frames extrahiert für: ${selectedRun}`);
      await loadFramesForRun(selectedRun);
    } catch {
      addLog('Fehler bei Frame-Extraktion.');
    } finally {
      setIsExtracting(false);
    }
  };

  // -------------------------------------------------------------------------
  // DATA EXTRACTION
  // -------------------------------------------------------------------------

  const totalDuration = dashboardData?.total_duration ?? 0;
  const netTime = dashboardData?.assembly_time_net ?? 0;
  const anomalyCount = dashboardData?.anomaly_count ?? 0;

  // KPI-Cards basierend auf aktivierten Modellen
  const kpiCards = useMemo(() => {
    const cards = [];

    // HMM-basierte KPIs (Zeiten)
    if (enabledModels.enable_hmm) {
      cards.push({
        id: 'total_duration',
        label: '⏱️ Gesamtdauer',
        value: `${Number(totalDuration).toFixed(2)} s`,
        borderColor: 'border-indigo-500',
      });
      cards.push({
        id: 'assembly_time_net',
        label: '⚙️ Netto-Zeit',
        value: `${Number(netTime).toFixed(2)} s`,
        borderColor: 'border-green-500',
      });
    }

    // VAE-basierte KPIs (Anomalien)
    if (enabledModels.enable_vae) {
      cards.push({
        id: 'anomaly_count',
        label: '⚠️ Anomalien',
        value: String(anomalyCount),
        borderColor: 'border-red-500',
      });
    }

    return cards;
  }, [enabledModels, totalDuration, netTime, anomalyCount]);

  const detectedPartNames = useMemo(
    () => Object.keys(dashboardData?.parts_summary ?? {}).map((name) => name.toUpperCase()),
    [dashboardData]
  );

  const clusterNames = useMemo(
    () => {
      const names = config?.cluster_names ?? '';
      if (!names) return [];
      return names
        .split(',')
        .map((name) => name.trim())
        .filter((name) => name.length > 0);
    },
    [config]
  );

  // -------------------------------------------------------------------------
  // RENDER
  // -------------------------------------------------------------------------

  return (
    <div className="animate-in space-y-6 fade-in duration-500">
      {/* KPI Cards */}
      {isLoadingData ? (
        <div className="grid grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="rounded-xl border-l-4 border-gray-200 bg-white p-6 shadow-xl animate-pulse">
              <p className="text-[10px] font-bold uppercase text-gray-300">Lädt...</p>
              <span className="mt-2 block h-8 w-24 bg-gray-200 rounded" />
            </div>
          ))}
        </div>
      ) : kpiCards.length > 0 ? (
        <div className={`grid gap-6 ${kpiCards.length === 1 ? 'grid-cols-1 max-w-md' : kpiCards.length === 2 ? 'grid-cols-2' : 'grid-cols-3'}`}>
          {kpiCards.map((card) => (
            <div key={card.id} className={`rounded-xl border-l-4 ${card.borderColor} bg-white p-6 shadow-xl`}>
              <p className="text-[10px] font-bold uppercase text-gray-400">{card.label}</p>
              <span className="text-2xl font-black text-[#32325d]">{card.value}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="rounded-xl border-2 border-dashed border-gray-300 bg-gray-50 p-8 text-center">
          <p className="text-sm text-gray-500">
            Keine KPIs verfügbar - alle Modelle sind deaktiviert
          </p>
        </div>
      )}

      {/* Teile-Erkennung & Frame-Inspektor */}
      <div className="grid grid-cols-1 gap-6 pb-10 lg:grid-cols-3">
        {/* Teile-Erkennung - nur anzeigen wenn HMM aktiviert */}
        {enabledModels.enable_hmm && (
          <div className="h-fit rounded-xl bg-white p-6 shadow-xl">
            <h3 className="mb-6 flex items-center gap-2 border-b pb-2 text-xs font-bold uppercase">
              <List size={14} /> Teile-Erkennung
            </h3>
            <div className="space-y-3">
              {clusterNames.map((partName) => {
                const isDetected = detectedPartNames.includes(partName.toUpperCase());
                return (
                  <div
                    key={partName}
                    className="flex items-center justify-between rounded bg-slate-50 p-2"
                  >
                    <span className="text-sm font-bold">{partName}</span>
                    {isDetected ? (
                      <div className="flex items-center gap-2 text-green-500 text-[10px] font-bold">
                        GEFUNDEN <CheckCircle2 size={18} />
                      </div>
                    ) : (
                      <XCircle className="text-red-400" size={18} />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Frame-Inspektor */}
        <div className={`rounded-xl bg-white p-6 shadow-xl ${enabledModels.enable_hmm ? 'lg:col-span-2' : 'lg:col-span-3'}`}>
          <h3 className="mb-4 text-xs font-bold uppercase">📸 Frame Inspektor</h3>

          <div className="flex items-end gap-6 rounded-xl bg-slate-50 border p-6">
            <div className="flex-1">
              <label className="text-[10px] font-bold uppercase text-gray-400">
                Run auswählen
              </label>
              <select
                className="mt-1 w-full border-b bg-transparent py-2 font-bold text-[#32325d] outline-none disabled:opacity-50"
                value={selectedRun}
                onChange={(e) => loadRunData(e.target.value)}
                disabled={isLoadingRuns}
              >
                {isLoadingRuns ? (
                  <option>Lädt Runs...</option>
                ) : availableRuns.length === 0 ? (
                  <option>Keine Runs vorhanden</option>
                ) : (
                  availableRuns.map((runId) => (
                    <option key={runId} value={runId}>
                      {runId}
                    </option>
                  ))
                )}
              </select>
            </div>
            <button
              onClick={handleExtractFrames}
              disabled={isExtracting || isLoadingData || !selectedRun}
              className="flex items-center gap-2 rounded bg-[#5e72e4] px-6 py-2.5 text-[10px] font-bold uppercase text-white disabled:opacity-50"
            >
              {isExtracting ? (
                <RefreshCw className="animate-spin" size={14} />
              ) : (
                <ImageIcon size={14} />
              )}
              {isExtracting ? 'Extrahiere...' : 'Frames extrahieren'}
            </button>
          </div>

          <div className="mt-8">
            {frames.length > 0 ? (
              <div className="flex flex-col items-center">
                <div className="mb-6 aspect-[4/3] w-full max-w-2xl overflow-hidden rounded bg-black">
                  <img
                    src={`${API_URL}/frame/${selectedRun}/${frames[frameIndex]}`}
                    className="h-full w-full object-cover"
                    alt={`Frame ${frameIndex + 1}`}
                  />
                </div>
                <input
                  type="range"
                  className="w-full cursor-pointer accent-[#5e72e4] outline-none focus:outline-none focus:ring-0"
                  min="0"
                  max={frames.length - 1}
                  value={frameIndex}
                  onChange={(e) => setFrameIndex(parseInt(e.target.value, 10))}
                />
                <div className="mt-2 text-sm font-bold text-gray-600">
                  {frameIndex + 1} / {frames.length}
                </div>
              </div>
            ) : (
              <div className="mt-6 flex h-64 items-center justify-center rounded border-2 border-dashed text-gray-300 text-sm">
                Keine Frames vorhanden. Bitte auf "Frames extrahieren" klicken.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
