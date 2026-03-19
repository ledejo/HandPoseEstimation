import { useCallback, useEffect, useState } from 'react';
import axios from 'axios';

import { API_URL, MAX_LOG_ENTRIES } from './api/config';
import ConfigTab from './components/tabs/ConfigTab';
import DashboardTab from './components/tabs/DashboardTab';
import VideoTab from './components/tabs/VideoTab';
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';

// -------------------------------------------------------------------------
// KONSTANTEN
// -------------------------------------------------------------------------

const TAB_LIVE = 'live';
const TAB_TRAIN = 'train';
const TAB_DASHBOARD = 'dashboard';
const TAB_CONFIG = 'config';
const MODE_PRODUCTION = 'production';
const MODE_TRAINING = 'train';
const ACTION_START = 'start';
const ACTION_STOP = 'stop';

// -------------------------------------------------------------------------
// HAUPT-KOMPONENTE
// -------------------------------------------------------------------------

/**
 * Haupt-App-Komponente für die Hand Pose Estimation Anwendung.
 *
 * Verwaltet:
 * - Tab-Navigation (Live, Dashboard, Train, Config)
 * - Recording-Status und Modi
 * - Log-System für Benutzer-Feedback
 * - Kommunikation mit dem Backend-API
 */
export default function App() {
  const [activeTab, setActiveTab] = useState(TAB_LIVE);
  const [recordingMode, setRecordingMode] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [logs, setLogs] = useState(['Bereit']);
  const [config, setConfig] = useState(null);
  const [enabledModels, setEnabledModels] = useState({
    enable_hmm: true,
    enable_vae: true,
    enable_dbscan: true,
  });
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [backendError, setBackendError] = useState(null);

  const addLog = useCallback((message) => {
    setLogs((prevLogs) => [
      `[${new Date().toLocaleTimeString()}] ${message}`,
      ...prevLogs.slice(0, MAX_LOG_ENTRIES - 1),
    ]);
  }, []);

  const getTabMode = useCallback(
    () => (activeTab === TAB_TRAIN ? MODE_TRAINING : MODE_PRODUCTION),
    [activeTab]
  );

  const canToggleRecording = useCallback(
    (tabMode) => !recordingMode || recordingMode === tabMode,
    [recordingMode]
  );

  const runProductionAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    addLog('Analysiere Prozessdaten (MediaPipe + ML)...');
    try {
      await axios.post(`${API_URL}/analysis`);
      addLog('✅ ML-Analyse erfolgreich abgeschlossen.');
    } catch {
      addLog('❌ Fehler bei der Datenanalyse.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [addLog]);

  const runTrainingAnalysis = useCallback(() => {
    addLog('🏋️ Verarbeitung im Hintergrund gestartet...');
    void axios
      .post(`${API_URL}/analysis/training`)
      .then(() => {
        addLog('✅ Trainingsdaten werden im Hintergrund verarbeitet');
      })
      .catch(() => {
        addLog('❌ Server-Fehler beim Starten der Verarbeitung.');
      });
  }, [addLog]);

  // ===== INITIALES LADEN DER CONFIG & BACKEND-CHECK =====
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await axios.get(`${API_URL}/config`, { timeout: 5000 });
        setConfig(response.data);
        if (response.data.enabled_models) {
          setEnabledModels(response.data.enabled_models);
        }
        setIsBackendReady(true);
        setBackendError(null);
      } catch (error) {
        console.error('Backend nicht erreichbar:', error.message);
        setBackendError(
          error.code === 'ECONNABORTED'
            ? 'Backend antwortet nicht (Timeout)'
            : 'Backend nicht erreichbar'
        );
        // Retry nach 2 Sekunden
        setTimeout(checkBackend, 2000);
      }
    };

    checkBackend();
  }, []);

  // ===== CLEANUP BEIM TAB-SCHLIESSEN =====
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Synchrone Anfrage beim Unload (keepalive verhindert Timeout)
      if (navigator.sendBeacon) {
        navigator.sendBeacon(`${API_URL}/cleanup`, JSON.stringify({ mode: 'production' }));
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // -------------------------------------------------------------------------
  // HELPER FUNKTIONEN
  // -------------------------------------------------------------------------

  /**
   * Behandelt Start/Stop der Videoaufnahme.
   *
   * RESTful API:
   * - Start: POST /api/recordings mit { "mode": "production" | "train" }
   * - Stop: PATCH /api/recordings/current
   *
   * - Production-Modus: Blockiert Frontend während ML-Analyse
   * - Training-Modus: Startet Hintergrund-Verarbeitung ohne Blockierung
   */
  const handleRecord = async () => {
    const currentTabMode = getTabMode();

    if (!canToggleRecording(currentTabMode)) {
      addLog(`Beenden nur im ${recordingMode}-Tab möglich!`);
      return;
    }

    const isStarting = !recordingMode;

    try {
      if (isStarting) {
        // RESTful: POST /api/recordings
        await axios.post(`${API_URL}/recordings`, { mode: currentTabMode });
        setRecordingMode(currentTabMode);
        addLog(`${currentTabMode.toUpperCase()}: Start`);
      } else {
        // RESTful: PATCH /api/recordings/current
        await axios.patch(`${API_URL}/recordings/current`);
        setRecordingMode(null);
        addLog(`${currentTabMode.toUpperCase()}: Stop`);

        // Starte Analyse je nach Modus
        if (currentTabMode === MODE_PRODUCTION) {
          await runProductionAnalysis();
        } else if (currentTabMode === MODE_TRAINING) {
          runTrainingAnalysis();
        }
      }
    } catch {
      addLog('Kamera-Fehler bei Aufnahme.');
    }
  };

  /**
   * Rendert den Inhalt des aktuell aktiven Tabs.
   */
  const renderTabContent = () => {
    switch (activeTab) {
      case TAB_DASHBOARD:
        return <DashboardTab addLog={addLog} enabledModels={enabledModels} config={config} />;
      case TAB_CONFIG:
        return <ConfigTab addLog={addLog} config={config} setConfig={setConfig} enabledModels={enabledModels} />;
      case TAB_LIVE:
      case TAB_TRAIN:
        return <VideoTab activeTab={activeTab} recordingMode={recordingMode} />;
      default:
        return null;
    }
  };

  // -------------------------------------------------------------------------
  // RENDER
  // -------------------------------------------------------------------------

  // Loading/Error Screen wenn Backend nicht bereit
  if (!isBackendReady) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[linear-gradient(87deg,#5e72e4,#825ee4)]">
        <div className="text-center">
          <div className="mb-6 inline-block h-16 w-16 animate-spin rounded-full border-4 border-solid border-white border-t-transparent"></div>
          <h1 className="mb-2 text-2xl font-bold text-white">
            {backendError ? '⚠️ Backend-Verbindungsfehler' : 'Initialisiere Backend...'}
          </h1>
          <p className="text-lg text-white/80">
            {backendError || 'Bitte warten, Backend wird gestartet...'}
          </p>
          {backendError && (
            <p className="mt-4 text-sm text-white/60">
              Stelle sicher, dass der Backend-Server läuft (Port 8000)
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-white font-sans text-[#525f7f] antialiased">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} logs={logs} />

      <main
        className={`flex h-full flex-1 flex-col bg-[linear-gradient(87deg,#5e72e4,#825ee4)] ${
          activeTab === TAB_DASHBOARD || activeTab === TAB_CONFIG
            ? 'overflow-y-auto'
            : 'overflow-hidden'
        }`}
      >
        <Header
          activeTab={activeTab}
          isAnalyzing={isAnalyzing}
          recordingMode={recordingMode}
          handleRecord={handleRecord}
        />

        <div className="flex min-h-0 flex-1 flex-col px-8 pb-8">
          {renderTabContent()}
        </div>
      </main>
    </div>
  );
}
