import { Play, Square, RefreshCw } from 'lucide-react';

// -------------------------------------------------------------------------
// HEADER KOMPONENTE
// -------------------------------------------------------------------------

/**
 * Rendert den Seitenkopf mit Tab-Titel und Aufnahme-Button.
 *
 * @param {Object} props - Komponenten-Parameter.
 * @param {string} props.activeTab - Aktiver Tab (`live`, `train`, `dashboard`, `config`).
 * @param {boolean} props.isAnalyzing - Kennzeichnet laufende Analyse.
 * @param {string|null} props.recordingMode - Aktueller Aufnahmemodus.
 * @param {() => Promise<void>} props.handleRecord - Handler für Start/Stop.
 * @returns {JSX.Element} Header-Element.
 */
export default function Header({ activeTab, isAnalyzing, recordingMode, handleRecord }) {
  const isLiveOrTrainTab = activeTab === 'live' || activeTab === 'train';
  const activeVideoMode = activeTab === 'train' ? 'train' : 'production';
  const isActiveModeRecording = recordingMode && recordingMode === activeVideoMode;
  const isButtonDisabled = (recordingMode && recordingMode !== activeVideoMode) || isAnalyzing;

  return (
    <header className="flex shrink-0 items-center justify-between p-8 text-white">
      <h2 className="text-2xl font-bold uppercase tracking-tight">{activeTab}</h2>

      {isLiveOrTrainTab && (
        <button
          onClick={handleRecord}
          disabled={isButtonDisabled}
          className={`rounded px-8 py-2 text-xs font-bold uppercase shadow-lg transition-all disabled:opacity-30 flex items-center gap-2 ${
            isAnalyzing
              ? 'bg-orange-400 text-white'
              : recordingMode
              ? 'bg-white text-red-500'
              : 'bg-white text-[#5e72e4]'
          }`}
        >
          {isAnalyzing ? (
            <>
              <RefreshCw className="animate-spin" size={14} /> Analysiere...
            </>
          ) : isActiveModeRecording ? (
            <>
              <Square size={14} /> Stop
            </>
          ) : (
            <>
              <Play size={14} /> Start
            </>
          )}
        </button>
      )}
    </header>
  );
}
