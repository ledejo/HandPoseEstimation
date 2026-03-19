import { useMemo, useState } from 'react';
import { Radio } from 'lucide-react';

import { API_URL } from '../../api/config';

// -------------------------------------------------------------------------
// VIDEO TAB KOMPONENTE
// -------------------------------------------------------------------------

/**
 * Rendert den Video-Tab mit MJPEG-Stream und REC-Indikator.
 *
 * @param {Object} props - Komponenten-Parameter.
 * @param {string} props.activeTab - Aktiver Tab (`live` oder `train`).
 * @param {string|null} props.recordingMode - Aktiver Aufnahmemodus.
 * @returns {JSX.Element} Video-Ansicht.
 */
export default function VideoTab({ activeTab, recordingMode }) {
  const activeVideoMode = activeTab === 'train' ? 'train' : 'production';
  const isActiveModeRecording = recordingMode === activeVideoMode;
  const [showRoiOverlay, setShowRoiOverlay] = useState(false);
  const videoSrc = useMemo(
    () => `${API_URL}/video/${activeVideoMode}?show_rois=${showRoiOverlay ? 'true' : 'false'}`,
    [activeVideoMode, showRoiOverlay]
  );

  return (
    <div className="relative mx-auto w-full max-w-5xl overflow-hidden rounded-xl bg-white shadow-2xl mb-8 flex-1">
      <div className="absolute inset-0 bg-black">
        <img
          src={videoSrc}
          className="h-full w-full object-contain opacity-100 transition-opacity duration-700"
          alt="Live feed"
        />

        <button
          type="button"
          onClick={() => setShowRoiOverlay((prev) => !prev)}
          className="absolute left-6 top-6 rounded-full border border-white/20 bg-black/50 px-4 py-2 text-[10px] font-black uppercase tracking-widest text-white backdrop-blur-md"
        >
          {showRoiOverlay ? 'ROI Overlay: AN' : 'ROI Overlay: AUS'}
        </button>

        {isActiveModeRecording && (
          <div className="absolute right-6 top-6 flex items-center gap-2 rounded-full border border-white/20 bg-black/40 px-4 py-2 backdrop-blur-md">
            <div className="h-3 w-3 animate-pulse rounded-full bg-red-600" />
            <span className="text-[10px] font-black uppercase tracking-widest text-white">
              REC
            </span>
          </div>
        )}

        {!recordingMode && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-4 text-white/50">
            <Radio size={64} className="animate-pulse opacity-20" />
            <p className="text-xl font-black uppercase tracking-widest">Standby</p>
          </div>
        )}
      </div>
    </div>
  );
}
