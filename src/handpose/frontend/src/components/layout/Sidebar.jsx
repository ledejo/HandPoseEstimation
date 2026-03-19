import { Tv, LayoutDashboard, Video, Settings } from 'lucide-react';

// -------------------------------------------------------------------------
// NAVIGATION KONFIGURATION
// -------------------------------------------------------------------------

/**
 * Konfiguration der Navigation-Tabs.
 * Jedes Item definiert ID, Label, Icon-Komponente und Farbe.
 */
const NAV_ITEMS = [
  { id: 'live', label: 'Live Analyse', icon: Tv, color: 'text-cyan-400' },
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, color: 'text-indigo-500' },
  { id: 'train', label: 'Recorder', icon: Video, color: 'text-red-500' },
  { id: 'config', label: 'Einstellungen', icon: Settings, color: 'text-green-500' },
];

// -------------------------------------------------------------------------
// SIDEBAR KOMPONENTE
// -------------------------------------------------------------------------

/**
 * Rendert die Sidebar mit Navigation und Log-Liste.
 *
 * @param {Object} props - Komponenten-Parameter.
 * @param {string} props.activeTab - Aktiver Tab.
 * @param {(tabId: string) => void} props.setActiveTab - Setter für Tab-Wechsel.
 * @param {string[]} props.logs - Liste der Log-Einträge.
 * @returns {JSX.Element} Sidebar-Element.
 */
export default function Sidebar({ activeTab, setActiveTab, logs }) {
  return (
    <aside className="z-50 flex w-[260px] flex-shrink-0 flex-col border-r bg-white shadow-xl">
      <div className="flex h-20 items-center justify-center gap-2 border-b font-bold text-[#32325d]">
        <div className="text-lg font-bold">
          AI Quality Assistant
        </div>
      </div>

      <nav className="flex-1 space-y-1 p-4">
        {NAV_ITEMS.map((tabConfig) => (
          <button
            key={tabConfig.id}
            onClick={() => setActiveTab(tabConfig.id)}
            className={`w-full rounded-lg px-4 py-3 text-sm font-bold flex items-center gap-4 transition-colors ${
              activeTab === tabConfig.id
                ? 'bg-[#f6f9fc] text-[#5e72e4]'
                : 'hover:bg-gray-50'
            }`}
          >
            <tabConfig.icon
              size={18}
              className={activeTab === tabConfig.id ? 'text-[#5e72e4]' : tabConfig.color}
            />
            {tabConfig.label}
          </button>
        ))}
      </nav>

      <div className="m-4 h-32 shrink-0 overflow-y-auto rounded-xl border bg-slate-50 p-4 font-mono text-[10px]">
        {logs.map((logEntry) => (
          <div key={logEntry}>{logEntry}</div>
        ))}
      </div>
    </aside>
  );
}
