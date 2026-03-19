// -------------------------------------------------------------------------
// API KONFIGURATION
// -------------------------------------------------------------------------

/**
 * Basis-URL für alle API-Anfragen an das Backend.
 * @constant {string}
 */
export const API_URL = 'http://localhost:8000/api';

/**
 * Standard-Konfigurationswerte für Prozess-Parameter.
 * @constant {Object}
 */
export const DEFAULT_CONFIG = {
  hmm_states: 5,
  dbscan_clusters: 3,
  cluster_names: '',
  camera_type: 1,
};

// -------------------------------------------------------------------------
// KONSTANTEN FÜR UI
// -------------------------------------------------------------------------

/**
 * Maximale Anzahl von Log-Einträgen, die in der Sidebar angezeigt werden.
 * @constant {number}
 */
export const MAX_LOG_ENTRIES = 20;

/**
 * Timeout für API-Anfragen in Millisekunden.
 * @constant {number}
 */
export const API_TIMEOUT = 30000;
