import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import App from './App';

vi.mock('axios');

describe('Hand Pose Estimation - App Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    axios.get.mockResolvedValue({ data: {} });
    axios.post.mockResolvedValue({ status: 200 });
    axios.patch.mockResolvedValue({ status: 200 });
  });

  const renderReadyApp = async () => {
    render(<App />);
    await screen.findByText('AI Quality Assistant');
  };

  it('rendert die Sidebar und das initiale Live-Video-Tab', async () => {
    await renderReadyApp();

    expect(screen.getByText('AI Quality Assistant')).toBeInTheDocument();
    expect(screen.getByText(/Bereit/i)).toBeInTheDocument();
  });

  it('wechselt zum Dashboard-Tab, wenn in der Sidebar geklickt wird', async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes('/config')) {
        return Promise.resolve({
          data: { cluster_names: 'Box A, Box B, Box C' },
        });
      }
      if (url.includes('/results')) {
        return Promise.resolve({ data: ['run1', 'run2'] });
      }
      return Promise.resolve({ data: {} });
    });

    await renderReadyApp();

    const dashboardButton = await screen.findByRole('button', { name: /dashboard/i });
    fireEvent.click(dashboardButton);

    await waitFor(() => {
      expect(screen.getByText('⏱️ Gesamtdauer')).toBeInTheDocument();
      expect(screen.getByText('⚠️ Anomalien')).toBeInTheDocument();
    });
  });

  it('startet den Production-Recording-Workflow', async () => {
    axios.post.mockResolvedValue({ status: 200 });

    await renderReadyApp();

    const startButton = await screen.findByRole('button', { name: /start/i });
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith('http://localhost:8000/api/recordings', {
        mode: 'production',
      });
    });

    expect(screen.getByText(/PRODUCTION: Start/i)).toBeInTheDocument();
  });

  it('wechselt zum Config-Tab und lädt die initiale Konfiguration', async () => {
    const mockConfig = {
      dbscan_clusters: 3,
      cluster_names: 'Feder, Kappe, Mine',
    };
    axios.get.mockImplementation((url) => {
      if (url.includes('/config')) {
        return Promise.resolve({ data: mockConfig });
      }
      return Promise.resolve({ data: [] });
    });

    await renderReadyApp();

    const configButton = await screen.findByRole('button', { name: /einstellungen/i });
    fireEvent.click(configButton);

    await waitFor(() => {
      expect(screen.getByText('Prozess-Konfiguration')).toBeInTheDocument();
      expect(axios.get).toHaveBeenCalledWith(
        'http://localhost:8000/api/config',
        expect.objectContaining({ timeout: 5000 })
      );
    });
  });
});
