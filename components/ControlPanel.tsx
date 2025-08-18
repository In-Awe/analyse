
import React, { useCallback } from 'react';
import { OHLCV, IndicatorSettings } from '../types';
import { UploadIcon } from './icons';

interface ControlPanelProps {
  onDataLoaded: (data: OHLCV[]) => void;
  indicatorSettings: IndicatorSettings;
  onIndicatorChange: (settings: IndicatorSettings) => void;
  dataLoaded: boolean;
  onError: (message: string) => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  onDataLoaded,
  indicatorSettings,
  onIndicatorChange,
  dataLoaded,
  onError,
}) => {
  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.split('\n').slice(1); // Skip header
        const data: OHLCV[] = lines.map((line) => {
          const [timestamp, open, high, low, close, volume] = line.split(',');
          const ts = parseInt(timestamp, 10);
          return {
            timestamp: ts,
            date: new Date(ts).toLocaleDateString(),
            open: parseFloat(open),
            high: parseFloat(high),
            low: parseFloat(low),
            close: parseFloat(close),
            volume: parseFloat(volume),
          };
        }).filter(d => !isNaN(d.timestamp) && !isNaN(d.close)).sort((a,b) => a.timestamp - b.timestamp);
        
        if (data.length === 0) {
            throw new Error("CSV file is empty or in the wrong format.");
        }
        
        onDataLoaded(data);
      } catch (err) {
        onError(err instanceof Error ? err.message : "Failed to parse CSV file.");
      }
    };
    reader.onerror = () => onError("Error reading file.");
    reader.readAsText(file);
    event.target.value = ''; // Reset input to allow re-uploading the same file
  }, [onDataLoaded, onError]);

  const handleIndicatorToggle = (indicator: keyof IndicatorSettings) => {
    onIndicatorChange({
      ...indicatorSettings,
      [indicator]: {
        ...indicatorSettings[indicator],
        enabled: !indicatorSettings[indicator].enabled,
      },
    });
  };

  const handlePeriodChange = (indicator: keyof IndicatorSettings, value: string) => {
    const period = parseInt(value, 10);
    if (period > 0) {
      onIndicatorChange({
        ...indicatorSettings,
        [indicator]: { ...indicatorSettings[indicator], period },
      });
    }
  };
  
  return (
    <div className="flex flex-col space-y-6 h-full">
      <div>
        <h2 className="text-xl font-semibold text-cyan-400 mb-3">1. Load Data</h2>
        <label
          htmlFor="csv-upload"
          className="w-full flex items-center justify-center px-4 py-3 bg-gray-700 text-gray-300 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors"
        >
          <UploadIcon />
          <span className="ml-2">Upload CSV File</span>
        </label>
        <input
          id="csv-upload"
          type="file"
          accept=".csv"
          className="hidden"
          onChange={handleFileChange}
        />
        <p className="text-xs text-gray-500 mt-2">
          Format: timestamp,open,high,low,close,volume (with header)
        </p>
      </div>

      <div className={`transition-opacity duration-500 ${dataLoaded ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
        <h2 className="text-xl font-semibold text-cyan-400 mb-3">2. Technical Indicators</h2>
        <div className="space-y-4">
          {/* SMA Settings */}
          <div className="bg-gray-700 p-3 rounded-lg">
            <div className="flex items-center justify-between">
              <label htmlFor="sma-toggle" className="font-medium">Simple Moving Average (SMA)</label>
              <input
                type="checkbox"
                id="sma-toggle"
                checked={indicatorSettings.sma.enabled}
                onChange={() => handleIndicatorToggle('sma')}
                className="form-checkbox h-5 w-5 text-cyan-500 bg-gray-800 border-gray-600 rounded focus:ring-cyan-600"
              />
            </div>
            {indicatorSettings.sma.enabled && (
              <div className="mt-2">
                <label htmlFor="sma-period" className="text-sm text-gray-400">Period</label>
                <input
                  type="number"
                  id="sma-period"
                  value={indicatorSettings.sma.period}
                  onChange={(e) => handlePeriodChange('sma', e.target.value)}
                  className="w-full mt-1 bg-gray-800 border border-gray-600 rounded-md px-2 py-1 focus:ring-cyan-500 focus:border-cyan-500"
                />
              </div>
            )}
          </div>

          {/* RSI Settings */}
          <div className="bg-gray-700 p-3 rounded-lg">
            <div className="flex items-center justify-between">
              <label htmlFor="rsi-toggle" className="font-medium">Relative Strength Index (RSI)</label>
              <input
                type="checkbox"
                id="rsi-toggle"
                checked={indicatorSettings.rsi.enabled}
                onChange={() => handleIndicatorToggle('rsi')}
                className="form-checkbox h-5 w-5 text-cyan-500 bg-gray-800 border-gray-600 rounded focus:ring-cyan-600"
              />
            </div>
            {indicatorSettings.rsi.enabled && (
              <div className="mt-2">
                <label htmlFor="rsi-period" className="text-sm text-gray-400">Period</label>
                <input
                  type="number"
                  id="rsi-period"
                  value={indicatorSettings.rsi.period}
                  onChange={(e) => handlePeriodChange('rsi', e.target.value)}
                  className="w-full mt-1 bg-gray-800 border border-gray-600 rounded-md px-2 py-1 focus:ring-cyan-500 focus:border-cyan-500"
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
