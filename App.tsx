
import React, { useState, useCallback, useMemo } from 'react';
import { OHLCV, ChartData, IndicatorSettings, ChatMessage, MessageSender } from './types';
import ControlPanel from './components/ControlPanel';
import PriceChart from './components/PriceChart';
import ChatPanel from './components/ChatPanel';
import { calculateSMA, calculateRSI } from './services/indicatorService';
import { analyzeChartData } from './services/geminiService';

const App: React.FC = () => {
  const [rawData, setRawData] = useState<OHLCV[]>([]);
  const [indicatorSettings, setIndicatorSettings] = useState<IndicatorSettings>({
    sma: { enabled: false, period: 20 },
    rsi: { enabled: false, period: 14 },
  });
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDataLoaded = useCallback((data: OHLCV[]) => {
    setRawData(data);
    setError(null);
    setChatHistory([{
        sender: MessageSender.GEMINI,
        text: `Successfully loaded ${data.length} data points. What would you like to analyze?`
    }]);
  }, []);

  const chartData = useMemo<ChartData[]>(() => {
    if (rawData.length === 0) return [];
    
    let data: ChartData[] = [...rawData];
    const closePrices = rawData.map(d => d.close);

    if (indicatorSettings.sma.enabled) {
      const smaValues = calculateSMA(closePrices, indicatorSettings.sma.period);
      data = data.map((d, i) => ({ ...d, sma: smaValues[i] }));
    }
    if (indicatorSettings.rsi.enabled) {
      const rsiValues = calculateRSI(closePrices, indicatorSettings.rsi.period);
      data = data.map((d, i) => ({ ...d, rsi: rsiValues[i] }));
    }
    return data;
  }, [rawData, indicatorSettings]);
  
  const handleSendMessage = useCallback(async (message: string) => {
    if (!message.trim() || isAnalyzing) return;
    
    setChatHistory(prev => [...prev, { sender: MessageSender.USER, text: message }]);
    setIsAnalyzing(true);
    setError(null);

    try {
      const geminiResponse = await analyzeChartData(message, chartData, indicatorSettings);
      setChatHistory(prev => [...prev, { sender: MessageSender.GEMINI, text: geminiResponse }]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred.";
      setError(`Failed to get analysis: ${errorMessage}`);
      setChatHistory(prev => [...prev, { sender: MessageSender.GEMINI, text: `Sorry, I encountered an error. ${errorMessage}` }]);
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, chartData, indicatorSettings]);


  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col p-4 font-sans">
      <header className="w-full mb-4">
        <h1 className="text-3xl font-bold text-cyan-400 text-center">Crypto LLM Trading Analyst</h1>
      </header>
      
      <main className="flex-grow grid grid-cols-1 lg:grid-cols-12 gap-4 h-[calc(100vh-80px)]">
        <div className="lg:col-span-3 bg-gray-800 rounded-lg p-4 overflow-y-auto">
          <ControlPanel 
            onDataLoaded={handleDataLoaded} 
            indicatorSettings={indicatorSettings}
            onIndicatorChange={setIndicatorSettings}
            dataLoaded={rawData.length > 0}
            onError={setError}
          />
        </div>
        
        <div className="lg:col-span-9 grid grid-rows-3 gap-4 h-full">
            <div className="row-span-2 bg-gray-800 rounded-lg p-4 flex flex-col">
              {rawData.length > 0 ? (
                <PriceChart data={chartData} indicators={indicatorSettings} />
              ) : (
                <div className="flex-grow flex items-center justify-center text-gray-500">
                    <p>Upload a CSV file to begin analysis.</p>
                </div>
              )}
            </div>
            <div className="row-span-1 bg-gray-800 rounded-lg flex flex-col">
              <ChatPanel 
                history={chatHistory} 
                onSendMessage={handleSendMessage} 
                isLoading={isAnalyzing} 
              />
            </div>
        </div>
      </main>
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white p-3 rounded-lg shadow-lg">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default App;
