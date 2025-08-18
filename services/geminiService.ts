
import { GoogleGenAI } from "@google/genai";
import { ChartData, IndicatorSettings } from '../types';

if (!process.env.API_KEY) {
  throw new Error("API_KEY environment variable is not set.");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const formatDataForPrompt = (data: ChartData[]): string => {
  if (data.length === 0) return "No data available.";
  
  const header = Object.keys(data[0]).join(',');
  const rows = data.map(d => 
      `${d.date},${d.open.toFixed(2)},${d.high.toFixed(2)},${d.low.toFixed(2)},${d.close.toFixed(2)},${d.volume.toFixed(0)},${d.sma?.toFixed(2) ?? 'N/A'},${d.rsi?.toFixed(2) ?? 'N/A'}`
  );

  return `${header}\n${rows.join('\n')}`;
};

export const analyzeChartData = async (
  prompt: string,
  chartData: ChartData[],
  indicators: IndicatorSettings
): Promise<string> => {
  if (chartData.length === 0) {
    return "There is no data loaded to analyze. Please upload a CSV file first.";
  }

  // Use the last 100 data points for context to avoid overly large prompts
  const dataSubset = chartData.slice(-100);
  const dataContext = formatDataForPrompt(dataSubset);

  const systemInstruction = `You are an expert cryptocurrency trading analyst. Your role is to analyze the provided market data and identify patterns, trends, and potential trading signals.
  - The user will provide a query about the data.
  - You will be given a subset of recent OHLCV (Open, High, Low, Close, Volume) data in CSV format.
  - The data may also include columns for technical indicators like SMA (Simple Moving Average) and RSI (Relative Strength Index).
  - Provide a concise, insightful analysis based on the data. Do not just state the obvious.
  - Explain your reasoning clearly. If you see a pattern (e.g., a bullish divergence on RSI), describe it.
  - Your response should be helpful for someone trying to understand the market dynamics.
  - Be objective and data-driven. Do not provide financial advice.
  `;
  
  const fullPrompt = `
Here is the recent market data:
\`\`\`csv
${dataContext}
\`\`\`

Active Indicators:
- SMA(${indicators.sma.period}): ${indicators.sma.enabled ? 'ON' : 'OFF'}
- RSI(${indicators.rsi.period}): ${indicators.rsi.enabled ? 'ON' : 'OFF'}

User Query: "${prompt}"

Your Analysis:
`;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: fullPrompt,
      config: {
        systemInstruction,
        temperature: 0.5,
      }
    });

    return response.text.trim();
  } catch (error) {
    console.error("Gemini API call failed:", error);
    throw new Error("The AI analyst could not be reached. Please check your API key and network connection.");
  }
};
