import React from 'react';
import { ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { ChartData, IndicatorSettings } from '../types';

interface PriceChartProps {
  data: ChartData[];
  indicators: IndicatorSettings;
}

const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-gray-700 p-3 border border-gray-600 rounded-lg shadow-lg">
        <p className="label font-bold text-cyan-400">{`${new Date(data.timestamp).toLocaleString()}`}</p>
        <p className="text-sm">Open: <span className="font-mono">{data.open.toFixed(2)}</span></p>
        <p className="text-sm">High: <span className="font-mono">{data.high.toFixed(2)}</span></p>
        <p className="text-sm">Low: <span className="font-mono">{data.low.toFixed(2)}</span></p>
        <p className="text-sm">Close: <span className="font-mono text-green-400">{data.close.toFixed(2)}</span></p>
        <p className="text-sm">Volume: <span className="font-mono">{data.volume.toLocaleString()}</span></p>
        {data.sma && <p className="text-sm text-yellow-400">SMA: <span className="font-mono">{data.sma.toFixed(2)}</span></p>}
        {data.rsi && <p className="text-sm text-purple-400">RSI: <span className="font-mono">{data.rsi.toFixed(2)}</span></p>}
      </div>
    );
  }
  return null;
};

const PriceChart: React.FC<PriceChartProps> = ({ data, indicators }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
        <defs>
            <linearGradient id="colorVolume" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.1}/>
            </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#4a4a4a" />
        <XAxis 
            dataKey="timestamp" 
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(unixTime) => new Date(unixTime).toLocaleDateString()}
            tick={{ fill: '#a0a0a0' }} 
            stroke="#a0a0a0"
        />
        <YAxis yAxisId="left" orientation="left" stroke="#a0a0a0" tick={{ fill: '#a0a0a0' }} />
        <YAxis yAxisId="right" orientation="right" stroke="#a0a0a0" domain={[0, 100]} tick={{ fill: '#a0a0a0' }} hide={!indicators.rsi.enabled} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ color: '#a0a0a0' }} />

        <Line 
          yAxisId="left" 
          type="monotone" 
          dataKey="close" 
          stroke="#10b981" 
          strokeWidth={2} 
          dot={false}
          name="Close Price"
        />
        {indicators.sma.enabled && (
          <Line 
            yAxisId="left" 
            type="monotone" 
            dataKey="sma" 
            stroke="#facc15" 
            strokeWidth={1.5}
            dot={false}
            name={`SMA(${indicators.sma.period})`}
          />
        )}
        {indicators.rsi.enabled && (
          <Line 
            yAxisId="right" 
            type="monotone" 
            dataKey="rsi" 
            stroke="#a855f7" 
            strokeWidth={1.5}
            dot={false}
            name={`RSI(${indicators.rsi.period})`}
          />
        )}
         <Bar yAxisId="left" dataKey="volume" fill="url(#colorVolume)" barSize={10} name="Volume" />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default PriceChart;