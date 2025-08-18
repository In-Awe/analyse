
export interface OHLCV {
  timestamp: number;
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ChartData extends OHLCV {
  sma?: number;
  rsi?: number;
}

export enum MessageSender {
  USER = 'user',
  GEMINI = 'gemini',
}

export interface ChatMessage {
  sender: MessageSender;
  text: string;
}

export interface IndicatorSettings {
  sma: {
    enabled: boolean;
    period: number;
  };
  rsi: {
    enabled: boolean;
    period: number;
  };
}
