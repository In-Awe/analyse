
export const calculateSMA = (data: number[], period: number): (number | undefined)[] => {
  if (period <= 0 || data.length < period) {
    return Array(data.length).fill(undefined);
  }

  const result: (number | undefined)[] = Array(period - 1).fill(undefined);
  let sum = data.slice(0, period).reduce((a, b) => a + b, 0);
  result.push(sum / period);

  for (let i = period; i < data.length; i++) {
    sum += data[i] - data[i - period];
    result.push(sum / period);
  }
  return result;
};

export const calculateRSI = (data: number[], period: number): (number | undefined)[] => {
  if (period <= 0 || data.length <= period) {
    return Array(data.length).fill(undefined);
  }
  
  const result: (number | undefined)[] = Array(period).fill(undefined);
  let gains = 0;
  let losses = 0;

  for (let i = 1; i <= period; i++) {
    const change = data[i] - data[i - 1];
    if (change > 0) {
      gains += change;
    } else {
      losses -= change;
    }
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  const calculateRSIValue = (gain: number, loss: number) => {
    if (loss === 0) return 100;
    const rs = gain / loss;
    return 100 - (100 / (1 + rs));
  };
  
  result.push(calculateRSIValue(avgGain, avgLoss));

  for (let i = period + 1; i < data.length; i++) {
    const change = data[i] - data[i - 1];
    let currentGain = 0;
    let currentLoss = 0;

    if (change > 0) {
      currentGain = change;
    } else {
      currentLoss = -change;
    }

    avgGain = (avgGain * (period - 1) + currentGain) / period;
    avgLoss = (avgLoss * (period - 1) + currentLoss) / period;
    
    result.push(calculateRSIValue(avgGain, avgLoss));
  }
  
  return result;
};
