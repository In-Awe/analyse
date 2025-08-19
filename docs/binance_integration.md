# Binance API Integration Blueprint

## WebSocket Streams
- Kline: `wss://stream.binance.com:9443/ws/btcusdt@kline_1m`
- Book Ticker: `btcusdt@bookTicker`
- Trades: `btcusdt@aggTrade`

## REST Endpoints
- Place Order: `POST /api/v3/order`
- Cancel Order: `DELETE /api/v3/order`
- Account Info: `GET /api/v3/account`

## Rate Limits
- Request Weight: 1200/minute
- Orders: 10/second, 200k/day

## Error Handling
- 429: Rate limit exceeded
- 418: IP ban
