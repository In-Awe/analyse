# Binance Integration Blueprint

This document provides the technical blueprint for safe and robust integration with Binance (Spot) API.

## Streams and endpoints
- WebSocket:
`wss://stream.binance.com:9443/ws/<symbol>@kline_1m` (primary 1-minute candles)
`wss://stream.binance.com:9443/ws/<symbol>@aggTrade` (optional tick-level)
- REST Endpoints (signed where required):
`POST /api/v3/order` — create order
`DELETE /api/v3/order` — cancel
`GET /api/v3/openOrders` — open orders

## Authentication & Security
- Use HMAC SHA256 signing for REST endpoints. Keep API keys out of the repo — use env vars or Vault.
- Set recvWindow to a conservative value (e.g., 5000 ms) for critical calls.

## Order execution strategy
- Primary: LIMIT orders (GTC) to attempt maker fee capture.
- Fallback: MARKET orders when immediate execution required.
- Cancellation policy: cancel unfilled limit orders after a configurable timeout (e.g., 60s) and optionally retry as IOC or MARKET depending on risk.

## Rate limit handling
- Track REQUEST_WEIGHT and ORDERS limits with a token-bucket.
- On 429: exponential backoff + metric emission.
- On 418: immediate halt to REST requests, escalate alert and stop trading until manual review.

## Slippage & execution price model
- For backtests: assume execution at next bar open ± alpha * ATR_next_bar (alpha configurable, default 0.5).
- For live: use a combination of limit posting and liquidity-aware sizing to minimize market impact.

## Recommended libraries
- python-binance for REST/WebSocket helpers (audit before production)
- aiohttp or websockets for lightweight WebSocket clients

## Testing & staging
- Provide a dry-run mode that emits orders to logs only.
- Staging account with sandbox keys for live integration tests.

## Message formats

Order request example:

```json
{
"type": "order_request",
"side": "BUY",
"order_type": "LIMIT",
"price": 50010.0,
"quantity": 0.001
}
```

Order status handling: reconcile fills using User Data Stream and record trade events to persistent ledger.

--
End of Binance blueprint
