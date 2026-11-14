# Polymarket FAA outage attribution forecaster

Forecasts the Yes price of every trading market of Polymarket's
FAA outage attribution family (Iran, foreign actor, hack, AI sandbox
escape), minute by minute, 240 minutes ahead, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95). Generic over the markets in the case
file: the practice cases hold NVIDIA-event markets, the final holds
the four FAA markets; the program treats both the same way.

## Method

- **Median = persistence**: the last price at the checkpoint.
- **Adaptive sigma per market**: the winsorised mean absolute
  1-minute move over the last 360 minutes (cap 0.05), divided by
  E|N(0,1)| = 0.7979, floored at 0.0003 and capped at 0.02 per
  sqrt(minute). Quiet markets get a narrow spread; volatile ones
  get a wide one.
- **Quantiles**: standard normal z-scores (Acklam) times
  sigma * sqrt(horizon minute).

Python 3.13, standard library only. Reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Practice backtest (6 cases, mean pinball loss)

| | model | house baseline |
|---|---|---|
| overall | 0.2787 | 0.3636 |

-23.3% vs the house baseline on all six practice cases
(practice is NVIDIA-event data; the final is FAA data, so this
is no guide to the final score).
