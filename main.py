"""FAA outage attribution forecaster (Polymarket, make.fun program/v1).

Reads one case file per run (every *.in in $MAKEFUN_INPUT_DIR, else
/input) and writes one forecast file per case to $MAKEFUN_OUTPUT_DIR
(else /output): for each market, 240 rows x 5 quantiles
[0.05, 0.25, 0.5, 0.75, 0.95] of the Yes price, minute by minute.

Method (generic over the markets in the case file -- the practice
cases hold NVIDIA-event markets, the final holds the four FAA
attribution markets; both are treated the same way):
  * median = persistence: the last price at the checkpoint;
  * spread = adaptive sigma per market: the winsorised mean absolute
    1-minute move over the last 360 minutes, divided by E|N(0,1)|,
    floored at 0.0003 and capped at 0.02 per sqrt(minute);
  * quantiles = standard-normal z-scores (Acklam's inverse CDF)
    times sigma * sqrt(minute after the checkpoint), clamped to
    [0, 1] and rounded to 6 decimals.
Python 3.13, standard library only, fully deterministic.
Note: the final's FAA markets are only hours old at the deadline,
so a case may hold far fewer than 4,320 prices per market -- the
sigma window simply uses whatever history the case holds (at least
10 moves, else the 0.0025 fallback).
"""
import glob, json, math, os


def _horner(coeffs, x):
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam), accurate to ~1e-9."""
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return _horner(c, q) / (_horner(d, q) * q + 1.0)
    if p <= 0.97575:
        q = p - 0.5
        r = q * q
        return _horner(a, r) * q / (_horner(b, r) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -_horner(c, q) / (_horner(d, q) * q + 1.0)


def _round6(x):
    v = round(min(1.0, max(0.0, x)) * 1000000.0) / 1000000.0
    return v + 0.0


def _sigma_recent(prices):
    """Per-sqrt-minute move scale from the last 360 one-minute moves."""
    n = len(prices)
    w = 360 if n - 1 > 360 else n - 1
    if w < 10:
        return 0.0025
    total = 0.0
    for i in range(n - w - 1, n - 1):
        d = abs(prices[i + 1] - prices[i])
        total += d if d < 0.05 else 0.05
    v = total / w / 0.7978845608028654
    return 0.0003 if v < 0.0003 else (0.02 if v > 0.02 else v)


def _forecast_market(prices, horizon, zs):
    x = prices[-1] if prices else 0.5
    sg = _sigma_recent(prices) if prices else 0.0025
    rows = []
    for h in range(1, horizon + 1):
        s = sg * math.sqrt(h)
        rows.append([_round6(x + z * s) for z in zs])
    return rows


def main():
    inp = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(inp, "*.in"))):
        cid = os.path.basename(path)[:-3]
        with open(path) as f:
            case = json.load(f)
        horizon = case.get("horizon_minutes", 240)
        taus = case.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])
        zs = [_norm_ppf(t) for t in taus]
        fc = [{"id": m["id"],
               "q": _forecast_market(m["prices"], horizon, zs)}
              for m in case["markets"]]
        with open(os.path.join(out, cid + ".out"), "w") as f:
            f.write(json.dumps({"forecasts": fc}, separators=(",", ":")) + "\n")


main()
