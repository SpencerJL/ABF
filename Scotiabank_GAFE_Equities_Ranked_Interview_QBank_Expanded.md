# Scotiabank GAFE Equities — Ranked Interview Question Bank

**Prepared for Jia Lu — Chris interview focus**  
**Expanded Markdown edition with follow-up answers**

> Main answers: 60–90 seconds. Follow-ups: 30–60 seconds.  
> Technical structure: **definition → intuition → desk relevance → limitation**  
> Incident structure: **reproduce → classify → isolate → benchmark → communicate → resolve**

---

# Tier 1 — Must nail

## 1. What determines the price of a European call?

**Main answer**

A European call depends mainly on spot, strike, time to maturity, volatility, interest rates and dividends. Higher spot increases value; higher strike reduces it. More time generally adds optionality. Higher volatility increases value because the payoff is convex: upside is retained while downside payoff is floored at zero. Higher rates generally increase call value because the strike is paid later and has a lower present value. Higher dividends generally reduce call value because they lower the expected future stock price.

**Follow-up: Which inputs are directly observed?**  
Spot, curves, dividend data and contractual terms are observed or sourced. Volatility is inferred from option prices as implied volatility.

**Follow-up: Why quote implied volatility?**  
It normalises option prices across strikes, maturities and spot levels, making relative value and surface shape easier to compare.

**Follow-up: What changes for a put?**  
Higher spot lowers put value; higher strike raises it; higher rates generally lower it; higher dividends generally raise it. Volatility still raises the value of a standard long put.

---

## 2. Explain delta, gamma, vega and theta.

**Main answer**

Delta is first-order spot sensitivity and the local hedge ratio. Gamma is the change in delta as spot moves and captures curvature. Vega is sensitivity to implied volatility. Theta is sensitivity to time passing. A long vanilla option is generally positive gamma and vega but negative theta.

**Follow-up: Why is gamma high near the strike?**  
Near the strike, a small spot move changes the probability of finishing in the money most rapidly, so delta changes fastest.

**Follow-up: Why is delta hedging imperfect?**  
It is local and model-based. Spot can jump, hedging is discrete, volatility changes, liquidity is imperfect and transaction costs exist.

**Follow-up: Why not maximise gamma?**  
Positive gamma usually costs negative theta, rehedging costs and vega exposure.

---

## 3. What is implied volatility, and why do traders quote it?

**Main answer**

Implied volatility is the volatility input that makes a pricing model reproduce the observed option price. It is backed out from the market rather than directly measured from historical returns. Traders quote it because it normalises prices across strikes, maturities, spot, rates and dividends.

**Follow-up: Historical versus implied volatility?**  
Historical volatility measures realised past movement. Implied volatility is extracted from today’s market price and includes expectations, risk premia and supply-demand effects.

**Follow-up: Can implied volatility be negative?**  
No. Standard volatility parameters are non-negative. Failure to find a positive implied volatility usually indicates inconsistent inputs or price bounds.

---

## 4. Why is equity volatility skew usually downward?

**Main answer**

Lower-strike puts normally trade at higher implied volatility because investors demand crash protection, equity declines often coincide with rising volatility, downside jumps and fat tails are not captured by constant-volatility lognormal models, and dealer hedging costs reinforce the imbalance.

**Follow-up: What happens when skew steepens?**  
Downside puts become relatively more expensive. Products sensitive to downside tails and barriers can reprice materially.

**Follow-up: Which products are most sensitive?**  
Barriers, digitals, reverse convertibles, worst-of structures and autocallables.

---

## 5. A trader says the price is wrong. How do you investigate?

**Main answer**

First reproduce the trade using the same valuation timestamp. Then separate the issue into trade terms, market data, model, numerical engine and system state. Check underlying, strike, maturity, payoff, fixings, spot, curves, dividends, volatility surface, model parameters, numerical settings, code version and caches. Compare with an independent benchmark, isolate the driver, quantify the impact and communicate any temporary workaround.

**Follow-up: What if spot barely moved?**  
Run a factor-by-factor P&L explain: volatility level and skew, rates, dividends, time decay, correlation, barrier or observation-date effects, model recalibration and code/data changes.

**Follow-up: What if only one trader is affected?**  
Compare that trader’s trade capture, snapshot, configuration, cache, permissions and release with an unaffected user. A single-user problem often indicates a local override, stale cache or configuration issue.

**Follow-up: How do you communicate?**  
State what is affected, approximate impact, whether the price is usable, the current hypothesis, workaround and next update.

---

## 6. Design a Python pricing library.

**Main answer**

I would separate the system into:

```text
trade request / API
        ↓
validation and normalisation
        ↓
product
        ↓
immutable market snapshot
        ↓
model
        ↓
pricing engine
        ↓
risk and scenarios
        ↓
controls, logging and response
```

A package structure could be:

```text
products/
market_data/
models/
pricing_engines/
calibration/
risk/
scenarios/
api/
tests/
monitoring/
```

- **Product:** contractual terms and payoff logic.
- **Market snapshot:** spot, curves, dividends, volatility surfaces, correlation, fixings, corporate actions and timestamp/version metadata.
- **Model:** market dynamics and calibrated parameters.
- **Pricing engine:** analytical, tree, PDE or Monte Carlo implementation.
- **Risk layer:** Greeks, scenarios, stresses and P&L explain.
- **API/controls:** input validation, entitlements, diagnostics, logging and structured errors.
- **Tests/monitoring:** unit, regression, benchmark, performance and production checks.

The separation improves reuse, testing, model comparison, debugging, auditability and controlled deployment.

**Follow-up: Why not put the model inside the product?**  
The contract and pricing assumptions are different concepts. The same product may be priced using Black–Scholes, local volatility, stochastic volatility, PDE or Monte Carlo. Embedding one model creates tight coupling and makes comparison and governance harder.

**Follow-up: What belongs in a market snapshot?**  
All inputs required to reproduce valuation at a specific time: spot, FX, discount and funding curves, dividend curves, volatility surfaces, correlations, fixings, corporate actions, approved model parameters, data sources and version metadata.

**Follow-up: Why immutable?**  
It prevents inputs changing mid-calculation, supports safe caching and makes valuations reproducible and auditable.

**Follow-up: Where should calibration live?**  
In a separate calibration module or service. It consumes market quotes and produces model parameters plus quality diagnostics. Pricing should not silently recalibrate independently for each trade.

**Follow-up: How support several models for one product?**  
Use common interfaces and composition. The product defines the contract, the model defines dynamics, and the engine defines the numerical method. Configuration selects the approved combination.

**Follow-up: How avoid a huge inheritance hierarchy?**  
Prefer composition and small interfaces. For example, a product can contain a barrier specification and payoff component rather than inheriting through many specialised subclasses.

**Follow-up: How handle errors?**  
Use structured categories: invalid trade, missing data, calibration failure, numerical non-convergence and internal error. Never silently return a price after an unapproved fallback.

**Follow-up: How improve performance?**  
Profile first, then consider snapshot reuse, batching, vectorisation, caching, avoiding repeated calibration, efficient NumPy kernels and computing only requested outputs.

**Strong 90-second version**

> I would separate product, market data, model and numerical engine. The product stores contractual terms and payoff logic. A timestamped immutable market snapshot contains spot, curves, dividends, volatility surfaces, correlations and fixings. The model defines the dynamics and calibrated parameters, while the pricing engine applies an analytical, PDE or Monte Carlo method. A separate risk layer computes Greeks and scenarios, and the API handles validation, logging and diagnostics. This lets the same product be priced under different models, improves testing and makes production issues easier to isolate. I would also separate calibration, add regression tests against approved benchmarks, and monitor stale data, calibration quality and latency.

---

## 7. How would you validate a newly implemented model?

**Main answer**

Validate mathematical correctness, pricing correctness, risk correctness, production behaviour and business suitability. Use analytical benchmarks, an independent implementation, limiting cases, monotonicity, convergence, Greek comparisons, calibration-quality checks, stress tests, regression tests, performance tests and monitoring.

**Follow-up: What limiting cases?**  
Maturity or volatility approaching zero, barriers far away, grid refinement, more Monte Carlo paths and American value being at least European value.

**Follow-up: What if prices match but Greeks differ?**  
Check bump size, shock convention, recalibration versus frozen parameters, interpolation, smoothing, numerical noise, units and market-data conventions.

---

## 8. Your Python pricing process is five times slower after a release. Where do you start?

**Main answer**

Measure before optimising. Reproduce with representative data, profile end to end, separate CPU, memory, I/O and network time, identify the dominant bottleneck, fix the largest cause, benchmark the improvement and confirm outputs are unchanged.

**Follow-up: What if 90% is market-data calls?**  
Batch and deduplicate requests, load one immutable snapshot, pass data into pricing rather than fetching inside each trade, and use controlled caching.

**Follow-up: When use NumPy instead of pandas?**  
Pandas for labelled tables, joins and reporting; NumPy for dense numerical kernels and vectorised pricing.

---

## 9. How autonomous can you be in London?

**Main answer**

Autonomy means clarifying the objective, reproducing the issue, structuring the investigation, gathering evidence, resolving familiar problems, communicating progress and escalating specialist issues with a precise diagnosis.

**Follow-up: When do you escalate?**  
When impact is material, output may be unreliable, use exceeds approved scope, multiple users are affected, controls are involved or specialist expertise is required.

**Follow-up: What if Canada and New York are offline?**  
Work within validated procedures, use approved fallbacks only, communicate limitations and avoid unsupported certainty.

---

## 10. Why Monte Carlo versus PDE versus Black–Scholes?

**Main answer**

Black–Scholes is a fast analytical benchmark for European vanillas. PDE is attractive for low-dimensional products, barriers and early exercise, with stable Greeks. Monte Carlo scales better to multi-asset and path-dependent products such as baskets and autocallables.

**Follow-up: Why is early exercise harder in Monte Carlo?**  
The exercise decision depends on conditional continuation values, which standard forward simulation does not directly provide.

**Follow-up: Why does PDE suffer in high dimensions?**  
Each extra state variable adds a grid dimension, causing computation and memory to grow rapidly.

---

# Tier 2 — Likely

## 11. Explain a butterfly spread.

A long call butterfly is long one low-strike call, short two middle-strike calls and long one high-strike call. It has limited gain and loss, with maximum payoff near the middle strike.

**Who buys it?** Someone expecting spot to finish near the middle strike.  
**Why not sell a straddle?** The butterfly has long wings that cap tail losses.

---

## 12. Explain a barrier option and why it is hard to hedge.

A barrier option activates or terminates when spot crosses a level. It is path dependent. Hedging is difficult because delta and gamma can change sharply near the barrier, jumps can cross it, monitoring rules matter and skew/model dynamics are important.

**Knock-in versus knock-out?** Knock-in activates; knock-out terminates.  
**In-out parity?** Matching knock-in plus knock-out equals the vanilla, subject to matching terms and rebates.

---

## 13. Prices match the old system but Greeks do not. Why?

Possible causes: bump size, one-sided versus central differences, sticky-strike versus sticky-delta shocks, recalibration versus frozen parameters, interpolation, smoothing, numerical noise, path reuse, units and curve/dividend conventions.

**How isolate it?** Fix one snapshot, align definitions, compare one shock at a time, freeze calibration and use a simple benchmark trade.

---

## 14. Explain a binomial tree.

Build up/down stock states, calculate terminal payoffs and work backwards using risk-neutral probabilities and discounting. For American options compare continuation with immediate exercise at each node.

**Why risk-neutral probability?** It is consistent with no-arbitrage and replication, not a subjective forecast.

---

## 15. What is a volatility surface?

A volatility surface maps implied volatility across strike or moneyness and maturity. It is built from market option prices and interpolated between liquid quotes.

**Calendar arbitrage?** Total variance should generally not decrease with maturity for comparable moneyness.  
**Butterfly arbitrage?** Call prices must be convex across strike; otherwise a non-negative butterfly could have a negative price.  
**Total variance?** Implied volatility squared times maturity.

---

## 16. How would you validate a volatility surface?

Check market fit, strike convexity, calendar consistency, interpolation, extrapolation, date-to-date stability, shock behaviour, delta conventions, reproducibility and impact on representative products and Greeks.

**Negative risk-neutral density?** It indicates invalid strike interpolation or static arbitrage.

---

## 17. Delta suddenly jumped. What could cause it?

Spot crossing a strike/barrier, near expiry, observation dates, surface recalibration, corporate actions, dividends, interpolation discontinuity, bump definition, model switch, stale results or a release.

**How provide P&L explain?** Move sequentially from yesterday’s state to today’s, changing spot, vol, time, rates, dividends, correlation and calibration separately.

---

## 18. How would you migrate legacy functionality into Python?

Capture approved benchmark trades, reproduce interfaces, migrate incrementally, compare prices and Greeks automatically, investigate differences, profile performance, run in parallel and obtain approval before retirement.

**What if the legacy result is wrong?** Document it, agree the correct behaviour with model/business owners, and encode the approved result in tests.

---

# Tier 3 — Possible

## 19. Explain an autocallable.

It has periodic observation dates and redeems early if the underlying is above an autocall level. Final principal protection may depend on a downside barrier. Risks include path dependence, changing maturity, skew, dividends and correlation.

**Why difficult to hedge?** Sensitivities change sharply around observation dates and barriers.

---

## 20. Explain a digital option.

A digital pays a fixed amount if a condition is met. Its discontinuous payoff creates concentrated delta and gamma near the strike close to expiry.

**Why large hedging costs?** Tiny spot moves can require large hedge changes, and jumps make replication imperfect.

---

## 21. Why can two models fit vanilla prices but give different exotic prices?

Vanillas constrain terminal distributions at quoted maturities but not path behaviour, future smile dynamics, barrier-touch probabilities or joint distributions.

**Local versus stochastic volatility?** Local vol fits today’s surface closely; stochastic vol may provide richer future dynamics but adds calibration risk.

---

## 22. Local volatility versus stochastic volatility?

Local volatility is deterministic as a function of spot and time. Stochastic volatility makes volatility itself random.

**When use a hybrid?** When both accurate current-surface fit and more realistic future smile dynamics matter.

---

## 23. Why is an index hedge not sufficient?

It removes broad beta but leaves single-name basis, dispersion, correlation, skew, dividends, constituent mismatch and nonlinear risk.

**How hedge a worst-of basket?** Combine index and single-name hedges, monitor correlation/dispersion and manage downside skew and changing worst-name identity.

---

## 24. How handle a calibration failure during trading hours?

Assess materiality, stop unreliable output if needed, use only approved fallbacks, notify the desk, then check quotes, missing data, arbitrage, initial parameters, bounds, optimiser settings and tolerances.

**When is fallback acceptable?** Only when approved, transparent, monitored, time-limited and clearly flagged.

---

# Tier 4 — Stretch

## 25. What are vanna and volga?

Vanna is spot-vol cross-sensitivity; volga is the change in vega as volatility changes.

**Which products?** Long-dated and smile-sensitive exotics such as barriers, digitals and autocallables.

---

## 26. Why can a long barrier call have negative delta?

Near a knock-out barrier, a spot rise increases vanilla value but also increases the probability of losing the option. The knock-out effect can dominate.

**Can a long exotic have negative gamma?** Yes. Barrier and early-redemption features can create locally negative curvature.

---

## 27. How calculate Monte Carlo Greeks?

Bump-and-revalue, pathwise differentiation, likelihood ratio or adjoint methods. Common random numbers reduce noise.

**Why are discontinuous payoffs difficult?** Derivatives may not exist pathwise and small bumps can change barrier or exercise status.

---

## 28. What would make you reject a model even if it fits the market?

Unstable calibration, unrealistic parameters, arbitrage, discontinuous Greeks, poor stress or hedge behaviour, unreliable computation or disproportionate operational complexity.

**How compare models?** Compare calibration, exotic prices, Greeks, stress behaviour, smile dynamics, hedging P&L, parameter stability, performance and robustness.

---

# Last-hour memory anchors

| Topic | Anchor |
|---|---|
| Call inputs | Spot up call up; strike up call down; vol up call up; rates up call up; dividends up call down |
| Greeks | Delta = spot; gamma = change in delta; vega = vol; theta = time |
| Wrong price | Trade → timestamp → data → model → numerics → release/cache → benchmark → communicate |
| Wrong Greeks | Bump → shock convention → recalibration → interpolation → noise → units |
| Architecture | Product → snapshot → model → engine → risk → API → tests/monitoring |
| Autonomy | Investigate independently, communicate early, escalate with evidence |
| MC vs PDE | MC for high-dimensional/path-dependent; PDE for low-dimensional/barriers/early exercise |
| Chris lens | Can Jia safely represent London and diagnose live issues without constant supervision? |
