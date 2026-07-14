# Scotiabank GAFE Equities — CV Defence

**Prepared for Jia Lu — Chris interview focus**

> Use this section to make the CV narrative concrete and defensible. Describe the business problem, your exact contribution, the implementation, controls, users and limitations. Distinguish clearly between what you personally owned, what you contributed to and what another team owned.

## Answer structure

For project questions:

> **Problem → users → data → method/tool → my ownership → controls/testing → result → limitation/improvement**

For ownership questions:

> **I personally did X; I contributed to Y; Z was owned by another team.**

---

# Tier 1 — Highest probability

## 1. Tell me about the most significant Python analytics tool you built at Deutsche Bank.

### What Chris is testing

- Was it a real implemented workflow or only analysis in a notebook?
- What did you personally build?
- Can you explain architecture, lineage, validation and users?

### Model answer

> “The business problem was to consolidate and analyse portfolio-level risk data consistently across books. I worked on the Python workflow that ingested source extracts, applied mappings and validation rules, aggregated sensitivities and scenario outputs, and produced controlled outputs for risk review. My direct contribution was in the data transformation, reconciliation logic, exception handling and analytical output layer. I worked with upstream data and downstream stakeholders rather than owning every source system. The main controls were completeness checks, reconciliations to source totals, tolerance-based exception reporting and repeatable runs. The value was a more consistent and faster investigation process. If redesigning it, I would further separate ingestion, validation, calculation and presentation and strengthen automated regression testing.”

### Follow-up: What was the architecture?

```text
source extracts / APIs
        ↓
schema and quality validation
        ↓
normalisation and mapping
        ↓
aggregation / sensitivity logic
        ↓
reconciliation and exception report
        ↓
controlled output / reporting layer
```

Do not call it a pricing platform if it was a risk-data workflow.

### Follow-up: Who used it?

> “The immediate users were risk and quantitative stakeholders reviewing portfolio movements and scenario outputs. The outputs also supported downstream reporting and investigation discussions.”

### Follow-up: How was it tested?

> “I used source-to-output reconciliations, known-sample tests, tolerance checks, previous-run comparisons and targeted tests for mapping and aggregation logic. For changes, I compared old and new outputs on representative portfolios before release.”

### Follow-up: What did you personally own?

> “I owned substantial parts of the Python transformation, aggregation, validation and investigation workflow. I did not own the upstream risk engines or the formal pricing models that generated every sensitivity.”

---

## 2. Your CV says ‘scalable analytical frameworks’. What do you mean by scalable?

### Model answer

> “I do not use scalable only to mean faster. I mean the framework could handle more books and data without manual redesign, reuse common logic across portfolios, separate configuration from code, provide consistent controls and be extended without breaking existing outputs. In practice that meant modular functions, mapping tables rather than hard-coded rules, batch processing, clear data contracts, logging, reconciliation and repeatable outputs.”

### Follow-up: How large was the data?

> “It was large enough that manual processing and row-by-row logic were not sustainable. The important engineering issue was reducing repeated transformations and I/O, using vectorised operations where appropriate and ensuring aggregate totals reconciled.”

### Follow-up: Was it production software?

> “It was an implemented internal analytical workflow used in recurring processes. I would distinguish that from owning a low-latency front-office pricing service. The production requirements were reproducibility, data quality, controls and reliable scheduled execution rather than microsecond latency.”

---

## 3. Walk me through your sensitivity analytics work. How were DV01 and CS01 calculated?

### Model answer

> “At a conceptual level, DV01 and CS01 measure the change in value for small interest-rate or credit-spread shocks. In my role, my strongest hands-on ownership was around consolidating, validating, aggregating and analysing sensitivities produced across portfolios, rather than building every underlying trade-level pricing model. I worked with book and hierarchy mappings, sign and unit consistency, scenario aggregation, movement analysis and reconciliation to source systems. When a sensitivity moved unexpectedly, I decomposed it by book, product, curve or tenor and checked trade population, market data, mapping and upstream calculation changes.”

### Follow-up: Did you calculate them by bump-and-revalue?

> “The underlying risk engines may use bump-and-revalue or analytical sensitivities depending on the product. My direct work was primarily on the controlled aggregation and analysis of those results. I understand the distinction and would verify the exact methodology before comparing outputs.”

### Follow-up: What could make DV01 double overnight?

- new or matured trades;
- market moves changing duration;
- curve or mapping changes;
- notional or currency conversion issues;
- sign or unit errors;
- stale, missing or duplicated data;
- upstream model or release changes.

---

## 4. Your CV mentions CVA exposure modelling. What exactly did you do?

**This is a danger question because Chris has deep XVA experience.**

### Model answer

> “My involvement was mainly in exposure and risk-data workflows supporting CVA monitoring and reporting. I worked with counterparty-level outputs, aggregation, validation, scenario comparison and movement investigation. I did not own the core stochastic exposure simulation or the bank’s full CVA pricing methodology. My contribution was making the outputs reliable, explainable and usable for risk review.”

### Follow-up: How is CVA calculated conceptually?

> “Conceptually, CVA is the discounted expected loss from counterparty default over the life of the portfolio. It combines expected positive exposure, default probability and loss given default, with netting, collateral and wrong-way-risk considerations where relevant.”

### Follow-up: Where does EPE come from?

> “From simulated or otherwise modelled future exposure profiles under market scenarios, after applying netting and collateral assumptions. Expected positive exposure is the average positive exposure at each future time point.”

### Follow-up: What did you not own?

> “I did not own the Monte Carlo exposure engine, model calibration or formal XVA methodology approval.”

---

## 5. Tell me about a time you investigated a major risk movement.

### Model answer

> “I first fixed the comparison population and valuation dates, then decomposed the movement by portfolio, product, risk factor and source. I checked trade population, market-data changes, mapping changes, currency conversion and upstream releases. Once I isolated the driver, I reconciled it to an independent source or prior-day result, quantified the contribution and communicated whether it was economic, data-related or process-related.”

### Follow-up: Give me a concrete example.

Use a real example that includes:

- what moved;
- how you decomposed it;
- the root cause;
- how you validated it;
- what changed afterward.

Avoid claiming a trader-facing pricing incident if it was actually a reporting or mapping issue.

---

## 6. How much direct interaction have you had with traders?

### Model answer

> “My interaction has been more through risk, quantitative and technology workflows than sitting directly on a trading desk. I have supported analyses relevant to trading portfolios and communicated with stakeholders around risk movements and data issues, but I would not overstate the amount of direct trader partnership. That is one reason this role is attractive: it would move me closer to live pricing and desk decision support.”

### Follow-up: How will you adapt to a trader-facing environment?

> “I would focus on concise communication: impact first, then cause, evidence, workaround and next action. My experience handling time-sensitive risk questions has trained me to structure ambiguous problems; I now need to apply that discipline closer to the desk.”

### Follow-up: What if a trader challenges you aggressively?

> “I would not become defensive. I would clarify the exact discrepancy, reproduce it using the same inputs, state what I know and do not know, and return with evidence and a time-bound update.”

---

## 7. Your CV says advanced Python. What makes your Python advanced?

### Model answer

> “My strongest Python experience is in quantitative data workflows, analytical tooling, automation, validation and portfolio-level processing using pandas, NumPy and related libraries. I am comfortable structuring reusable modules, profiling workflows, handling exceptions, testing transformations and integrating data sources. I would distinguish that from being a specialist in low-latency systems or large distributed architecture. I am actively strengthening the production-engineering side because this role requires a higher standard of library design and testing.”

### Follow-up: How would you improve slow pandas code?

- profile first;
- remove repeated I/O and API calls;
- avoid `iterrows` and Python loops;
- vectorise with NumPy or pandas;
- select only required columns;
- optimise dtypes;
- avoid repeated copies and merges;
- batch operations;
- cache stable inputs.

### Follow-up: List versus tuple?

A list is mutable; a tuple is immutable. Tuples communicate fixed structure and can be hashable when their contents are hashable.

### Follow-up: Composition versus inheritance?

Prefer composition when behaviours vary independently; it reduces coupling and avoids deep inheritance hierarchies.

---

## 8. Tell me about the AI-agent or LLM work on your CV.

### Model answer

> “The work was exploratory and prototype-oriented rather than a fully autonomous production trading agent. The objective was to improve analytical workflows such as retrieving relevant documentation, combining structured data with user questions and accelerating recurring investigations. I worked on Python-based prototypes, prompt and context design, structured outputs and practical controls. The focus was workflow assistance, not replacing validated pricing or risk calculations.”

### Follow-up: Why use an LLM rather than rules?

> “Rules are preferable for deterministic calculations and controls. An LLM adds value where the input is unstructured, language varies or information must be retrieved and synthesised. I would keep numerical calculations and approvals in deterministic services.”

### Follow-up: How do you control hallucination?

- retrieval from approved sources;
- citations and traceability;
- structured schemas;
- tool-based calculation rather than free-text arithmetic;
- validation rules;
- confidence and exception handling;
- human approval for material decisions;
- logging and test sets.

### Follow-up: MCP or agents?

> “The core idea is controlled tool access: the model interprets the request, invokes approved data or calculation services, and returns a structured answer with provenance. The exact framework is secondary to permissions, schemas, observability and validation.”

---

# Tier 2 — Likely

## 9. Tell me about the Howden credit portfolio platform.

### Model answer

> “The platform supported portfolio-level analysis of defaults, recoveries and losses under different assumptions and scenarios. The workflow combined exposure data, risk-driver assumptions and simulation or scenario logic to produce loss distributions, concentration views and stress results. My role included model development, analytical design, implementation and translating outputs for business users.”

### Follow-up: What products were involved?

Answer only with the actual portfolio types you worked on, such as loans, credit insurance or structured-credit exposures. Do not imply liquid CDS pricing if it was not part of the work.

### Follow-up: How did you validate it?

- benchmark expected loss;
- test limiting cases;
- compare simulation with analytical cases;
- test sensitivity to PD, recovery and correlation;
- backtest where data allowed;
- reconcile exposures;
- review assumptions with stakeholders.

### Follow-up: Was it a pricing model?

> “It was primarily a portfolio risk and scenario-analysis platform rather than a front-office mark-to-market pricing library.”

---

## 10. Explain the default and loss-distribution modelling you performed.

### Model answer

> “At portfolio level, the key elements are exposure, probability of default, loss given default and dependence across obligors. Expected loss can be calculated from the first three components, while the full loss distribution requires modelling default dependence and scenario uncertainty. I used the framework to examine concentration, tail loss and sensitivity to default and recovery assumptions.”

### Follow-up: How did you model correlation?

Use only the method you genuinely used. At minimum:

> “The purpose was to capture that defaults are not independent and become more clustered under common stress. I would be precise about whether the implementation used a factor approach, scenario correlation or another method.”

---

## 11. Your CV mentions time-series and statistical modelling at Deutsche Bank. Give an example.

### Model answer

> “My use of statistical methods has mainly supported movement analysis, scenario comparison, data-quality assessment and understanding risk-driver behaviour rather than owning a production trading forecast model. I have used time-series concepts to examine changes, identify unusual behaviour and support analytical interpretation.”

### Follow-up: What model did you use and why?

Only name a model you can explain fully. A simple and defensible example is better than mentioning an advanced model vaguely.

---

## 12. How did you collaborate with technology teams?

### Model answer

> “I translated analytical requirements into data and workflow specifications, clarified input/output contracts, tested changes on representative cases, investigated discrepancies and supported controlled rollout. I was often the bridge between the analytical user and the implementation or data teams.”

### Follow-up: Describe a disagreement.

Use a real case involving scope, data quality, timelines or interpretation. Show evidence-based resolution rather than blame.

---

## 13. What was the hardest technical problem in your current role?

Choose a problem detailed enough to survive five follow-ups:

- inconsistent identifiers and mappings across sources;
- reconciling aggregated risk to source totals;
- identifying duplicated or missing populations;
- improving a slow recurring pipeline;
- explaining a large day-on-day movement.

Use:

> **scale and ambiguity → diagnosis → implementation → controls → result → improvement**

---

## 14. Why move from your current role into equity-derivatives GAFE?

### Model answer

> “My current role has strengthened my ability to work with large risk datasets, sensitivities, scenarios and controlled Python workflows, but it is more portfolio-risk and process-oriented than the work I ultimately want to do. I want to move closer to product valuation, model behaviour and direct desk problem-solving. The GAFE role combines quantitative reasoning, Python platform development and live business support, so it is a demanding but coherent next step rather than an unrelated change.”

### Follow-up: Why equities if your background is rates and credit?

> “The product layer is new, but the transferable foundations are no-arbitrage, sensitivities, scenario analysis, numerical methods, data controls and model investigation. I recognise that volatility and equity-structured-product depth are the main learning curve, and I have been rebuilding those deliberately.”

---

## 15. Why did a mechanical-engineering PhD move into quantitative finance?

### Model answer

> “My PhD involved mathematical modelling, numerical simulation, uncertain physical systems and translating theory into measurable outputs. I became increasingly interested in applying the same quantitative discipline to financial risk and decision-making. The transition through financial engineering, portfolio modelling and risk analytics was therefore based on transferable modelling skills rather than a sudden change.”

### Follow-up: What specifically transfers?

- modelling assumptions;
- numerical methods;
- calibration and validation;
- sensitivity analysis;
- handling noisy data;
- explaining model limitations;
- research discipline.

---

# Tier 3 — Targeted follow-ups

## 16. What does ‘portfolio optimisation’ mean in your Howden work?

> “The work supported portfolio selection and risk trade-offs by showing expected loss, concentration and stress sensitivity under different portfolio compositions. I would not present it as high-frequency or market-neutral trading optimisation.”

---

## 17. What is the difference between reporting infrastructure and a pricing library?

> “Reporting infrastructure consumes and controls outputs, aggregates them and makes them explainable. A pricing library owns the valuation logic, models, numerical methods and sensitivities. My current experience is stronger on the first category, with quantitative understanding of the second. This role is attractive because it would deepen direct ownership of pricing and model implementation.”

---

## 18. What would you need to learn in the first three months?

> “The priority would be the desk’s product set, the existing Python architecture, volatility-surface conventions, approved models, production-support procedures and the division of responsibility across London, New York and Toronto. I would aim first to own standard investigations and small controlled changes, then broaden into product and model work.”

---

## 19. What would your previous manager say is your main development area?

> “They would probably say I can go very deep analytically and should continue becoming faster at converting that analysis into a concise commercial recommendation. I have improved by leading with impact and decision, then providing technical detail only as needed.”

Alternative:

> “My software experience has grown from analytical workflows, so I am deliberately strengthening formal architecture, automated testing and production-engineering discipline.”

---

## 20. What claim on your CV are you most proud of, and which requires the most context?

> “I am most proud of building analytical workflows that made complex portfolio outputs more reliable and explainable. The claim that requires most context is exposure to CVA and trading portfolios: it is genuine, but my ownership has been stronger in data, aggregation, controls and analysis than in the core stochastic pricing engine. I would rather make that distinction clearly and show how the experience transfers.”

---

# CV danger map

| CV phrase | What Chris may infer | Safe clarification |
|---|---|---|
| Python-based analytics and risk tooling | Production pricing-library ownership | Recurring controlled analytical workflows; clarify exact components owned |
| CVA exposure modelling | Core XVA Monte Carlo or model ownership | Exposure aggregation, validation, monitoring and movement analysis |
| Collaborate closely with trading | Daily direct desk partnership | Portfolio-facing stakeholder work; do not overstate trader contact |
| Advanced Python | Senior software engineer | Strong analytical Python; developing deeper production architecture skills |
| Scalable analytical frameworks | Distributed or low-latency platform | Modular, reusable, controlled and able to handle broader data and portfolio scope |
| Portfolio optimisation | Trading optimiser | Portfolio composition, concentration and scenario trade-off analysis |
| Structured products | Direct exotic-pricing ownership | Exposure to structured portfolios and analytics; clarify product depth |
| AI agents | Production autonomous system | Controlled prototypes for retrieval and workflow assistance |

---

# Five questions to practise tonight

1. **Describe the biggest Python workflow you personally built.**
2. **What exactly was your ownership in CVA?**
3. **How were DV01 and CS01 generated and used?**
4. **How much direct trader interaction have you had?**
5. **What does advanced Python mean in your case?**

For each answer, state:

- the concrete problem;
- your exact contribution;
- one technical detail;
- one control or test;
- one result;
- one boundary of your ownership.
