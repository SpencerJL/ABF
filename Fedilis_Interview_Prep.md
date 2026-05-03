# 🎯 Fidelis Quant Analyst – Interview Question Bank (Mel Style)

---

# 1. Mortgage Portfolio Analysis (Core Question)

## Question

We are asked to insure a residential mortgage portfolio with losses attaching at 3% and detaching at 10%. How would you
assess it?

## Answer

I would break the analysis into five parts: portfolio composition, key risk drivers, base loss, stress loss, and
structure.

First, I would segment the portfolio by LTV, borrower score, arrears status, region, and vintage. This helps identify
concentrations and hidden tail risks.

Second, I would focus on key drivers:

* Default risk: driven by affordability, borrower quality, arrears, and interest rates
* Loss given default: driven by LTV, house price decline, and recovery costs
* Concentration: region, vintage, or high-risk borrower segments

Third, I would estimate base-case loss by segment rather than relying on averages.

Fourth, I would stress the portfolio using scenarios such as:

* house price decline (10–30%)
* increased defaults
* worsening recoveries
* interest rate shocks

The key is to understand the loss distribution and probability of breaching the 3% attachment point.

Finally, I would assess whether 3% provides enough protection. If small changes push losses above 3%, the structure is
too thin.

Conclusion: I would support the deal only if losses remain well below attachment under realistic stress and the
portfolio is well diversified with reliable data.

---

# 2. Structured Credit / Tranche Risk

## Question

How do you think about risk in a tranche structure?

## Answer

I think of tranches as different layers of loss absorption.

Equity takes first losses, mezzanine absorbs intermediate losses, and senior is protected unless losses are severe.

Risk depends on:

* attachment point
* detachment point
* loss distribution of the underlying portfolio

For a mezzanine tranche, the key question is:
What is the probability that losses reach the attachment point, and how much loss occurs within that layer?

I focus on:

* sensitivity to default rates
* recovery assumptions
* correlation across assets

Conclusion: tranche risk is driven less by average loss and more by tail risk and how quickly losses accumulate.

---

# 3. Data Quality and Missing Data

## Question

What would you do if the dataset is incomplete or messy?

## Answer

This is common in practice.

First, I would identify what is missing and what is reliable.

Second, I would apply conservative assumptions where data is missing, for example:

* higher default assumptions
* lower recoveries

Third, I would test sensitivity to those assumptions to understand how much they impact the results.

Finally, I would flag key data gaps and, if material, request additional information before underwriting.

Conclusion: I would not block analysis due to imperfect data, but I would make assumptions explicit and test their
impact.

---

# 4. Stress Testing Approach

## Question

How would you design stress scenarios for a credit portfolio?

## Answer

I would focus on scenarios that impact the main drivers of loss:

* macro stress (recession, unemployment)
* asset price decline (e.g. house prices)
* interest rate increases
* liquidity stress (longer recovery timelines)

Rather than designing many scenarios, I would focus on a few meaningful ones and test sensitivity to key parameters.

The goal is to understand:

* how losses evolve
* what breaks first
* whether losses breach key thresholds

Conclusion: stress testing should be simple, targeted, and focused on downside risk.

---

# 5. Decision-Making Question

## Question

What would make you reject a deal?

## Answer

I would reject a deal if:

* the portfolio is highly sensitive to small changes in assumptions
* the attachment point is too close to expected loss
* there is significant concentration risk
* data quality is poor and uncertainty is high
* stress scenarios show losses breaching the insured layer too easily

Conclusion: I focus on downside protection rather than base-case performance.

---

# 6. Correlation and Concentration

## Question

How do you think about correlation in a portfolio?

## Answer

Correlation matters because it drives tail risk.

If assets are highly correlated, defaults can occur simultaneously under stress, leading to large losses.

I would assess correlation through:

* geographic concentration
* borrower type
* exposure to common macro drivers

Conclusion: higher correlation increases the probability of large losses and makes mezzanine risk more dangerous.

---

# 7. Translating Your Experience

## Question

How does your experience apply to this role?

## Answer

My experience is in analysing large portfolios using scenario-based frameworks.

I’ve worked on:

* stress testing portfolios under macro scenarios
* analysing sensitivity to key drivers like rates and spreads
* translating quantitative results into actionable insights

This is directly relevant to underwriting, where the goal is to understand how portfolio risk behaves under different
conditions and support decision-making.

---

# 8. Rapid Case Question

## Question

You see a portfolio with average LTV of 60%. Is it safe?

## Answer

Not necessarily.

Average metrics can be misleading.

I would want to see:

* distribution of LTVs
* exposure to high-LTV segments
* borrower quality
* regional concentration
* vintage

Conclusion: risk depends on distribution, not averages.

---

# 9. Sensitivity Analysis

## Question

What sensitivity analysis would you run?

## Answer

I would focus on key drivers:

* default rates
* recovery rates
* house price changes
* interest rates

I would test how losses change when these inputs move individually and jointly.

Conclusion: sensitivity analysis helps identify what drives risk and where the portfolio is most vulnerable.

---

# 10. Final Behavioural / Thinking Question

## Question

How do you approach a problem you don’t fully understand?

## Answer

I structure the problem first, make reasonable assumptions, and iterate.

I would:

* break the problem into key components
* identify main drivers
* proceed with assumptions
* refine as more information becomes available

Conclusion: I prioritise speed and structure over perfect understanding upfront.

# Mock Interview Q&A

### 1. Explaining Complex Analysis to Non-Technical Stakeholders

**Question:** Tell me about a time you explained a complex quantitative analysis to a non-technical stakeholder. How did
you ensure they understood?  
**Answer:** I simplified outputs into business-relevant terms, like probabilities of default over practical time
horizons. I used visualizations—clear charts showing distributions, attachment points, and expected losses—to help them
intuitively see the risk. I ensured they could act by linking the analysis directly to their decision.

### 2. Handling Poor Data Quality

**Question:** How did you handle poor data quality or missing data?  
**Answer:** I first assessed reliability—if data was unreliable, I used scenario analysis with conservative assumptions.
When possible, I flagged gaps, transparently communicated uncertainties, and tested sensitivity to assumptions. If
needed, I sought more data or adjusted the model accordingly.

### 3. Assessing an Asset-Backed Finance Opportunity

**Question:** How would you assess a new asset-backed finance deal?  
**Answer:** I’d start with portfolio composition—segmentation by LTV, vintage, arrears, and concentration. Next, I’d
identify key risk drivers: default rates, recovery rates, and correlations. I’d run a base-case and stress-case loss
analysis, then compare expected losses to the attachment point. If losses remain comfortably below attachment even in
stress, it’s likely underwritable; otherwise, we reconsider.

### 4. Motivation for the Underwriting-Focused Quant Role

**Question:** Why do you want this underwriting-focused quant role?  
**Answer:** I enjoy working close to real deals and supporting decisions. My experience in insurance and banking gives
me a holistic view. This role lets me combine technical modeling with commercial impact, and I’m motivated to contribute
directly to underwriting decisions.

---

You can copy and paste that into your bank. Let me know if you need to adjust anything further!

---

# 🎯 Final Reminder

Focus on:

* structure
* simplicity
* downside thinking
* decision-making

Avoid:

* over-complication
* excessive theory
* long explanations without conclusions

---

# Fidelis / Mel-Style Live Drill Question Bank

## How to Use This File

Practice each answer out loud in 45–75 seconds.

Your target answer style:

1. Start with structure: “I’d break this into three parts...”
2. Make assumptions quickly.
3. Focus on 2–3 key risk drivers.
4. Explain the analysis simply.
5. End with a clear underwriting conclusion.

---

# Section 1 — Technical / Credit Portfolio Drills

---

## 1. Structured Credit Transaction Backed by SME Loans — Missing Financials

### Question

Imagine we are evaluating a structured credit transaction backed by SME loans.

The tranche we are considering attaches at 5% and detaches at 15%.

You have loan-level data, but notice some loans have missing financials.

How would you handle this and still determine whether this tranche is underwritable?

### Model Answer

I would break the analysis into three parts: data quality, portfolio risk, and tranche protection.

First, I would assess the missing financials directly. I would check how material the missing data is by exposure size, sector, region, borrower type, and risk grade. If the missing financials are concentrated in high-risk or large exposures, that is more concerning than if they are small and diversified.

Second, I would apply conservative assumptions to the missing-data segment. For example, I would assign higher default rates, weaker recovery assumptions, and potentially higher correlation if the borrowers are concentrated in one sector or region. I would also run sensitivity analysis to see whether the tranche result changes materially when those assumptions are stressed.

Third, I would analyse the tranche itself. Since the tranche attaches at 5% and detaches at 15%, I would focus on the probability that portfolio losses exceed 5%, expected loss within the 5–15% layer, and whether severe recession scenarios could exhaust the tranche.

My conclusion would be: if the tranche remains protected even when I apply conservative assumptions to the loans with missing financials, the deal may be underwritable. But if the result depends heavily on optimistic assumptions for the missing-data segment, I would either ask for more information, recommend a higher attachment point, or decline the risk.

### Why This Answer Works

- It addresses missing financials upfront.
- It does not get stuck waiting for perfect data.
- It links data quality directly to underwriting decision.
- It focuses on the 5–15% tranche rather than general portfolio analysis.

---

## 2. Auto Loan Securitisation — Senior Tranche Sensitivity Analysis

### Question

We are looking at a securitisation of auto loans.

The average borrower FICO is 680, and delinquencies have ticked up recently.

What sensitivity analysis would you run to determine whether the senior tranche remains safe?

### Model Answer

I would focus on three main sensitivities: default rates, recovery rates, and residual values.

First, I would stress default rates because rising delinquencies may indicate that arrears are migrating into defaults. I would test scenarios where defaults increase by, for example, 20%, 30%, or more depending on recent delinquency trends.

Second, I would stress recoveries. In auto loans, recoveries depend on repossession efficiency, sale timelines, and used-car prices. If recoveries fall, loss given default increases.

Third, I would stress residual values. If used-car prices fall by 10–20%, recoveries could deteriorate materially, especially for higher LTV loans.

I would also check whether the average FICO of 680 hides a weak tail. The distribution matters more than the average. I would want to see how much exposure sits in lower FICO buckets and whether delinquencies are concentrated there.

My conclusion would be: if stressed losses remain far below the senior tranche attachment point, the senior tranche is likely safe. If losses approach the attachment point under realistic stress, I would reassess the structure or recommend additional protection.

### Why This Answer Works

- It names exact sensitivities upfront.
- It correctly avoids relying only on average FICO.
- It links auto collateral values to recovery risk.
- It ends with a clear tranche decision.

---

## 3. SME Loan Portfolio — Recession Stress Test

### Question

Imagine you have a portfolio of SME loans, and we are considering a recession scenario.

How would you structure a stress test to see if the portfolio breaks, and what would you look at first?

### Model Answer

I would start by defining the recession scenario, then translate it into portfolio loss drivers.

First, I would define the macro assumptions: GDP decline, weaker SME revenues, higher insolvencies, higher interest costs, and potentially weaker refinancing conditions.

Second, I would translate those assumptions into credit drivers. For SME loans, I would stress default rates, recovery rates, and default correlation. SMEs are often more vulnerable in recession because they may have weaker liquidity, lower bargaining power, and less access to refinancing.

Third, I would analyse concentration. I would check whether exposure is concentrated by sector, region, borrower size, or vintage. For example, if many borrowers are in cyclical sectors such as retail, construction, hospitality, or transport, defaults may rise together.

Fourth, I would calculate stressed losses and compare them with the attachment point or underwriting limit. I would want to know what breaks first: default rate, recovery, concentration, or timing of losses.

My conclusion would be: if the portfolio remains resilient under a credible recession and losses stay below attachment with sufficient margin, the risk may be underwritable. If losses breach attachment easily, especially due to sector or regional concentration, I would be cautious.

### Why This Answer Works

- It starts with the macro scenario.
- It converts macro stress into credit loss drivers.
- It highlights SME-specific risks.
- It focuses on what breaks first.

---

## 4. Structured Deal With Highly Uncertain Recoveries

### Question

Suppose we are evaluating a structured deal where recoveries are highly uncertain.

How would you analyse the tranche risk if recovery rates could swing widely, and what assumptions would you stress?

### Model Answer

I would focus on recovery sensitivity because uncertain recoveries can materially change loss given default and tranche loss.

First, I would create recovery scenarios rather than relying on a single recovery assumption. For example, I would test base recoveries, downside recoveries, and severe downside recoveries.

Second, I would stress both the level and timing of recoveries. Lower recovery rates increase loss severity, while longer recovery timelines can worsen cashflow timing and reduce the economic value of recoveries.

Third, I would consider what drives recovery uncertainty. That could be collateral value, legal enforcement process, asset liquidity, borrower insolvency process, or market conditions during asset sale.

Fourth, I would measure how tranche expected loss changes under each recovery assumption. For a mezzanine tranche, small recovery changes can have a large effect if portfolio losses are close to attachment.

My conclusion would be: if the tranche remains protected even under severe recovery haircuts and delayed recoveries, the risk may be acceptable. If the tranche loss is highly sensitive to recovery assumptions, I would recommend more conservative pricing, higher attachment, lower limit, or declining the deal.

### Why This Answer Works

- It separates recovery level and recovery timing.
- It connects recovery uncertainty directly to tranche loss.
- It shows practical underwriting actions.

---

## 5. Consumer Loan Portfolio With Rising Inflation

### Question

Suppose you are given a portfolio of consumer loans and inflation is rising.

What key metrics would you stress first, and how would you determine if portfolio performance is deteriorating?

### Model Answer

I would focus on borrower affordability, delinquency migration, defaults, and recoveries.

First, rising inflation reduces disposable income, so I would stress borrower affordability. I would look at whether borrowers have enough income buffer after essential spending.

Second, I would monitor early arrears and delinquency migration. For consumer loans, deterioration often appears first in 30-day and 60-day arrears before moving into defaults.

Third, I would stress default rates. I would test whether higher living costs and potentially higher interest rates push more borrowers into default.

Fourth, I would stress recoveries and loss given default. If household finances deteriorate broadly, recoveries may weaken and collection timelines may extend.

I would also segment the portfolio by borrower score, income band, vintage, loan type, and geography, because inflation may hit weaker borrowers first.

My conclusion would be: if arrears are rising but remain contained and stressed losses stay below attachment, the portfolio may still be underwritable. But if arrears migration accelerates and stressed losses move quickly into the insured layer, that would be a warning sign.

### Why This Answer Works

- It focuses on consumer affordability rather than abstract market risk.
- It uses arrears migration as an early warning signal.
- It links inflation to defaults and recoveries.

---

# Section 2 — Full Mock Interview Questions

---

## 6. Explaining Complex Quantitative Analysis to Non-Technical Stakeholders

### Question

Could you walk me through a time when you had to explain a complex quantitative analysis to a non-technical stakeholder?

How did you ensure they understood and could act on it?

### Model Answer

Yes. In my previous insurance-related work, I often had to explain modelling outputs to brokers, underwriters, or business stakeholders who were not interested in the technical mechanics of the model, but needed to understand the underwriting implication.

My approach was to translate the model output into decision-relevant terms. Instead of focusing on methodology, I focused on questions such as: what is the expected loss, how severe could the downside be, where does the attachment point sit relative to the loss distribution, and what assumptions drive the result.

I also used visualisation heavily. For example, I would show loss distributions, stress scenarios, and key thresholds so stakeholders could see whether the deal looked safe or whether losses were close to the risk layer.

The key was to connect the analysis directly to the decision. My goal was not to show technical complexity, but to help the stakeholder understand the risk, the uncertainty, and the action they should take.

### Short Interview Version

I try to translate quantitative analysis into the decision it supports. For underwriters, that means focusing on expected loss, downside risk, key assumptions, and where the risk sits relative to attachment. I use simple visuals such as loss distributions and stress comparisons, so the stakeholder can see the risk clearly and act on it.

### Why This Answer Works

- It shows business communication.
- It avoids sounding too technical.
- It directly fits the role’s business-facing modelling requirement.

---

## 7. Handling Poor Data Quality or Missing Data

### Question

Tell me about a time when data quality was poor.

What steps did you take to handle uncertainty or missing data in your analysis?

### Model Answer

Poor data quality is common in insurance and portfolio analysis, so my first step is always to separate what is reliable from what is uncertain.

I would begin with data validation: checking completeness, consistency, duplicates, outliers, definitions, and reconciliation to known totals. Then I would identify whether missing data is random or concentrated in a specific segment, because concentrated missing data can create hidden risk.

Where data is missing, I would apply conservative assumptions and test sensitivities. For example, if borrower financials or loss history are incomplete, I would use more conservative default, recovery, or severity assumptions for that segment.

I would also communicate the uncertainty clearly. I would tell stakeholders what data is missing, how I treated it, how sensitive the result is, and whether more information is required before making an underwriting decision.

My conclusion would be: I would not let imperfect data stop the analysis, but I would make uncertainty explicit and avoid giving false precision.

### Short Interview Version

I first validate the data and identify whether the gaps are material. Then I apply conservative assumptions to missing or unreliable segments and run sensitivity analysis. Most importantly, I communicate clearly what is uncertain and whether the decision is robust to those assumptions.

### Why This Answer Works

- It is practical.
- It shows control of data uncertainty.
- It links data gaps to underwriting judgement.

---

## 8. Assessing a New Asset-Backed Finance Opportunity

### Question

Could you walk me through how you would assess a new asset-backed finance opportunity that lands on your desk?

What is your step-by-step approach?

### Model Answer

I would break it into five steps: understand the asset, analyse portfolio quality, identify key risk drivers, run base and stress cases, and assess the structure.

First, I would understand the underlying asset: whether it is mortgages, consumer loans, SME loans, auto loans, leasing, or another asset type. The asset determines the key risk drivers.

Second, I would analyse portfolio composition. I would segment by factors such as LTV, borrower quality, arrears, vintage, geography, sector, and loan size. I would look for concentrations and weak segments.

Third, I would identify the main loss drivers: default rate, recovery rate, loss given default, prepayment if relevant, correlation, and timing of losses.

Fourth, I would estimate base-case and stress-case losses. I would focus on whether losses remain manageable under recession, collateral value decline, higher defaults, and lower recoveries.

Fifth, I would assess the structure. I would compare expected and stressed losses with attachment and detachment points, and assess whether the proposed protection gives enough margin.

My conclusion would be: I would support underwriting only if the portfolio quality, data, structure, and stress results all provide sufficient downside protection.

### Short Interview Version

I would start with the asset and portfolio composition, then identify the key loss drivers, run base and stress loss analysis, and compare the result with the structure. The key question is whether the attachment point gives enough protection under realistic downside scenarios.

### Why This Answer Works

- It is structured.
- It covers the JD: portfolio analysis, modelling, stress testing, underwriting support.
- It ends with decision relevance.

---

## 9. Motivation for the Underwriting-Focused Quant Role

### Question

Why do you want to move into this underwriting-focused Quant Analyst role, given your prior experience?

### Model Answer

The attraction for me is that the role combines quantitative modelling with direct underwriting impact.

Earlier in my career, I worked closely with insurance stakeholders and enjoyed translating technical analysis into something useful for risk selection, pricing, and underwriting decisions. More recently, at Deutsche Bank, I have developed portfolio-level analytics and scenario-based modelling, which has strengthened my understanding of credit exposure, sensitivities, and stress behaviour.

This role sits at the intersection of those experiences. It is hands-on modelling, but it is also close to the business and the underwriting decision. That is important to me because I want my technical work to be directly useful, not just model development in isolation.

I also like the visibility of the role. Supporting underwriters and decision-makers on complex credit portfolios is exactly where I think my background in portfolio analytics, credit modelling, and business communication can add value.

### Short Interview Version

I am attracted to this role because it combines technical modelling with real underwriting decisions. My background gives me both sides: insurance-facing analytics and banking portfolio risk experience. I want to use modelling not as an isolated technical exercise, but as a tool to help underwriters understand risk and make better decisions.

### Why This Answer Works

- It sounds motivated but not desperate.
- It connects insurance + DB experience.
- It fits the JD language: modelling, business interaction, visibility, underwriting support.

---

# Section 3 — Common Corrections From Today’s Drill

---

## Correction 1 — Avoid Overusing DV01 / CS01

DV01 and CS01 are relevant to fixed income and credit spread sensitivity, but they are not always the right first answer for consumer loans, SME loans, or mortgage portfolios.

For credit portfolios, lead with:

- default rates
- recovery rates
- loss given default
- arrears / delinquency migration
- collateral values
- concentration
- correlation

Use DV01 / CS01 only if the portfolio has explicit rate or spread exposure where those sensitivities are directly relevant.

---

## Correction 2 — Do Not Start Too Slowly

Avoid starting with:

> “There are a couple of things I need to consider...”

Better:

> “I’d break this into three parts: data quality, loss drivers, and tranche protection.”

This sounds faster, more senior, and more decisive.

---

## Correction 3 — Always Address the Specific Prompt First

If the question mentions missing financials, start with missing financials.

If the question mentions delinquencies, start with delinquency migration.

If the question mentions senior tranche, start with attachment protection.

Do not give a generic portfolio framework before answering the specific issue.

---

## Correction 4 — Conclude Every Answer

End with a clear underwriting decision:

> “If stressed losses remain below attachment with sufficient margin, I would be comfortable supporting the deal. If losses breach attachment under realistic stress, I would recommend changing the structure or declining the risk.”

---

# Section 4 — Memorisation Templates

---

## Template 1 — Portfolio / Deal Assessment

I would break this into asset quality, loss drivers, stress analysis, and structure.

First, I would segment the portfolio by the most relevant risk factors.

Second, I would identify the main loss drivers: defaults, recoveries, collateral values, concentration, and timing.

Third, I would run base-case and downside scenarios.

Finally, I would compare the loss results with the attachment and detachment points.

If stressed losses remain comfortably below attachment, the deal may be underwritable. If losses move into the insured layer too easily, I would reassess the structure or decline.

---

## Template 2 — Missing Data

I would first assess whether the missing data is material and whether it is concentrated in risky segments.

Then I would apply conservative assumptions to those exposures and run sensitivities.

I would be transparent about the data gaps and quantify whether the underwriting conclusion is robust to those assumptions.

If the conclusion changes materially depending on the missing data, I would ask for more information or recommend a more conservative structure.

---

## Template 3 — Sensitivity Analysis

I would stress the key drivers: default rates, recovery rates, collateral values, and correlation.

Then I would test how those shocks affect expected loss, tail loss, and the probability of breaching attachment.

The objective is not only to estimate expected loss, but to identify whether the structure remains protected under downside conditions.

---

## Template 4 — Behavioural Communication Answer

My approach is to translate technical output into business decisions.

I focus on the key numbers that matter to the stakeholder: expected loss, downside risk, sensitivity to assumptions, and the decision threshold.

I use simple visuals and avoid unnecessary modelling detail unless asked.

The goal is to help the stakeholder understand the risk clearly and act on it.

---

# Final Interview Reminder

Mel is likely to test both technical judgement and communication.

She has deep experience in mortgage, portfolio credit, underwriting, and risk pricing, so do not assume the interview will be purely commercial.

Your best positioning:

> I bring structured analytics, portfolio modelling, and clear communication to support underwriting decisions.

Your answer style:

> Fast structure, simple drivers, downside first, clear conclusion.
