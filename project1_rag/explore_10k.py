# %%
"""
Explore 10-K filings using edgartools.
Sample companies: Disney (DIS), Avis Budget (CAR), PagerDuty (PD)
"""

from edgar import set_identity, Company

set_identity("Matt Gentile mattgentile@example.com")

# %% [markdown]
# ## 1. Company Lookup

# %%
TICKERS = ["DIS", "CAR", "PD"]
companies = {t: Company(t) for t in TICKERS}

print("=" * 70)
print("COMPANY OVERVIEW")
print("=" * 70)
for ticker, co in companies.items():
    print(f"\n--- {co.name} ({ticker}) ---")
    print(f"  CIK:                {co.cik}")
    print(f"  Industry:           {co.industry}")
    print(f"  SIC:                {co.sic}")
    print(f"  Tickers:            {co.tickers}")
    print(f"  Exchanges:          {co.get_exchanges()}")
    print(f"  Fiscal Year End:    {co.fiscal_year_end}")

# %% [markdown]
# ## 2. List Recent 10-K Filings

# %%
print("\n" + "=" * 70)
print("RECENT 10-K FILINGS (last 5)")
print("=" * 70)

for ticker, co in companies.items():
    filings_10k = co.get_filings(form="10-K", amendments=False)
    recent = filings_10k.head(5)
    print(f"\n--- {co.name} ---")
    for f in recent:
        print(f"  {f.filing_date}  period={f.period_of_report}  accession={f.accession_no}")

# %% [markdown]
# ## 3. Latest 10-K Deep Dive

# %%
print("\n" + "=" * 70)
print("LATEST 10-K — STRUCTURED DATA")
print("=" * 70)

for ticker, co in companies.items():
    filing = co.get_filings(form="10-K", amendments=False).latest()
    tenk = filing.obj()

    print(f"\n{'─' * 70}")
    print(f"  {co.name} — Filed {filing.filing_date}, Period {filing.period_of_report}")
    print(f"{'─' * 70}")

    # 3a. Filing metadata
    print(f"  Accession #:   {filing.accession_no}")
    print(f"  XBRL:          {getattr(filing, 'is_xbrl', 'N/A')}")
    print(f"  Homepage:      {filing.homepage_url}")

    # 3b. Document sections available on TenK
    for section_name in ["business", "risk_factors", "management_discussion"]:
        section = getattr(tenk, section_name, None)
        if section:
            preview = str(section)[:200].replace("\n", " ")
            print(f"\n  [{section_name.upper()}] (first 200 chars):")
            print(f"    {preview}...")
        else:
            print(f"\n  [{section_name.upper()}] — not available")

    # 3c. Financial statements
    if tenk.financials:
        fin = tenk.financials
        print(f"\n  [INCOME STATEMENT]")
        print(f"    {fin.income_statement}")

        print(f"\n  [BALANCE SHEET]")
        print(f"    {fin.balance_sheet}")

        print(f"\n  [CASH FLOW STATEMENT]")
        print(f"    {fin.cash_flow_statement}")
    else:
        print("\n  [FINANCIALS] — no XBRL financial data available")

# %% [markdown]
# ## 4. Key Financial Metrics

# %%
print("\n" + "=" * 70)
print("KEY FINANCIAL METRICS (from company.get_financials())")
print("=" * 70)

for ticker, co in companies.items():
    print(f"\n--- {co.name} ---")
    try:
        fin = co.get_financials()
        revenue = fin.get_revenue()
        net_income = fin.get_net_income()
        total_assets = fin.get_total_assets()
        total_liabs = fin.get_total_liabilities()
        equity = fin.get_stockholders_equity()
        ocf = fin.get_operating_cash_flow()
        fcf = fin.get_free_cash_flow()

        def fmt(val):
            if val is None:
                return "N/A"
            return f"${val / 1e6:,.0f}M"

        print(f"  Revenue:              {fmt(revenue)}")
        print(f"  Net Income:           {fmt(net_income)}")
        print(f"  Total Assets:         {fmt(total_assets)}")
        print(f"  Total Liabilities:    {fmt(total_liabs)}")
        print(f"  Stockholders Equity:  {fmt(equity)}")
        print(f"  Operating Cash Flow:  {fmt(ocf)}")
        print(f"  Free Cash Flow:       {fmt(fcf)}")
    except Exception as e:
        print(f"  Error: {e}")

# %% [markdown]
# ## 5. Search Within a 10-K

# %%
print("\n" + "=" * 70)
print("SEARCH WITHIN LATEST 10-K")
print("=" * 70)

SEARCH_TERMS = ["artificial intelligence", "risk", "competition"]

for ticker, co in companies.items():
    filing = co.get_filings(form="10-K", amendments=False).latest()
    print(f"\n--- {co.name} ---")
    for term in SEARCH_TERMS:
        results = filing.search(term)
        print(f"  '{term}': {len(results)} mentions")

# %% [markdown]
# ## 6. Attachments & Exhibits

# %%
print("\n" + "=" * 70)
print("ATTACHMENTS & EXHIBITS (latest 10-K)")
print("=" * 70)

for ticker, co in companies.items():
    filing = co.get_filings(form="10-K", amendments=False).latest()
    attachments = filing.attachments
    print(f"\n--- {co.name} ({len(attachments)} attachments) ---")
    for att in list(attachments)[:10]:
        desc = getattr(att, "description", "") or ""
        doc_type = getattr(att, "document_type", "") or ""
        print(f"  [{doc_type}] {desc[:80]}")

# %% [markdown]
# ## 7. Markdown Export (RAG-ready)

# %%
print("\n" + "=" * 70)
print("MARKDOWN EXPORT — for RAG ingestion")
print("=" * 70)

for ticker, co in companies.items():
    filing = co.get_filings(form="10-K", amendments=False).latest()
    md = filing.markdown()
    filename = f"{ticker.lower()}_10k.md"
    with open(filename, "w") as f:
        f.write(md)
    char_count = len(md)
    print(f"  {co.name} → {filename}  ({char_count:,} chars)")

print("\nDone.")
