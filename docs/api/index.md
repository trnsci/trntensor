# API Reference

trntensor's public API is organized into four modules:

- **[einsum](einsum.md)** — `einsum`, `multi_einsum`
- **[plan](plan.md)** — `plan_contraction`, `ContractionPlan`, `estimate_flops`
- **[decompose](decompose.md)** — `cp_decompose`, `tucker_decompose`, and reconstructors
- **[quantum](quantum.md)** — fused chemistry primitives (`mp2_energy`, `ao_to_mo_transform`, `df_exchange`, `df_coulomb`)
- **[nki](nki.md)** — backend dispatch (`set_backend`, `get_backend`, `HAS_NKI`)
