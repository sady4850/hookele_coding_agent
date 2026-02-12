---
name: R/Stan to Python Migration
description: Guidance for converting R/RStan code to Python/PyStan 3. Covers environment setup, API mapping, Stan model refactoring, posterior extraction shapes, background execution for long sampling, and output formatting. Use this skill for RStan-to-PyStan conversion tasks, Bayesian inference migration, or any Stan model porting between R and Python.
---

# RStan to PyStan 3 Conversion

## Key Insight: Stan Model Code is Language-Agnostic

The Stan modeling language is identical between RStan and PyStan. Copy the Stan model code directly. The conversion challenge is the **wrapper code**: data preparation, sampler invocation, and posterior extraction.

## Pre-Conversion Checklist (Do All Before Writing Code)

1. **Install build tools FIRST** -- PyStan compiles Stan to C++ at runtime:
   ```bash
   apt-get update && apt-get install -y build-essential
   pip install pystan==3.10.0 numpy --break-system-packages
   ```
2. **Import is `stan` not `pystan`**: `import stan`
3. **Read the complete R script** -- identify all hyperparameters, data transformations, and output format
4. **Read `meta_public.json`** -- contains seed, dimensions (P), jitter, etc.
5. **Inspect data files** -- verify column counts match the R script's expectations (e.g. `H = cbind(1, X[,1], X[,2])`)

## CRITICAL: Model Refactoring (MANDATORY)

PyStan 3 writes `transformed parameters` to disk for every sample, causing massive slowdown with large matrices. **Move matrix computations from `transformed parameters` into the `model` block as local variables:**

```stan
// BAD (100x slower in PyStan 3):
transformed parameters {
  matrix[N,N] K = ...;
  matrix[N,N] L_K = cholesky_decompose(K);
}

// GOOD: Inside model block as local variables
model {
  matrix[N,N] K = cov_func(...) + diag_matrix(rep_vector(square(sigma), N));
  matrix[N,N] L_K = cholesky_decompose(K);
  y ~ multi_normal_cholesky(mu, L_K);
}
```

If `generated quantities` also uses these matrices, **duplicate the computation there** as local variables.

## Hyperparameter Mapping (RStan -> PyStan 3)

| RStan Parameter | PyStan 3 Equivalent | Notes |
|-----------------|---------------------|-------|
| `iter=2000, warmup=1000` | `num_warmup=1000, num_samples=500` | See calculation below |
| `chains` | `num_chains` | |
| `thin` | `num_thin` | |
| `adapt_delta` | `delta` | |
| `max_treedepth` | `max_depth` | |
| `init_r` | `init_radius` | |
| `seed` | N/A in `sample()` | Set via `random_seed` in `stan.build()` only |

**Critical num_samples calculation:**
```
num_samples = (RStan iter - RStan warmup) / RStan thin
```
Example: `iter=2000, warmup=1000, thin=2` -> `num_samples = (2000-1000)/2 = 500`

### Parameters to OMIT (unsupported in PyStan 3 `sample()`)

Do NOT pass: `adapt_gamma`, `adapt_kappa`, `adapt_t0`, `adapt_init_buffer`, `adapt_term_buffer`, `adapt_window`, `save_warmup`, `refresh`, `seed`, `random_seed`

## CRITICAL: Posterior Extraction -- Array Shapes

**This is the #1 cause of failure.** PyStan 3 returns arrays with shape `(param_dims, num_draws)` -- TRANSPOSED from what you might expect.

```python
# Scalar parameters (alpha, sigma): shape = (num_draws,)
alpha_mean = np.asarray(fit["alpha"]).mean()          # single number

# Vector parameters (rho[3], beta[3]): shape = (3, num_draws)
rho_means = np.asarray(fit["rho"]).mean(axis=1)       # 3 numbers, one per dimension
beta_means = np.asarray(fit["beta"]).mean(axis=1)      # 3 numbers, one per dimension
```

**ALWAYS verify shapes before computing statistics:**
```python
for param in ["alpha", "sigma", "rho", "beta"]:
    arr = np.asarray(fit[param])
    print(f"{param}: shape={arr.shape}")
```

### Common Shape Mistakes

| Mistake | Result | Correct |
|---------|--------|---------|
| `fit["rho"].mean()` | 1 scalar (wrong) | `fit["rho"].mean(axis=1)` -> 3 values |
| `fit["rho"].mean(axis=0)` | num_draws values (wrong) | `fit["rho"].mean(axis=1)` -> 3 values |

## Build and Sample Pattern

```python
import stan
import numpy as np

# Build (random_seed goes here, NOT in sample())
posterior = stan.build(stan_code, data=data_dict, random_seed=1)

# Sample (only supported params)
fit = posterior.sample(
    num_chains=4,
    num_samples=500,       # (iter - warmup) / thin
    num_warmup=1000,
    num_thin=2,
    delta=0.93,
    max_depth=14,
    init_radius=0.1
)
```

## Long-Running Sampling: Background Execution

Stan sampling for GP models takes 2-5 minutes -- **longer than typical command timeouts**. Always run in background:

```bash
nohup python3 /app/pystan_analysis.py > /tmp/pystan.log 2>&1 &
echo $!   # capture PID
```

Then poll with **long sleep intervals** to conserve iterations:
```bash
sleep 60 && tail -n 5 /tmp/pystan.log
```

Use `sleep 60` (not `sleep 10/20`) -- sampling takes minutes, not seconds. The model compilation is cached after the first run.

**Cholesky warnings** like `cholesky_decompose: A is not symmetric` are informational NUTS proposal rejections, not failures. Ignore them.

## Output Format

Save as numeric-only CSV (no headers, no index):
```python
np.savetxt("/app/alpha_est.csv", [alpha_mean], fmt="%.18e")
np.savetxt("/app/sigma_est.csv", [sigma_mean], fmt="%.18e")
np.savetxt("/app/rho_est.csv", rho_means, fmt="%.18e")       # one value per line
np.savetxt("/app/beta_est.csv", beta_means, fmt="%.18e")      # one value per line
```

## Verification Checklist (Before Calling task_complete)

1. **File existence:** All 4 CSV files exist at expected paths
2. **File sizes:** Scalar CSVs ~25 bytes, vector CSVs ~75 bytes (NOT 50KB+)
3. **Value counts:** `alpha_est.csv` = 1 value, `rho_est.csv` = P values (e.g. 3), etc.
4. **Values reasonable:** Alpha ~1, sigma ~0.1, rho in [0.1, 5], beta in [-2, 2]
