# Indirect disclosure: the rank/nlargest lesson

**A verb can be disclosive even when it never returns a value and every released
number passes `min_n`.** This is the most important — and least obvious —
security property in safepy, so it gets its own note.

## The rank case

`rank` returns a *private* column (a position, not a value), and every exit is
`min_n`-suppressed. By the surface reading of the compute-private principle it
looks safe. It is not:

```python
df.assign(rk=df.salary.rank())[df.rk <= 6].salary.sum()   # sum of the 6 smallest
df.assign(rk=df.salary.rank())[df.rk <= 5].salary.sum()   # sum of the 5 smallest
# difference = the 6th-smallest salary, exactly. Both queries have n>=min_n.
```

The same shape with `nlargest(k)` / `head(k)` / `nth`:

```python
df.salary.nlargest(10).sum() - df.salary.nlargest(9).sum()   # = the 10th value
```

`min_n` guarantees "≥ k people contributed." It says **nothing** about *which*
people, or that two aggregates were composed to differ by exactly one
value-ordered individual. That is **indirect disclosure by differencing**, and
the per-query exit check cannot see it.

## The rule this gives us

When judging any new verb, the question is not "does it return a value" but:

> Does it let the analyst compose an aggregate over an **attacker-chosen,
> value-ordered subset** whose size they control by one?

- **Population-preserving** ops (reshape, recode, `map`, `agg`, `transform`,
  `shift`, `cumsum`) keep the whole group in every aggregate — no one-row knob.
  Safe by construction.
- **Value-ordered selection** (`rank`, `nlargest`, `nsmallest`, `head`, `tail`,
  `nth`, `first`, `last`, `mode`, `idxmax`/`idxmin`) hands out that knob. **Deny
  them**, even when they look private, until the audit layer exists.

## Why this is only *contained*, not *solved*

The differencing primitive is already latent in primitives we must keep:

```python
df[df.salary <= v].salary.sum()   # scan v -> order statistics, no rank needed
```

So denying `rank`/`nlargest` is **defense-in-depth that removes the ergonomic,
turnkey forms** — it does not close the underlying gap. The real fix is a
**multi-query audit / budget layer** that tracks how released aggregates relate
across a session (two sums differing by one controllable row → refuse or charge
budget). That layer is the linchpin of safepy's security story; see
[further-work.md](further-work.md). Until it lands:

1. Default-deny anything that selects a value-ordered subset.
2. Treat "returns a private column" as **necessary but not sufficient** for
   safety — always ask the differencing question above.
3. Every new release path must suppress on its *own* contributing count, and we
   accept that cross-query differencing remains an open, tracked risk.
