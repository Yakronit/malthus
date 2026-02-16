# malthus

A simulation showing how connectivity influences the demographic composition of a population which has two families type: low fertitlity and high fertility, under the assumption that low fertility is contegeous.
Main result: If low fertility families are below the replacement rate and high fertility are above the replacement rate then in certain conditions a stable population is possible and Malthus curse is avoided.

Here is gpt summary of the model:

# Branching–Infection Dynamics on a Mixing Graph

This project models two node types (A and B) evolving on a dynamic graph. The dynamics combine:
- type-dependent reproduction (branching),
- irreversible infection (B -> A),
- locality defined via random-walk mixing,
- graph growth that preserves local neighborhoods.

The goal is to understand when type A dominates, when type B dominates, and when the outcome depends on initial conditions.

---

## State

At each generation t:
- Graph: G_t = (V_t, E_t)
- Node type function: tau_t(v) in {A, B}

---

## Mixing / Locality Parameter (k)

Locality is defined using random walks.

For a given walk length L:
- start at node x
- run an L-step simple random walk
- let y be the endpoint

We choose L so that the expected shortest-path distance from x to y is about k:

  E[ dist(x, y) ] ≈ k

Interpretation:
- small k => infection is very local
- large k => infection is more global

(Note: in simulation, L is calibrated numerically so this condition holds approximately.)

---

## Per-Generation Dynamics

Each generation consists of two stages:

### 1) Infection (irreversible)

Each A-type node performs m infection attempts.

For each attempt:
- sample a target node using the random-walk endpoint distribution (defined by L, hence by k)
- convert the target to type A

Rules:
- A stays A
- B becomes A
- infection is irreversible

### 2) Reproduction / Graph Expansion

Every node reproduces into same-type offspring:
- each A node creates n A-offspring
- each B node creates N B-offspring

Offspring inherit the parent’s type.

Edges are “lifted” locally so that offspring of neighboring parents tend to be connected, preserving locality across generations (descendants have similar random-walk neighborhoods to their parents).

---

## Competing Forces

Two mechanisms compete:

1) Branching advantage:
- If N > n, B reproduces faster than A.

2) Infection advantage:
- A converts B to A.
- Strength depends on:
  - m (number of infection attempts per A node)
  - k (how local/global infection targets are)

---

## Rough Invasion Threshold (mean-field intuition)

When A is rare and infection targets are approximately “well mixed”, a rough condition for A to invade is:

  n * (1 + m) > N

Example: n = 2, N = 8

  2 * (1 + m) > 8
  1 + m > 4
  m > 3

So you expect:
- m <= 3: B often dominates (especially when k is small / infection is local)
- m >= 4: A can dominate (more easily as k increases)

The exact boundary shifts with k because local infection can “waste” attempts by mostly hitting existing A-clusters.

---

## Phase Behavior for n=2, N=8

Empirically, sweeping parameters (k, m) typically shows three regimes:

- B-dominant region: small m, small k
- A-dominant region: large m and/or large k
- transition region: near the boundary; outcomes can be sensitive to initial clustering

---

## Key Takeaway

Yes: there are parameter settings where A dominates and others where B dominates.
For n=2, N=8, the critical infection level is roughly around m ≈ 4 under good mixing,
with the required m increasing when infection is more local (smaller k).
