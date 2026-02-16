# malthus

A simulation showing how connectivity influences the demographic composition of a population which has two families type: low fertitlity and high fertility, under the assumption that low fertility is contegeous.
Main result: If low fertility families are below the replacement rate and high fertility are above the replacement rate then in certain conditions a stable population is possible and Malthus curse is avoided.

Here is gpt summary of the model:


Branching–Infection Dynamics on a Mixing Graph

This project studies a two-type evolutionary process on a dynamically expanding graph with random-walk–defined locality.

The model combines:

Type-dependent reproduction

Irreversible infection

Locality defined by random-walk mixing

Graph growth that preserves neighborhood structure

The goal is to understand under which parameters type A dominates, type B dominates, or both coexist.

Model Overview
Graph Structure

At each generation 
t
t:

We have a graph 
Gt=(Vt,Et)
G
t
	​

=(V
t
	​

,E
t
	​

).

Each node has type 
τt(v)∈{a,b}
τ
t
	​

(v)∈{a,b}.

Mixing Property (parameter 
k
k)

Locality is defined via random walks.

Start at node 
x
x.

Run an 
L
L-step simple random walk.

Let 
Y
Y be the endpoint.

Choose 
L
L so that

E[d(x,Y)]≈k
E[d(x,Y)]≈k

where 
d(⋅,⋅)
d(⋅,⋅) is shortest-path distance.

Small 
k
k → infection is local.

Large 
k
k → infection is more global.

Dynamics Per Generation

Each generation consists of two stages:

1. Infection (Irreversible)

Each node of type A performs 
m
m infection attempts.

For each attempt:

Sample a node via the random-walk endpoint distribution.

Convert that node to type A.

Properties:

Infection is irreversible (B → A only).

Infection locality depends on 
k
k.

2. Reproduction

Each node reproduces into same-type offspring:

Type A produces 
n
n offspring.

Type B produces 
N
N offspring.

Offspring inherit their parent’s type.

Graph edges are lifted locally so that descendants of neighboring parents tend to remain neighbors. This preserves locality structure across generations.

Competing Forces

Two mechanisms compete:

Branching Advantage

If 
N>n
N>n, type B multiplies faster.

Infection Advantage

Type A converts B-nodes at rate controlled by:

m
m (infection attempts per A node)

k
k (how globally infection spreads)

Key Insight: Invasion Condition

When A is rare, a rough mean-field invasion criterion is:

n(1+m)≳N
n(1+m)≳N

For example, with:

n=2,N=8
n=2,N=8

we obtain:

2(1+m)≳8⇒m≳3
2(1+m)≳8⇒m≳3

Interpretation:

If 
m≤3
m≤3: B typically dominates.

If 
m≥4
m≥4: A can dominate (especially if mixing is sufficiently global).

This boundary shifts with 
k
k:

Larger 
k
k → more global infection → easier for A to win.

Smaller 
k
k → more local infection → harder for A to overcome B’s branching advantage.

Phase Behavior (for 
n=2,N=8
n=2,N=8)

Empirically, the system exhibits three regimes:

1. B Dominance

Small 
m
m

Small 
k
k

Infection too weak to offset branching disadvantage

2. A Dominance

Large 
m
m

Or sufficiently large 
k
k

Infection overcomes branching disadvantage

3. Transitional / Mixed Regime

Near the critical boundary

Long transient coexistence

Sensitive to initial clustering

Why This Model Is Interesting

This process couples:

Branching processes

Contagion dynamics

Random-walk geometry

Expanding small-world networks

It creates a nontrivial phase diagram in parameter space:

(m,k,n,N)
(m,k,n,N)

Even with irreversible infection, type A does not always win — dominance depends on whether infection compensates for reproductive disadvantage.

Conceptual Interpretation

The model can be viewed as:

Evolutionary advantage vs. cultural contagion

Minority ideology spreading through social mixing

Innovation diffusion under demographic disadvantage

Competing growth vs. conversion dynamics

Core Takeaway

Yes — there are parameter regimes where:

Type A dominates

Type B dominates

Or both coexist for long periods

For 
n=2,N=8
n=2,N=8, the rough critical infection level is:

m≈4
m≈4

with the exact boundary depending on mixing 
k
k.

The model exhibits a genuine phase transition driven by the interaction of reproduction, infection, and network mixing.
