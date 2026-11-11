# Bellman's Lost-in-a-Forest Problem: Exact LP Verification & Differential Evolution Engine

A computational geometry and evolutionary search framework for **Bellman's Lost-in-a-Forest Problem** (1955).

Given a known forest shape $F \subset \mathbb{R}^2$, an explorer knows the geometry and dimensions of the forest but is lost at an unknown starting point and unknown orientation. The objective is to determine a planar path $\gamma$ of minimal length $E(\gamma)$ guaranteeing escape from the forest in the worst-case scenario:

$$E(\gamma) = \inf \left\{ L \ge 0 : \forall (R, t) \in \mathcal{G}, \; R \gamma[0, L] + t \not\subset F \right\}$$

where $\mathcal{G}$ represents the group of allowable spatial motions.

---

## ⚠️ Conventions: Proper Rigid Motions $SE(2)$ vs Full Congruence $E(2)$

A critical distinction in the lost-in-a-forest and universal cover literature:
1. **Proper Rigid Motions $SE(2)$ (Orientation-Preserving / Translations + Rotations, Chiral):**
   - The forest $F$ can be translated and rotated, but **not reflected** (no mirror flip).
   - In asymmetric shapes (e.g., 30-60-90 triangle), paths depend on chirality.
   - For example, the 3-piece Z-path with $E = \frac{3\sqrt{2}-\sqrt{6}}{2} \approx 0.896575$ achieves zero margin under $SE(2)$, but under reflection fits inside the triangle with margin $+0.0104$.
2. **Full Euclidean Congruence $E(2)$ (Translations + Rotations + Reflections):**
   - Allows reflections. Classical arc-covering literature (e.g., Wetzel 1973, Wichiramala & Panraksa 2026) operates in this setting, showing that covering a 30-60-90 triangle under full congruence requires $E(\text{hyp}=1) \ge \approx 0.911$.

*All asymmetric geometry results in this repository are explicitly tagged with their motion convention.*

---

## Core Verification Architecture

1. **Exact Containment Test (Convex Polygons):**
   For any path $\gamma$ and orientation $\theta$, maximum clearance (margin) is formulated as a Chebyshev-center linear program (LP). Since the margin function $s(\theta)$ is $\rho$-Lipschitz bounded ($\rho = \max \|p - c\|$), containment is solved **exactly via branch-and-bound**, guaranteeing zero false negatives.
2. **Evolutionary Optimization:**
   Self-adaptive Differential Evolution (`jDE` with `current-to-pbest/1`), stagnation detection, restart perturbations, and multi-resolution ladder refinement (`--ladder 6,10,14...`) to approximate smooth curve limits with Richardson extrapolation.

---

## Literature Benchmarks & Independent Verification

| Forest Shape | Motion Group | Reference / Literature Status | Engine Value $E(n)$ | Notes |
|---|---|---|---|---|
| **Square** | $E(2)$ | $\sqrt{2} \approx 1.414214$ (Bellman 1955) | `1.41422` | Exact diameter bound verified |
| **Equilateral Triangle** | $E(2)$ | $\sqrt{27/28} \approx 0.981981$ (Besicovitch 1965, Coulton–Movshovich 2006) | `0.981982` | Difference: $+1.9 \times 10^{-6}$ (Z-path) |
| **Isosceles Right (45-45-90)** | $E(2)$ | $3/\sqrt{5} \approx 1.341641$ (Movshovich–Wetzel 2003) | `1.341641` | Difference: $+2.5 \times 10^{-7}$ ($\theta = \arctan(3/4)$) |
| **Golden Gnomon (108-36-36)** | $E(2)$ | $E \approx 0.79273$ (Temerev–Doria, July 2026; $C = 1.282676$ for leg=1) | `0.793052` | Within $+0.04\%$ of exact analytical optimum |
| **30-60-90 Triangle** | **$SE(2)$ (Chiral)** | $\frac{3\sqrt{2}-\sqrt{6}}{2} \approx 0.896575$ (Tight for $SE(2)$) | `0.896576` | Fits under reflection ($E(2)$ requires $E \ge 0.911$) |
| **Regular Pentagon** | $E(2)$ | $1.000000$ (Classical diameter) | `1.000000` | Regular polygons ($k \ge 4$) strictly favour diameter |
| **Infinite Strip ($w=1$)** | $E(2)$ | $\zeta \approx 2.278292$ (Zalgaller 1982 broadworm) | `2.309401` ($n=8$, $4/\sqrt{3}$) | Polygonal approximation to smooth broadworm |

---

## The $E_Z \to E_C$ Branch Crossing in Isosceles Triangles

In isosceles triangles parametrized by apex angle $\alpha$:
* For acute and right triangles ($\alpha \le 90^\circ$), piecewise linear **Z-paths** dominate.
* As apex angle $\alpha$ widens, vertical clearance decreases, and the escape path transitions to a **C/U-curve (Gibbs' tunnel / horseshoe arc)**.
* **Literature Baseline:** Temerev & Doria (July 2026) report the analytical Z-to-tunnel branch transition at base angle $\approx 42.29^\circ$ (apex $\alpha_c \approx 95.42^\circ$), and merger with scaled Zalgaller broadworms at base angle $\approx 27.6^\circ$.
* **Optimization Landscape:** Stochastic jDE search tracks the Z-basin smoothly up to $\alpha \approx 97.2^\circ$, where the global C-basin is rapidly captured ($E_C \approx 0.914$), demonstrating a steep branch crossing $E(\alpha) = \min(E_Z(\alpha), E_C(\alpha))$.

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/EmreAlgurr/LostForestProblem.git
cd LostForestProblem

# Requirements
pip install numpy matplotlib shapely

# Run multi-resolution ladder search
py -3.13 main.py --shape square
py -3.13 main.py --shape triangle --ladder 6,10,14
py -3.13 main.py --shape triangle_30_60_90 --ladder 6,10,14
py -3.13 main.py --shape obtuse_120 --ladder 6,10,14
py -3.13 main.py --shape needle_5 --ladder 10,20,30

# Visualize results
py -3.13 main.py --show results/triangle_n14.json
```

---

## Repository Structure

* `lostforest/shapes.py`: Geometric forest definitions, Chebyshev LP margin solver, exact branch-and-bound verification, composite shapes.
* `lostforest/path.py`: Polygonal genome parametrization, prefix clipping, binary search escape distance, simulation.
* `lostforest/evolve.py`: Self-adaptive Differential Evolution (`jDE`), restart heuristics, multiprocessing.
* `lostforest/visualize.py`: 4-panel diagnostic plots (evolved path, critical placement, stochastic walk, convergence curve).
* `results_log.md`: Detailed research log with mathematical derivations, literature comparisons, and conventions.
* `scratch/`: Standalone experiment scripts and parameter sweep drivers.
