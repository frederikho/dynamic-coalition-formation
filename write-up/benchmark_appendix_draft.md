# What farsightedness adds: the RICE coalition results against static benchmarks

*Draft — Markdown for iteration; converts to LaTeX once the section order settles.*

Figures live in `reports/figures/`. All results below use the
`heyen_lehtomaa_2021` effectivity rule — the protocol and approval committees of
[Heyen & Lehtomaa (2021)](https://academic.oup.com/oocc/article/1/1/kgab010/6370712),
whose framework this work extends — together with the
`power_threshold_RICE_by_GDP` scenario (a coalition must hold at least 50.1% of
GDP-weighted power to deploy), majority approval, and δ = 0.99, across 42
three-country RICE payoff tables.

---

## 1. The question, and three ways to answer it

Suppose three countries can each go it alone, pair up, or all cooperate, and
whoever ends up holding enough power decides how much solar radiation
management to deploy. Which arrangement do we expect to see?

Economics offers several answers, and they are not the same answer.

**Internal and external stability** — the workhorse of the international
environmental agreements literature — asks a local question about each possible
arrangement. An arrangement is *internally stable* if no member would do better
by walking out (with the others staying together). It is *externally stable* if
no outsider would do better by joining. An arrangement passing both tests is a
candidate outcome. The test looks one step ahead and no further.

**The γ-core** asks a cooperative question. Could any group of countries, by
banding together and letting everyone else fragment into singletons, make *all*
of its own members strictly better off than they are now? If so, that group
"blocks" the arrangement. Arrangements nobody can block are the prediction. This
considers every possible group at once, but it also looks only one step ahead:
it compares today's payoffs, not where the deviation eventually leads.

**Our farsighted equilibrium** asks a dynamic question. Countries take turns
proposing changes; changes need approval from the countries the rules give a
veto to; and everyone evaluates a proposal not by today's payoff but by the
whole future path it sets off. The prediction is the set of arrangements the
process settles into and never leaves — the *absorbing states*.

## 2. The one idea that makes the comparison legitimate

These look like different kinds of object, but they are closer than they seem.

Internal stability says country *i* stays in arrangement *x* if

> u<sub>i</sub>(stay) ≥ u<sub>i</sub>(leave)

where *u* is the immediate payoff. Our equilibrium's approval condition says a
country consents to a move if

> V<sub>i</sub>(after) > V<sub>i</sub>(before)

where *V* is the long-run value — the discounted stream of payoffs along the
path the move triggers. These are **the same inequality with a different value
object**. And because V = (I − δP)<sup>−1</sup>(1 − δ)u, as the discount factor
δ goes to zero the future stops mattering, V collapses onto u, and the farsighted
conditions become the static ones.

So the comparison in this appendix measures exactly one thing: **the wedge
between u and V** — how often, and how much, anticipating the consequences of a
move reverses the ranking of that move. Everything below is a way of looking at
that wedge.

## 3. How to read the figures

Each figure shows the same transition graph two or three times over, once per
solution concept, so they can be read against each other directly.

Nodes are arrangements: `( )` means everyone alone, `(NDEUSA)` means India and
the USA cooperate while Russia stands alone, `(NDERUSUSA)` is the grand
coalition. Arrows are the probabilities of moving between arrangements in a
period, and are identical in both panels — only the node colouring differs.

**Left panel — the farsighted equilibrium.** Red marks an absorbing
arrangement: once there, the system stays. Grey marks a transient one.

**Middle panel — internal and external stability.** Solid red marks an
arrangement passing both tests. Diagonal stripes mark one test only: leaning one
way for internally stable, the other for externally stable. Grey marks an
arrangement failing both.

**Right panel, where present — the γ-core.** Red marks an arrangement no
coalition can block; grey marks one that can be blocked.

Red therefore means "stable, in this concept's sense" in every panel, and the
question each figure poses is whether the red nodes coincide.

`G` on each node is the `W_SAI` welfare aggregate from the RICE runs — a measure
of deployment, **not** degrees Celsius, despite the visualiser's label, which
needs fixing before publication.

Countries: `USA`, `CHN` China, `NDE` India, `RUS` Russia, `BRA` Brazil.
`burke`, `kalkuhl`, `andreoni` are alternative climate-damage specifications.

---

## 4. Result 0: most of the apparent disagreement was our mistake

Before any substantive finding, a correction that reshapes the rest.

Classical external stability assumes **open membership**: an outsider who wants
to join simply joins, and the incumbents cannot refuse. Our framework never
works that way — every transition needs approval from the countries the
effectivity rule empowers. Comparing against open membership therefore
manufactures disagreements that are about the membership rule, not about
farsightedness.

Recomputing external stability so that accession also requires the incumbents to
gain — the **consent** variant — changes the picture substantially:

| relation of farsighted prediction to static prediction | open membership | consent |
|---|---|---|
| identical | 27 | **29** |
| farsighted picks a subset of the static set | 4 | **10** |
| farsighted admits more than the static set | 5 | **1** |
| no overlap at all | 5 | **1** |
| undefined | 1 | 1 |

Four of five "no overlap" cases and four of five "farsighted admits more" cases
dissolve. **The headline is not that farsightedness contradicts static
stability. It is that farsightedness mostly agrees with static stability, and
where it differs it is usually *sharper* — it selects among arrangements static
analysis cannot choose between.**

Everything below uses the consent variant.

**Figure:** `agree_kalkuhl_usachnnde_2035-2060.png`

This is what agreement looks like, and it is worth having in view before the
disagreements. Both concepts name `(NDEUSA)` and nothing else; the graph shows
every other arrangement draining into it. Cases like this are the majority.

One qualification that recurs below: agreement is weaker than the count makes it
sound. Of the 29 agreeing cases, only a handful have *both* concepts naming a
single arrangement, as here. In the rest, both return a shortlist of two or
three and the shortlists happen to coincide — that is two concepts declining to
choose, not two concepts converging on an answer.

---

## 5. Which deviations count, and what Heyen & Lehtomaa assume

Result 0 fixed *who has to consent* to a move. There is a second and larger
ambiguity underneath it: *which moves exist at all*. It determines several of
the results below, so it is worth settling explicitly rather than by
implementation accident — which is how we first settled it, wrongly.

### The ambiguity

When a country contemplates leaving a coalition, what destination is it allowed
to contemplate?

Take `(NDERUS)` from the table in the next section: India and Russia in a
coalition, the USA alone.

**Narrow reading.** A departing member goes it alone; the rest stay together.
Russia leaving `(NDERUS)` therefore lands in `( )`, where it earns −11.6800 —
exactly what it had. Russia is indifferent, the weak inequality holds, and the
arrangement is internally stable.

**Broad reading.** A departing member may land anywhere one step can take it,
including inside another coalition. Russia leaving India *and joining the USA*
lands in `(RUSUSA)`, where it earns −11.2799. That is a strict gain, and the
arrangement is internally unstable.

The same arrangement, opposite verdicts, from one modelling choice. This is the
same choice that separates the α-, β-, γ- and δ-characteristic functions in
cooperative game theory: not whether a deviation is permitted, but what the
deviator is assumed to face afterwards.

Note also where the disagreement belongs. It is tempting to file Russia's
re-partnering under *external* stability, because it can be described as "Russia
joins the USA's singleton". But the trigger is a member leaving, so it is an
**internal**-stability question with an enlarged destination set. Filing it
under external stability mislabels which condition failed — a trap our own
visualiser fell into, reporting `(NDERUS)` as "internally stable, externally
unstable" when under the broad reading it is internally unstable and under the
narrow reading it is both.

### What Heyen & Lehtomaa assume

The 2021 paper takes the **narrow** reading, and its external stability is the
**consent** variant. This is not stated as a definition, but it is pinned down
by their results.

Their definition of the benchmark (main paper, p. 2):

> "a coalition is stable if it is **internally stable (no member wants to
> leave)** and **externally stable (no outsider wants to join)**. While
> insightful, this approach is static..."

Under weak governance (p. 3):

> "the prediction would be the same also for the more conventional, static
> approach: **any coalition with W as a member would violate the condition of
> internal stability as W always prefers to leave**."

W "leaving" here means W deploying its ideal level alone — leave-to-singleton.

The decisive passage is the power-threshold example (p. 5):

> "Such transitions would be lost under a static approach. In fact, **all of the
> states (WTC), (WT), and (TC) would be equilibrium predictions if the countries
> weren't farsighted**."

Applying our implementation to their own Figure 1B period payoffs reproduces
that list exactly:

| state | internal | external (open) | external (consent) |
|---|---|---|---|
| `( )` | ✓ | ✗ | ✗ |
| `(TC)` | ✓ | ✗ | ✓ |
| `(WC)` | ✗ | ✗ | ✓ |
| `(WT)` | ✓ | ✗ | ✓ |
| `(WTC)` | ✓ | ✓ | ✓ |

Narrow internal plus consent external gives `{(TC), (WT), (WTC)}` — their three
states. Two corollaries follow, neither of them stated in the paper:

1. **Their external stability must be the consent variant.** Open membership
   yields only `{(WTC)}`: both `(WT)` and `(TC)` admit a joiner who would gain
   (C and W respectively), but in each case the incumbents lose and would
   refuse. Their stated result is not reproducible without incumbent consent.
2. **The broad reading contradicts them.** From `(WT)`, country T abandoning W
   to join C gains 1.94 → 14.44. Under the broad reading `(WT)` fails and the
   static prediction collapses to `{(WTC), (TC)}`, making their sentence false.

So the narrow/consent reading is Heyen & Lehtomaa's working definition, and it
is what we use throughout this appendix.

### What this exposes in the original paper

Adopting their definition also makes visible something the 2021 paper does not
address.

Their *dynamic* model places no such restriction on moves. The Supplementary
Material (p. 2) is explicit:

> "we place **no restrictions on what transitions a country can propose**. Any
> player might then suggest other countries to join forces or existing
> coalitions to disintegrate, even if it is not itself a member of those
> coalitions."

So the paper contrasts a **narrow** static benchmark against a **broad** dynamic
model, and attributes the entire difference to farsightedness. Two distinct
things are bundled together:

| step | what changes | effect isolated |
|---|---|---|
| narrow static → broad static | richer deviation set, still myopic | the **move set** |
| broad static → farsighted MPE | same moves, continuation values | **farsightedness** |

But the *proposable* move set and the *effective* one are not the same thing,
and the gap between them is where this argument nearly went wrong.

Under `heyen_lehtomaa_2021` the transition `(NDERUS) → (RUSUSA)` is indeed
permitted — any of the three countries may propose it, and it appears in no list
of forbidden moves. It nevertheless carries probability zero in equilibrium,
because every proposal needs its approval committee and that committee includes
the **abandoned partner**. When Russia proposes leaving India for the USA, India
must consent, and India loses (−192.4006 against −188.2693). India refuses.

So re-partnering requires the consent of precisely the country with a reason to
withhold it, and under unanimity that veto almost always binds. What remains
genuinely unilateral is plain exit to a singleton — the narrow deviation, which
the literature's benchmark already models. The effective move set is therefore
much closer to the narrow one than the quotation above suggests.

Our first guess was that the move set would prove the larger of the two effects,
on the reasoning that a myopic country offered the re-partnering move would
simply take it. The veto is why that reasoning fails: the move is available to
propose and unavailable to execute. In Section 6's headline case, widening the
deviation set changes nothing whatsoever.

We therefore report our results against Heyen & Lehtomaa's definition, for
comparability with the paper. Measuring the split properly — a myopic benchmark
with the same *effective* move set, vetoes included — remains outstanding work,
listed in Section 12.

### Why not simply allow every transition?

The natural question is why the literature does not just permit every move
between every arrangement and call an arrangement stable when no profitable
one-step move exists. Four reasons, in increasing order of seriousness.

**It is not well defined without saying who moves.** Going from `(WT)` to
`(TC)` requires T to leave W *and* C to accept T. That is a move by the pair
{T, C}, not by T alone. A "fully connected" graph is silent on who effects each
transition, and as soon as one specifies it, the approval question of Result 0
returns. Heyen & Lehtomaa's protocol and approval committees are precisely an
answer to this; the fully connected graph is the question, not the answer.

**It still needs a counterfactual.** Even granted that a group can move, one
must say what the countries left behind do — the α/β/γ/δ problem again.
Reachability alone does not determine the deviators' payoff.

**Predictions thin out or vanish.** Every deviation added is another way for an
arrangement to fail. The narrow deviation set is part of why internal/external
stability yields non-empty predictions in cartel models at all; enlarging it
tends toward emptiness, which is the classical core-emptiness problem.

**Myopia stops making sense.** This is the real reason. If every arrangement is
one step from every other, a myopic deviator is assuming that the arrangement it
moves to will sit still — while the model it inhabits says that arrangement is
equally one step from everywhere else. The richer the move set, the less
coherent the myopic evaluation becomes. This is Harsanyi's objection to treating
deviations myopically, and it is what motivated the farsighted concepts
(Chwe's largest consistent set and its successors). **A rich move set does not
call for a myopic stability concept; it calls for a farsighted one** — which is
what this framework is.

The concept the question gestures at does exist, in a disciplined form: it is
the **γ-core**, where any coalition may deviate and the complement fragments
into singletons. We compute it, and Section 10 reports what it gives — a
prediction so permissive that it leaves three of five arrangements open in the
median case. That is the price of the rich move set without foresight.

---

## 6. Result 1: farsightedness breaks ties that static analysis cannot see

**Figure:** `selects_burke_usarusnde_2035-2060.png`

This is the most robust finding in the set, and the most interesting.

Two further tables, `burke_ndeusarus_2060` and `burke_usarusnde_2035-2060`
under the 2035–2060 payoff horizon, reproduce it exactly — same two candidate
arrangements, same selection, headroom 188× and 192×. Their transition graphs
are visually identical to the one below, so only one is shown.

Take `burke_usarusnde_2035-2060`. Static analysis says two arrangements are
stable: `(NDERUS)` (India–Russia) and `(RUSUSA)` (Russia–USA). It cannot choose
between them. Farsightedness picks `(RUSUSA)` and rejects `(NDERUS)`.

The payoffs show why:

| arrangement | NDE | RUS | USA | deployment |
|---|---|---|---|---|
| `( )` everyone alone | −188.2693 | −11.6800 | −20.1407 | 73.53 |
| `(NDERUS)` | −188.2693 | −11.6800 | −20.1407 | 73.53 |
| `(NDEUSA)` | −192.5680 | −12.3580 | −20.9789 | 168.31 |
| `(RUSUSA)` | −192.4006 | **−11.2799** | **−20.0316** | 19.14 |
| `(NDERUSUSA)` | −190.7254 | −11.5543 | −20.1909 | 79.61 |

Look at the first two rows. They are **identical**. India and Russia together
control less than 50.1% of GDP-weighted power, so their coalition cannot deploy
anything, and forming it changes nothing for anybody. It is a treaty with no
content.

Now apply the static test to `(NDERUS)`. Would India do better by walking out?
It would get exactly the same payoff. Would Russia? Exactly the same. Neither
strictly gains from leaving, so the weak inequality holds and the arrangement
counts as internally stable. **Static analysis certifies an empty treaty as
stable, purely because leaving it is not an improvement.**

Farsightedness refuses the tie, and the route it takes is worth following.

Russia cannot simply defect into `(RUSUSA)`. That transition needs India's
consent, and India loses there (−192.4006 against −188.2693), so India vetoes
it. What Russia *can* do without anyone's permission is walk out into `( )` —
where its payoff is unchanged, −11.6800 either way. A myopic Russia is
indifferent and has no reason to move at all.

But `( )` is not where the process stops. From `( )` it moves on to `(RUSUSA)`
two thirds of the time, and there Russia is strictly better off (−11.2799).
Russia therefore leaves an arrangement it is indifferent about, in order to
reach one it is not, two steps later. The figure shows exactly this: `(NDERUS)`
bleeding to `( )` a third of the time, and `( )` feeding `(RUSUSA)` at two
thirds — while static analysis had called `(NDERUS)` a resting point.

Note what this rules out. The tie is not broken by handing countries a richer
set of moves: the direct re-partnering move exists in our model and is vetoed.
It is broken by Russia valuing an indifferent step for where it leads. Foresight
is doing the work, and nothing else is available to do it.

**Why this matters more than it first appears.** These ties are not accidents of
the data. They are structural. Under a power threshold, *every* coalition too
small to deploy is payoff-identical to no coalition at all. All 42 tables contain
such ties. Static stability is systematically indeterminate exactly where the
power threshold bites, and farsightedness is what resolves it — by asking not
"is this arrangement an improvement?" but "where does this arrangement lead?"

This also settles a worry we had earlier. The tiny payoff differences in these
tables looked like a robustness problem. They are not a problem; they are the
subject. The action is precisely at the ties.

A second finding is folded into the same case, and it belongs to the benchmark
rather than to foresight: **the classical concept mis-specifies the outside
option once a power threshold makes small coalitions inert.** Internal stability
asks whether Russia would rather stand alone, gets "no difference", and stops.
The relevant question was never whether standing alone is better, but where
standing alone leads.

The equilibrium here is comfortably determined: the smallest value difference it
depends on is 87 times the verification tolerance (188× and 11× for the other
two tables in this group).

The mechanism is not specific to the `burke` damage specification. Under
`kalkuhl`, three further tables show it — `kalkuhl_usachnnde_2035-2050`,
`_2050-2065`, and `_2035-2060` at the long payoff horizon — with `(CHNNDE)` as
the empty treaty (China and India being unable to deploy) and `(CHNUSA)` as the
selection. Their margins are far tighter (headroom 1.1 to 1.8, barely above the
verification tolerance), so they corroborate the pattern rather than
establishing it.

All of these cases differ from the static prediction by a single arrangement.
The cases where the two concepts diverge *further* — where static analysis
admits three arrangements, or where the two share none at all — are in
Sections 8 and 9.

---

## 7. Result 2: "more cooperation" is not a well-defined direction

**Figure:** `selects_burke_usachnnde_2035-2060B.png`

It is tempting to summarise the last section as "farsightedness restrains the
free driver." It does not. Compare the case below with the one in Section 6:
the *same* structure of disagreement, and opposite economics. The contrast lives
in the numbers rather than in the graphs, which is itself the point — the two
transition graphs look alike.

| case | static admits | farsighted picks | deployment of the pick | change in expected deployment |
|---|---|---|---|---|
| burke_usarusnde_2035-2060 | `(NDERUS)`, `(RUSUSA)` | `(RUSUSA)` | 19.14 — the **lowest** available | **−27.19** |
| burke_usachnnde_2035-2060B | `(CHNNDE)`, `(NDEUSA)` | `(NDEUSA)` | 179.24 — the **highest** available | **+48.04** |

Same mechanism — in both, the rejected arrangement is an empty treaty
payoff-identical to `( )` — and opposite consequences for deployment.
Farsightedness selects the least-deploying arrangement in one case and the
most-deploying in the other. The direction follows the damage parameterisation,
not any general principle.

**This is a genuine economic point, not a measurement artefact.** In the
international environmental agreements literature, abatement is a *free-rider*
good: bigger coalition → more abatement → better outcome, and "coalition size"
is a serviceable proxy for "cooperation." Solar geoengineering is a
*free-driver* good. A larger coalition averages over more heterogeneous ideal
cooling levels, and admitting a reluctant member pulls deployment down. Size,
deployment, and welfare come apart.

The sharpest illustration: the two cases with the largest deployment gaps in the
entire dataset (−54.4 and +2.2) have a difference in coalition size of **exactly
zero**. Both concepts name a two-country coalition; they simply name a different
one. Any summary that ranks outcomes by coalition size would record these as
"no difference."

We therefore report the three gaps — size, deployment, welfare — separately, and
do not collapse them into a "more/less cooperative" verdict.

---

## 8. Result 3: deviations that get undone

**Figure:** `disjoint_kalkuhl_usachnnde_2035-2100.png`

This is the mirror image of the 2021 paper's headline, and the mechanism is
worth spelling out even though — see the caveat — the specific case is fragile.

Static analysis says the grand coalition `(CHNNDEUSA)` is **unstable**: China
does strictly better by walking out into `(NDEUSA)`, by 1.27 × 10⁻⁴. Static
analysis therefore predicts `(NDEUSA)`. The farsighted equilibrium predicts the
grand coalition — and it is an exceptionally strong attractor, with every other
arrangement flowing into it, in one step from most of them.

Why does China stay? Because leaving does not last. From `(NDEUSA)`, the process
returns to the grand coalition 83% of the time directly, and the remaining 17%
goes to `( )`, from which it returns with certainty. China's exit is *always*
undone. Its gain is one period of advantage against a permanent return to where
it started, and in value terms that gain shrinks by a factor of nearly a
thousand:

| country | immediate gain from China's exit | gain in long-run value | shrinkage |
|---|---|---|---|
| CHN | +1.273 × 10⁻⁴ | +1.294 × 10⁻⁷ | **984×** |
| NDE | +1.484 × 10⁻³ | +7.214 × 10⁻⁶ | 206× |
| USA | −1.188 × 10⁻⁴ | −1.402 × 10⁻⁶ | 85× |

This is the mechanism from the 2021 paper's fourth (supplementary) scenario,
appearing here in RICE data: under majority approval, a country contemplating
departure realises the remaining members can readmit it, or reassemble without
its consent, so the deviation buys nothing durable. Note that this configuration
does use majority rather than unanimity approval.

And the arrangement farsightedness sustains **deploys less** than the one static
analysis predicts (0.84 against 0.98). Full cooperation means restraint. That is
the free-driver signature again, and it runs opposite to the intuition imported
from abatement games.

> **Caveat — this case is tolerance-sensitive.** The equilibrium was verified at
> a tolerance of 10⁻⁵, but the smallest value difference its strategies rely on
> is 8.4 × 10⁻⁸ — about 120 times *smaller* than the tolerance. That means
> verification treated a real difference as indifference. At a tighter tolerance
> this profile would be rejected and the absorbing set could differ. The
> mechanism above is real and the 984× shrinkage is a genuine calculation, but
> this case should illustrate the mechanism, not support an empirical claim about
> China, India and the USA.

---

## 9. Result 4: ...and the same logic can dissolve a grand coalition

**Figure:** `grand_kalkuhl_usarusnde_2035-2060.png`

The closest RICE analogues of the 2021 paper's *main* result. Here static
analysis accepts the grand coalition and farsightedness rejects it:

| case | static admits | farsighted picks | change in deployment |
|---|---|---|---|
| kalkuhl_usarusnde_2035-2060 | `(NDERUS)`, `(NDEUSA)`, `(NDERUSUSA)` | `(NDEUSA)` | +2.90 |
| kalkuhl_usachnrus_2035-2100 | `(CHNRUS)`, `(CHNUSA)`, `(CHNRUSUSA)` | `(CHNUSA)` | +0.04 |

In the first figure the grand coalition is visibly leaking — a third of the time
to `(NDEUSA)`, a third to `(NDERUS)` — and everything funnels into `(NDEUSA)`.
Note the direction: **disintegration raises deployment** (15.59 against 14.63).
The grand coalition was restraining the free driver; breaking it up releases it.

The second case, `kalkuhl_usachnrus_2035-2100`, is the same picture with China
and Russia in place of India and Russia; its transition graph adds nothing to
the one below and is omitted.

Both cases are also tolerance-sensitive (headroom 0.63 and 0.017) and rest on
very small deciding margins — in the first, Russia's choice turns on −12.509
versus −12.508. As above: good for illustrating the mechanism, not for an
empirical claim.

That we see the 2021 paper's mechanism running in *both* directions across
different damage specifications — dissolving grand coalitions here, sustaining
one in Result 3 — is itself the point. Farsightedness does not push toward or
away from cooperation. It enforces consistency between what countries expect and
what actually happens.

---

## 10. Result 5: the farsighted prediction never leaves the γ-core

Across all 41 tables where the comparison is defined, the farsighted absorbing
set is **always a subset of the γ-core's unblocked set** — equal in 17 cases, a
strict subset in 24. It is never disjoint, never partially overlapping, never
larger.

In plain terms: the farsighted equilibrium never settles on an arrangement that
a coalition could profitably overturn in the cooperative sense, and it is
strictly more decisive than the γ-core, which typically leaves three of the five
arrangements open (range 1–5).

**This is a refinement result**, and it is the cleanest statement available from
this dataset. But it is an *empirical* regularity over 41 tables, not a theorem,
and we should not assert otherwise. γ-blocking uses immediate payoffs and lets
any coalition deviate; our equilibrium uses long-run values and only permits
transitions the effectivity rule sanctions and the approval committee accepts. A
γ-blocked arrangement could in principle be absorbing — if the blocking
coalition's move requires the consent of an outsider who refuses. It does not
happen here. Whether that is forced by the `heyen_lehtomaa_2021` rule or is a
property of these particular tables is an open question, and worth its own
section if it holds up.

**Figure:** `gammacore_kalkuhl_usachnnde_2035-2060_p2300.png`

The figure shows all three concepts on one game. The γ-core leaves `(CHNUSA)`,
`(NDEUSA)` and `(CHNNDEUSA)` open — three of the five arrangements, and it
cannot choose between them. Internal/external stability names `(CHNNDE)` and
`(CHNUSA)`, a different pair, and one of its two is an arrangement the γ-core
rules out. The farsighted equilibrium names `(CHNUSA)` alone — the single
arrangement lying inside both static predictions, and the one the process
actually reaches, in one step from everywhere.

That is the nesting result in one picture: red shrinks from left to right in the
γ-core panel to a single node under foresight, and never moves outside. The
case's headroom is 1.8, so it is not tolerance-sensitive.

Two further γ-core observations: the grand coalition is unblocked in only 8 of
42 cases; and in 4 cases the *transferable-utility* core is non-empty while the
no-transfer core blocks the grand coalition — gains from full cooperation that
exist but cannot be realised without side payments, which this framework
excludes by assumption.

---

## 11. Two warnings from the data

### The empty game

**Figure:** `frozen_burke_usarusnde_2060-2080_p2060.png`

Five isolated nodes, each a 100% self-loop, no arrows between them. Nothing ever
happens.

The reason is in the payoff table: all five arrangements have **identical**
payoffs, differing only at machine precision (relative spread 1.7 × 10⁻¹⁴), with
deployment identically zero. The payoff horizon for this table is the single
year 2060, before deployment ramps up, so the coalition structure has no
consequences at all. Every country is indifferent between everything, therefore
*every* strategy profile is an equilibrium, and the solver returned the one where
nobody moves.

This is a degenerate **input**, not a degenerate equilibrium, and it is a
warning about single-year payoff horizons. It is the only such table here — the
next-narrowest has a relative spread of 3.7 × 10⁻⁵ against a median of
2.3 × 10⁻² — and the comparison marks it undefined rather than scoring it.

### Tolerance sensitivity, and how we measure it

Every equilibrium here was verified at a numerical tolerance of 10⁻⁵: value
differences smaller than that were treated as indifference, which the model
permits (an indifferent country may approve or refuse freely). That is a
reasonable convention, but it has a consequence worth stating plainly.

An equilibrium's strategies rest on the *sign* of certain value differences —
wherever a country plays a pure "always approve" or "always refuse", the
equilibrium is asserting that V(after) is definitely above or below V(before).
We record the smallest such difference the profile relies on, and call its ratio
to the verification tolerance the **headroom**:

> headroom = (smallest decisive value difference) / (verification tolerance)

Headroom above 1 means every distinction the equilibrium depends on is larger
than the tolerance that certified it. Headroom below 1 means at least one real
strict preference was treated as indifference during verification — and a
tighter tolerance would have rejected the profile, possibly yielding a different
absorbing set.

Two measurement details matter. First, differences at the level of floating-point
residue (below 10⁻¹² of the value scale) are excluded: those are states whose
values are *mathematically* equal, where indifference is genuine and nothing
rests on the sign. Including them makes almost every table look fragile, which
was our first, wrong, reading of this diagnostic. Second, headroom is the
dynamic counterpart of the static margin `ies_min_margin`, which measures the
same fragility on one-period payoffs; the two can differ, and a case can be
solid on one and fragile on the other.

Nine of 42 cases have headroom below 1:

| table | relation | smallest decisive gap | headroom |
|---|---|---|---|
| burke_usarusnde_2080-2100 | agree | 1.5 × 10⁻¹⁰ | 0.000015 |
| andreoni_rusndechn_2035-2060 | foresight admits more | 7.4 × 10⁻¹⁰ | 0.000074 |
| burke_usachnnde_2060-2080 (sai=0.01) | agree | 1.6 × 10⁻⁸ | 0.0016 |
| burke_usachnnde_2080-2100 (sai=0.01) | agree | 4.1 × 10⁻⁸ | 0.0041 |
| kalkuhl_usachnnde_2035-2100 | no overlap | 8.4 × 10⁻⁸ | 0.0084 |
| kalkuhl_usachnbra_2035-2100 | agree | 1.0 × 10⁻⁷ | 0.0101 |
| kalkuhl_usachnrus_2035-2100 | foresight sharper | 1.7 × 10⁻⁷ | 0.0170 |
| kalkuhl_usachnnde_2035-2060 | agree | 5.0 × 10⁻⁶ | 0.4989 |
| kalkuhl_usarusnde_2035-2060 | foresight sharper | 6.3 × 10⁻⁶ | 0.6328 |

The pattern tracks the damage specification rather than anything about
coalitions: 5 of 9 `kalkuhl` tables are fragile against 3 of 28 `burke` and 1 of
5 `andreoni`. The `kalkuhl` specification at century-long horizons separates the
five arrangements by so little that the equilibrium conditions are close to
vacuous. Whether that reflects genuinely flat welfare differences or limited
numerical resolution in those RICE runs is not something this analysis can
settle, and it should be checked against the runs themselves.

By contrast the cases carrying Result 1 have ample headroom — 87×, 188× and 11×.

**The practical rule for this appendix: cases with large headroom carry
empirical claims; cases with small headroom illustrate mechanisms.** Both are
worth showing, but they should not be asked to do the same work.

### The uncomfortable pattern: divergence and robustness trade off

Ranking every case by how far the two predicted sets actually diverge — counting
arrangements named by one concept and not the other — produces an almost perfect
inverse ordering against headroom:

| case | static admits | foresight picks | arrangements differing | headroom |
|---|---|---|---|---|
| kalkuhl_usachnnde_2035-2100 | `(NDEUSA)` | `(CHNNDEUSA)` | **2** (nothing shared) | 0.008 |
| kalkuhl_usachnrus_2035-2100 | three | `(CHNUSA)` | 2 | 0.017 |
| kalkuhl_usarusnde_2035-2060 | three | `(NDEUSA)` | 2 | 0.63 |
| burke_usachnnde_2035-2060B | two | `(NDEUSA)` | 1 | 11 |
| burke_usarusnde_2035-2060 | two | `(RUSUSA)` | 1 | 87 |
| burke_ndeusarus_2060 | two | `(RUSUSA)` | 1 | 188 |

**Every case diverging by more than one arrangement is tolerance-sensitive, and
every robustly determined case diverges by exactly one.** The figures that look
most striking are the ones resting on the least, and the cases we can stand
behind are the modest ones.

We do not think this is a coincidence of sampling. Large divergence requires the
long-run values to reorder arrangements substantially, and the tables where that
happens are the ones whose period payoffs barely separate the arrangements in
the first place — small denominators make for large reorderings. If so, the
trade-off is structural rather than bad luck, and it limits how strong a claim
this comparison can support on RICE data of this resolution. Testing that
explanation means checking whether the same tables' payoff spreads predict their
headroom, which we have not done.

---

## 12. What we cannot yet claim

1. **The move set and foresight are not yet separated.** Section 5 shows the
   reported gaps bundle two effects. Adding broad internal stability as a third
   column — same deviation set as the dynamic model, still myopic — would split
   them, and would tell us whether the striking cases are about foresight or
   about the classical benchmark's deviation set. Our expectation, from Section
   6, is that the second is the larger effect, but that is untested.
2. **We have not run the δ → 0 check.** Since static stability is the δ → 0
   limit of our conditions, re-solving at δ ≈ 0.1 should collapse the absorbing
   sets onto the statically stable sets. This would both validate the pipeline
   end-to-end and quantify how much work farsightedness is doing. It is the
   single most valuable outstanding test.
3. **The rules of the game never changed the answer.** Across all 42 tables,
   swapping `heyen_lehtomaa_2021` for `adjacent_step` never altered the
   predicted arrangement. Since institutional flexibility is a central selling
   point of the framework, this needs explaining: either the two rules are more
   alike than intended, or three countries is too small a stage for the protocol
   to matter.
4. **One equilibrium per table.** Each row reflects the equilibrium the solver
   converged to. Nothing rules out others, and where multiple equilibria exist
   the "farsighted prediction" is really "this equilibrium's prediction."
5. **Agreement is weaker than the count suggests.** Of the 29 cases where the
   two concepts agree, only a handful have both naming a *single* arrangement.
   The rest are coinciding shortlists — both concepts declining to choose, and
   happening to decline the same way.
