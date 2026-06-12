## 2026-06-12 13:02 — Technical — full document

**File reviewed:** `paper/main.tex`  
**Annotated file:** `paper/main_annotated.tex` (compiles cleanly, 154 KB)  
**Total issues:** 7 (0 critical, 3 major, 4 minor)

---

### Critical
None.

---

### Major

**1. Inconsistent percentage bases for context-length quality gains (§4.4, ~line 337)**

The text states T5-small F1 "+35%" (from 0.536 to 0.722) and GPT-2-small "+20%" (from 0.087 to 0.288).
The T5 figure is a *relative* increase: (0.722/0.536 − 1) = 34.7% ≈ 35%. ✓
The GPT-2 figure, however, works out to an *absolute* increase in F1 points: 0.288 − 0.087 = 0.201 ≈ 20 pp. As a relative increase it would be +231%.
Using different bases in the same sentence is misleading. Fix: use absolute F1-point gains for both (T5: +18.6 pp; GPT-2: +20.1 pp) or relative gains for both (T5: +35%; GPT-2: +231%).

**2. Misleading numerical comparison in latency analysis (§5.3, ~lines 418–420)**

The text equates GPT-2's "8,192 token-attention operations" with T5's "512 × 16 = 8,192 cross-attention operations", then concludes T5 does this "once in the encoder". Three problems:
- (a) GPT-2's exact cost is $\sum_{t=0}^{15}(512+t) = 8{,}312$, not 8,192 — the numerical equality is approximate.
- (b) In T5, it is the *encoder self-attention* ($O(L^2)$) that runs once; the *decoder cross-attention* still runs $T = 16$ times. The sentence conflates these.
- (c) The comparison omits T5 decoder self-attention ($O(T^2)$) and GPT-2 self-attention over the prompt. A cleaner argument: GPT-2's per-step cost grows as $O(L+t)$ with KV cache, while T5's per-step cost is $O(d_\text{ff})$ + a small $O(L)$ cross-attention term; the FFN dominates at typical $L$.

**3. Batch-size description does not match model configs (§3.3, ~line 203)**

The text states "effective batch size 32 (batch 4, gradient accumulation 8)". However, T5-large used batch=2 with grad\_accum=16 (= effective 32, but a different micro-batch size than stated). If different models used different micro-batch configurations, either state "effective batch 32 across all models (micro-batch varies by GPU memory)" or provide a per-model table.

---

### Minor

**4. FFN cost formula missing factor of 2 (§5.3, ~line 407)**

The cost of a single FFN is stated as $d_\text{model} \times d_\text{ff}$, but the T5 FFN has two matrix multiplications ($d_\text{model} \to d_\text{ff}$ and $d_\text{ff} \to d_\text{model}$), giving $2 \times d_\text{model} \times d_\text{ff}$ per token. The qualitative argument is unaffected, but the formula should include the factor of 2 for precision.

**5. Undefined symbol $d_{kv}$ (§5.3, ~line 409)**

`$L \times d_{kv}$` uses $d_{kv}$ without definition. In T5, cross-attention uses key/value heads of dimension $d_\text{kv} = d_\text{model} / n_\text{heads}$, but this is never stated. Either define the symbol or replace with $d_\text{model}$ (the dominant dimension).

**6. $c_\text{step}$ introduced without definition (§5.3, ~line 413)**

The symbol $c_\text{step}$ appears in $T \times c_\text{step}$ without explanation. Add a parenthetical such as "(per-step compute cost, dominated by FFN)" for clarity.

**7. Val-loss figures rounded to identical values (§4.2, ~line 279)**

XA-only val\_loss = 0.3059 and LoRA val\_loss = 0.3062 both round to 0.306, so "0.306 and 0.306" is technically correct but may confuse readers who expect two different numbers. Consider 3 significant figures (0.306 and 0.306) with a note, or report as ≈ 0.306 for both. Additionally, the causal claim that parameter freezing "prevents mild overfitting" is asserted but not tested — a train-vs-val loss comparison or learning curve would strengthen the argument.

---

### Summary

| Severity | Count | Key concern |
|---|---|---|
| Critical | 0 | — |
| Major | 3 | Inconsistent % bases; misleading latency arithmetic; batch-size mismatch |
| Minor | 4 | FFN factor-of-2; undefined $d_{kv}$; undefined $c_\text{step}$; identical rounded val-losses |

The core results and claims are sound. The most important fix before submission is **issue 1** (the +35%/+20% comparison), which a reviewer will catch immediately. **Issue 2** (the latency arithmetic) should be rewritten to avoid the coincidental numerical equality. **Issue 3** (batch size) is a reproducibility concern.
