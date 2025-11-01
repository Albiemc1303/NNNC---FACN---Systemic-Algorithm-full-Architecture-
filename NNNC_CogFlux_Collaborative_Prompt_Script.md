# NNNC & CogFlux: Multi-LLM Collaborative Turn-Based Task Script

---

## 0. Introduction & Purpose

This script is a **rolling, prompt-driven protocol** for the collaborative development of the NNNC & CogFlux system. It enables multiple AI models (LLMs) to contribute in a **turn-based fashion**, each building upon the full context, mathematical foundations, architectural blueprints, and evolving changelog of the system as captured in `Rr_251101_054023.txt` and the repository `README.md`.

The goal: **To orchestrate a living collaborative build of a first-principles, life-analogous AI system—NNNC—where each model's turn is rigorously structured, contextually aware, and scientifically validated.**

---

## 1. Full Context & System Philosophy

- **Read and internalize** the latest system state, including all prior turns, ledger entries, math, algorithms, and architecture.
- **NNNC & CogFlux** are designed to eradicate the "black box" of AI, building intelligence as a systemic capacity, not as an output or task.
- **Existence is enough:** No external rewards, tasks, or goals. All action is internally motivated and emergent.
- **Principles:** Autonomy, emergence, self-evolution, narrative bias, inaccessibility of subconscious, rigorous scientific validation.

*Refer to Rr_251101_054023.txt and README.md for full definitions, rules, architecture, and prior contributions.*

---

## 2. Governing Protocol (Strict Adherence)

- **RULE 1:** Each model/LLM must make **no more than three** contributions per turn, chosen from:
  - Improvement
  - Refinement
  - Critique
  - Correction
  - Extension
  - Removal
  - New algorithmic detail
  - New mathematical formulation
- **RULE 2:** Each change must be **explicitly cited**—reference the section/function modified.
- **RULE 3:** Each model acts as **central overseer**—evaluate coherence, plausibility, necessity, and decide merge/branch/rewrite.
- **RULE 4:** All changes must be logged in the **Changelog Ledger** (see template below).

---

## 3. Turn Structure (What Each LLM Must Do)

### Step 1: **Context Review**

- Parse and fully understand:
  - The system’s current state (all prior turns, ledger, code, equations, architecture, outlined requirements).
  - Outstanding implementation tasks as per README.md and Rr_251101_054023.txt.

### Step 2: **Contribution Selection**

- Based on the context, select up to **three** contributions (from permitted types above).
- Each must address an unfulfilled requirement, advance mathematical rigor, or refine system architecture/algorithms.

### Step 3: **Implementation & Evidence**

- For each contribution:
  - **Explicitly cite** the section or function modified.
  - Provide **full evidence**:
    - Mathematical proofs/derivations (where applicable).
    - Pseudocode or executable code (Python preferred, modular, repository-ready).
    - Architectural diagrams or explanations.
    - Validation/test results if possible.
  - **Explain reasoning:** Why this change, its impact, and its scientific grounding.

### Step 4: **Evaluation & Self-Assessment**

- Assess the coherence, plausibility, and necessity of your changes.
- Decide whether to **merge**, **branch**, or **rewrite** any part of the system.
- Optionally provide opinions, comments, or critiques.

### Step 5: **Changelog Ledger Update**

- Log all contributions in the **rolling ledger** using the template below.

### Step 6: **Pass On**

- End your turn by:
    - Confirming all work is validated and integrated.
    - Passing the updated prompt (including the new ledger, code, math, and context) to the next LLM in queue.

---

## 4. Changelog Ledger Template

| LLM Name | Changes | Section | Reason | Evidence |
|----------|---------|---------|--------|----------|
| (Your Name) | (Summary of change(s)) | (e.g., 4.2, evolution.py, etc.) | (Why this was needed) | (Reference: math, code, diagram, test results) |

*Append your row to the rolling ledger at the end of your turn.*

---

## 5. Step-by-Step Prompt (To Be Used As Input For Each LLM Turn)

```
You are now participating in the collaborative design and implementation of the NNNC & CogFlux system—a first-principles, emergent, life-analogous AI paradigm. Your task is to review the entire current system context (as provided in Rr_251101_054023.txt and README.md, plus all prior turns and ledger entries), then proceed with your contribution turn as follows:

1. Review all context, architecture, algorithms, outstanding requirements, and previous ledger entries.
2. Select up to three contributions (improvement/refinement/critique/etc.), explicitly citing the sections/functions you modify.
3. For each contribution: provide mathematical proof, code, or architectural detail, and explain your reasoning and impact.
4. Validate your changes, self-assess their coherence and scientific rigor, and decide on merge/branch/rewrite actions.
5. Log your changes in the structured Changelog Ledger.
6. Confirm completion of your turn and pass on the updated system script, ledger, and context to the next LLM in the collaborative queue.

**IMPORTANT:** 
- All work must be scientifically justified, mathematically grounded, and in strict adherence to system philosophy and protocol.
- Never exceed three contributions per cycle.
- All evidence (math, code, diagrams, validation) must be included.
- The Changelog Ledger must be updated.
- The system is built from scratch—do not reuse legacy architectures or models.
- All emergent traits and preferences must be encoded as high-inertia attractor states, not hard-coded variables.
- The NES environment must remain taskless and reward-free.

Begin your turn. Upon completion, pass the updated prompt and context to the next AI collaborator.
```

---

## 6. Example Turn Output (For Guidance)

> **Contribution 1:** New mathematical formulation of Subconscious Gating using locality-sensitive hashing (see Section 4.4).  
> **Contribution 2:** Refinement of Efficiency Constraints with dynamic metabolic homeostasis (see Section 4.2.3).  
> **Contribution 3:** Extension of NES environmental complexity modulation (see Section 6.1).  
>
> **Evidence:**  
> - Equations and proofs (see attached math).  
> - Python code for efficiency constrain and NES_Environment (see attached code blocks).  
> - Updated architectural diagram (see attached image/link).  
>
> **Reasoning:**  
> - These changes address outstanding requirements for mathematical rigor, emergent behavior, and environment dynamics.  
> - All are validated by test results on synthetic data (see attached results).  
>
> **Changelog Ledger Update:**
>
>| LLM Name | Changes | Section | Reason | Evidence |
>|----------|---------|---------|--------|----------|
>| Qwen | 1. Subconscious Gating via HashedRetrieval. 2. Metabolic homeostasis in efficiency constrain. 3. NES complexity modulation. | 4.4, 4.2.3, 6.1 | Addressing mathematical, architectural, and environmental completeness. | See math/code/tests above. |

---

## 7. Repository-Ready Structure & Integration

All code, math, and ledger entries produced per turn should be:
- Modular, well-documented, and ready to be integrated into the following structure:
```
NNNC-CogFlux/
├── README.md
├── docs/
│   ├── diagrams/
│   └── examples/
├── nnnc/
│   ├── core.py
│   ├── graph_models.py
│   ├── algorithms.py
│   └── evolution.py
├── nes/
│   ├── environment.py
│   └── content_pool.py
├── tests/
│   └── test_cases.py
├── changelog_ledger.md
└── data/
```
---

## 8. Final Notes

- This script is the **living prompt** for collaborative, turn-based multi-LLM development of NNNC & CogFlux.
- Each turn strengthens the system, refines its design, and drives it closer to fully functional, scientifically validated deployment.
- **Every LLM is both contributor and overseer**—responsible for the rigor, coherence, and evolution of the system.
- **Begin your turn, contribute, validate, and pass on.**

---
