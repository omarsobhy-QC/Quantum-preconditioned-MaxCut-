/-
RigettiB_Verifier.lean
Lean 4 / Mathlib verifier for weighted Max-Cut (MPES).

Model:
  - n nodes
  - edges are (u,v,w) with w : ℚ
  - partition p : Fin n → Bool
  - cutWeight p edges : ℚ
  - totalWeight edges : ℚ
  - mpes p edges : ℚ (as a rational ratio)

Certificate:
  - claimCut : ℚ
  - certOK p edges claimCut : Prop := cutWeight p edges = claimCut
  - verify theorem: certOK → cutWeight = claimCut

This is a formal verifier, suitable for embedding in a competition repo.
-/

import Mathlib.Data.Rat.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Tactic

open scoped BigOperators

namespace RigettiB

/-- Weighted undirected edge on `n` nodes. -/
structure Edge (n : Nat) where
  u : Fin n
  v : Fin n
  w : ℚ
deriving DecidableEq, Repr

/-- Total graph weight (sum of all edge weights). -/
def totalWeight {n : Nat} (edges : List (Edge n)) : ℚ :=
  (edges.foldl (fun acc e => acc + e.w) 0)

/-- Boolean XOR as inequality on Bool. -/
@[simp] def bxor (a b : Bool) : Bool :=
  (a && (!b)) || ((!a) && b)

/-- Edge is cut by partition `p` iff endpoints differ. -/
def edgeCut {n : Nat} (p : Fin n → Bool) (e : Edge n) : Bool :=
  bxor (p e.u) (p e.v)

/-- Cut weight (sum weights of cut edges). -/
def cutWeight {n : Nat} (p : Fin n → Bool) (edges : List (Edge n)) : ℚ :=
  edges.foldl
    (fun acc e => if edgeCut p e then acc + e.w else acc)
    0

/-- MPES ratio as a rational number. Defined as `cutWeight / totalWeight`. -/
def mpes {n : Nat} (p : Fin n → Bool) (edges : List (Edge n)) : ℚ :=
  cutWeight p edges / totalWeight edges

/-- A partition represented as a Vector Bool n. -/
abbrev PartVec (n : Nat) := Vector Bool n

/-- Convert vector partition into a function `Fin n → Bool`. -/
def PartVec.toFun {n : Nat} (pv : PartVec n) : Fin n → Bool :=
  fun i => pv.get i

@[simp] lemma totalWeight_nil {n} : totalWeight (n:=n) [] = 0 := by
  rfl

@[simp] lemma cutWeight_nil {n} (p : Fin n → Bool) : cutWeight p (n:=n) [] = 0 := by
  rfl

/-- If all edge weights are nonnegative then cutWeight is nonnegative. -/
theorem cutWeight_nonneg_of_weights_nonneg
    {n : Nat} (p : Fin n → Bool) (edges : List (Edge n))
    (h : ∀ e ∈ edges, 0 ≤ e.w) :
    0 ≤ cutWeight p edges := by
  -- fold-based monotonicity
  unfold cutWeight
  -- We prove by induction on edges list.
  induction edges with
  | nil =>
      simp
  | cons e es ih =>
      simp at h
      have h_es : ∀ e' ∈ es, 0 ≤ e'.w := by
        intro e' he'
        exact h.2 e' he'
      specialize ih h_es
      -- Expand foldl one step:
      simp [List.foldl] at *
      -- Case split on whether edge is cut
      by_cases hc : edgeCut p e
      · simp [hc]
        have hw : 0 ≤ e.w := h.1
        linarith
      · simp [hc]
        exact ih

/-- Certificate predicate: claimed cut equals computed cut. -/
def certOK {n : Nat} (p : Fin n → Bool) (edges : List (Edge n)) (claimCut : ℚ) : Prop :=
  cutWeight p edges = claimCut

/-- Certificate verification: if `certOK` holds, the cutWeight equals the claim. -/
theorem verify_cert
    {n : Nat} (p : Fin n → Bool) (edges : List (Edge n)) (claimCut : ℚ) :
    certOK p edges claimCut → cutWeight p edges = claimCut := by
  intro h
  exact h

/-
Optional: symmetry lemma — flipping all bits preserves cut weight.
-/

/-- Flip a partition: `p' i = ! (p i)`. -/
def flipPart {n : Nat} (p : Fin n → Bool) : Fin n → Bool :=
  fun i => !(p i)

@[simp] lemma bxor_flip_left (a b : Bool) : bxor (!a) b = bxor a b := by
  -- brute force on Bool
  cases a <;> cases b <;> decide

@[simp] lemma bxor_flip_right (a b : Bool) : bxor a (!b) = bxor a b := by
  cases a <;> cases b <;> decide

@[simp] lemma edgeCut_flip {n : Nat} (p : Fin n → Bool) (e : Edge n) :
    edgeCut (flipPart p) e = edgeCut p e := by
  unfold edgeCut flipPart
  -- bxor (!p u) (!p v) = bxor (p u) (p v)
  -- use simp lemmas
  have : bxor (!(p e.u)) (!(p e.v)) = bxor (p e.u) (p e.v) := by
    cases (p e.u) <;> cases (p e.v) <;> decide
  simpa [bxor] using this

/-- Flipping all partition bits preserves cutWeight. -/
theorem cutWeight_flip_invariant {n : Nat} (p : Fin n → Bool) (edges : List (Edge n)) :
    cutWeight (flipPart p) edges = cutWeight p edges := by
  unfold cutWeight
  induction edges with
  | nil =>
      simp
  | cons e es ih =>
      simp [List.foldl]
      -- foldl is annoying to rewrite directly; switch to foldr with BigOperators is heavier.
      -- We use a helper lemma by rewriting foldl on cons:
      -- but easiest: unfold with foldl definition in simp is enough because it expands one step.
      -- We'll reason with by_cases and ih on the remainder using a foldl-recursion lemma.
      -- Introduce a local lemma: for any acc, foldl step equality holds.
      let step := fun (acc : ℚ) (ed : Edge n) =>
        if edgeCut (flipPart p) ed then acc + ed.w else acc
      let step0 := fun (acc : ℚ) (ed : Edge n) =>
        if edgeCut p ed then acc + ed.w else acc
      -- Prove step agrees pointwise
      have hstep : ∀ acc ed, step acc ed = step0 acc ed := by
        intro acc ed
        by_cases hc : edgeCut p ed
        · have hc' : edgeCut (flipPart p) ed = true := by simpa [edgeCut_flip, hc]
          simp [step, step0, hc, hc']
        · have hc' : edgeCut (flipPart p) ed = false := by simpa [edgeCut_flip, hc]
          simp [step, step0, hc, hc']
      -- Now rewrite foldl with the same step; use List.foldl_congr
      -- Mathlib has List.foldl_congr in newer versions; provide a manual induction instead:
      -- We'll show for any acc: foldl step acc edges = foldl step0 acc edges.
      have hfold : ∀ acc, List.foldl step acc (e :: es) = List.foldl step0 acc (e :: es) := by
        intro acc
        -- expand one step then use induction on es
        simp [List.foldl, hstep]
        -- reduce to showing foldl on es matches; prove general lemma for es
        have : List.foldl step (step0 acc e) es = List.foldl step0 (step0 acc e) es := by
          -- prove by induction on es
          induction es generalizing (step0 acc e) with
          | nil => simp
          | cons e2 es2 ih2 =>
              simp [List.foldl, hstep]
              exact ih2 _
        simpa using this
      -- Apply with acc=0, then simplify
      have := hfold 0
      simpa [step, step0] using this

end RigettiB