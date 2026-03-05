/-
RigettiB_Certificate.lean
Executable Lean 4 certificate for Problem B (180 nodes).

Uses:
  - problemb.csv  (edge list)
  - partition from loopei_result_B.json

Verification strategy:
  1. Load CSV
  2. Parse edges
  3. Scale weights by 1e12 to integers
  4. Compute cut
  5. Compare to certified scaled cut
-/

import Lean
import Std
open Lean
open Std

namespace RigettiB

--------------------------------------------------------------------------------
-- PARAMETERS FROM CERTIFIED RUN (JSON)
--------------------------------------------------------------------------------

def certifiedCutFloat : Float :=
  7099.571726101166

def certifiedTotalFloat : Float :=
  7465.707486325382

def certifiedMPES : Float :=
  0.9509576606242809

def scale : Int := 1000000000000  -- 1e12 scaling for exact integer comparison

def certifiedCutScaled : Int :=
  (certifiedCutFloat * (Float.ofInt scale)).toUInt64.toInt

--------------------------------------------------------------------------------
-- PARTITION (180 bits) FROM JSON
--------------------------------------------------------------------------------

def partitionList : List Bool :=
[
true,false,false,true,false,true,false,false,true,true,false,true,false,false,false,true,
true,true,true,false,true,false,true,false,true,true,false,false,true,true,false,true,
false,true,false,false,false,true,false,false,false,true,false,true,false,false,true,true,
false,false,false,false,false,false,false,false,true,false,true,false,false,true,true,false,
false,true,true,false,false,true,false,true,true,true,false,true,true,false,false,true,
true,false,true,false,true,false,false,false,false,true,true,false,true,false,true,false,
false,true,true,true,true,true,false,true,false,true,false,false,true,false,false,false,
true,false,true,true,true,true,true,true,false,true,false,false,false,false,true,true,
false,true,true,false,false,true,false,false,false,true,false,true,false,true,true,true,
false,false,true,true,true,false,false,false,true,true,false,false,false,true,true,true,
false,false,true,true,false,false,false,false,false,true,true,true,true,true,false,false,
true,true,false
]

def partition (i : Nat) : Bool :=
  match partitionList.get? i with
  | some b => b
  | none   => false

--------------------------------------------------------------------------------
-- EDGE STRUCTURE
--------------------------------------------------------------------------------

structure Edge where
  u : Nat
  v : Nat
  w : Int  -- scaled weight

--------------------------------------------------------------------------------
-- CSV LOADING
--------------------------------------------------------------------------------

def parseLine (line : String) : Option Edge :=
  let parts := line.splitOn ","
  if parts.length < 3 then none
  else
    let u := parts.get! 0 |>.trim.toNat!
    let v := parts.get! 1 |>.trim.toNat!
    let wFloat := parts.get! 2 |>.trim.toFloat!
    let wScaled := (wFloat * Float.ofInt scale).toUInt64.toInt
    some { u := u, v := v, w := wScaled }

def loadEdges : IO (List Edge) := do
  let content ← IO.FS.readFile "problemb.csv"
  let lines := content.splitOn "\n"
  let data := lines.drop 1  -- drop header
  pure <| data.filterMap parseLine

--------------------------------------------------------------------------------
-- CUT COMPUTATION
--------------------------------------------------------------------------------

def isCut (e : Edge) : Bool :=
  partition e.u != partition e.v

def computeCut (edges : List Edge) : Int :=
  edges.foldl (fun acc e =>
    if isCut e then acc + e.w else acc) 0

--------------------------------------------------------------------------------
-- CERTIFICATE VERIFICATION
--------------------------------------------------------------------------------

def verify : IO Unit := do
  let edges ← loadEdges
  let cut := computeCut edges
  IO.println s!"Computed scaled cut: {cut}"
  IO.println s!"Certified scaled cut: {certifiedCutScaled}"
  if cut == certifiedCutScaled then
    IO.println "CERTIFICATE VERIFIED ✓"
  else
    IO.println "CERTIFICATE FAILED ✗"

#eval verify

end RigettiB