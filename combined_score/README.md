# Model 1 + Model 2: CycleSignal risk-awareness score

`cyclesignal_risk_score.py` combines **40% cycle pattern** (from Model 1's period-log flags)
and **60% profile score** (Model 2's Extra Trees output) into a 0-100 score with four bands.
The weights and bands are project heuristics, not clinically validated, and the score is
not a probability of disease or a diagnosis.

Run from the repository root:

    python combined_score/cyclesignal_risk_score.py                  # evaluate Model 2 + simulated demo scores
    python combined_score/cyclesignal_risk_score.py --evaluate-only

Demo output: `results/cyclesignal_demo_scores.json` (simulated pairings, not real participants).
The demo app imports these functions directly.
