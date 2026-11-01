# Validation-only training extension decision

This rule was frozen before inspecting the initial 7001–7010 test seeds. That
evaluation exposed a pedestrian-burst safety failure. A subsequent 8001–8010
evaluation was stopped because its models had been trained under the earlier
phase rule. Neither contributes final performance data. The rule itself remains
unchanged and is reapplied to 30 new replicas trained under the revised common
rule before the untouched final seeds 9001–9010 are evaluated; see
[the safety revision](safety_revision_v3.md).
All 30 model replicas first receive 150 training episodes. For each of the six
method–reward configurations, compare the validation score at episode 100 with
the score at episode 150, using the same three validation demand seeds and all
five independent training initialisations. Lower scores are better.

Extend **every** configuration and replica to 300 episodes if **any** one
configuration has both:

1. a decrease of at least 2% in its mean validation score from episode 100 to
   episode 150; and
2. a lower validation score at episode 150 in at least four of its five
   replicas.

Otherwise, retain the 150-episode models. A noisy change in one replica alone
does not trigger the much larger training budget. Record the six comparisons
and the resulting decision before running final evaluation. If extension is
triggered, train all 30 replicas for 300 episodes in a separate versioned
output directory with the same seeds, budgets, validation rule, and reward
definitions. The current training implementation does not retain the final
optimizer state, so a 300-episode run starts from the same initialisation and
replays the first 150 episodes instead of resuming from a validation-selected
checkpoint. Do not use test results to revisit this decision.
