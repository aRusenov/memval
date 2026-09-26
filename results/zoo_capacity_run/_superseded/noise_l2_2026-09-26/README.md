# noise_invariance with both l2 and raw rollouts (2026-09-26, intermediate)

State between the two 2026-09-26 reruns: panel 2 showed l2 and raw rollouts, and
`noise_rollout_auc` / tolerance / completion gap / margins came from l2. The
follow-up rerun dropped l2; raw now drives all of them. Values differ only for
DTS-ESN (rollout mean 0.367 l2 -> 0.305 raw) and EP (0.145 -> 0.158, rollout
tolerance 0.0 -> 0.1); AHN, theta, tPC and Chen are identical. The state before
both reruns (quantized curve) is in `../noise_quantized_2026-09-26/`.
