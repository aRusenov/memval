Spatial
```
python bin/run_benchmark.py --model original_eqprop --suite spatial \
    --benchmarks tmaze_completion --benchmark-args tmaze_completion:epochs=200
```


Cognitive phenomena (L0, descriptive; own output dir so the full-suite metrics.json is not rewritten)
```
python bin/run_benchmark.py --model hopfield --suite symbolic \
    --benchmarks cognitive_phenomena --output-dir results/cognitive_l0
```
2026-09-07 AHN: results/cognitive_l0/AsymmetricHopfieldNetwork/symbolic/cognitive_phenomena_metrics.json
(+ plots/cognitive_*.png). At ceiling from one pass; read margin / runner-up. See docs/cognitive_phenomena_design.md S9.
2026-09-07 theta + dts_esn added; all three re-run at n_seeds=5 (15 replicates) after the n=3 run
was found sign-unstable on every band index. theta is bit-identical to AHN (controlled-pair
regression test). See docs/cognitive_phenomena_design.md S9.

DTS-ESN capacity report (3 panels: serial position / list length / presentation rate; human schematic dotted)
```
python bin/esn_cognitive_report.py --seeds 6 --lengths 10 20 40 80 160 320 640 1280 2560 --passes 1 2 4 8 16
```
2026-09-09: results/cognitive_l0/DTSESNSequenceNetwork/symbolic/plots/esn_capacity_report.png (+ .json). ~80 s.
Random-vector lists, decoded over 2560 candidates. Training-matched (warm) probe holds 1.0 to L=320,
0.95 at 640, 0.74 at 1280, 0.45 at 2560; the house cold probe falls from L=160 (0.87) to 0.02 at 2560.
