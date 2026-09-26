"""Run upstream Vieth & Triesch headless and dump a trace, as the port's reference.

Upstream: gitmv/GABA_Modulated_STDP_Paper (MIT), Experiments/Experiment_char.py,
on trieschlab/PymoNNto master. Nothing here is modified model code -- the
behaviour modules are imported verbatim; only the network is built in-process
with a settable size, seed and step count, and no Qt UI or StorageManager.

    python vieth_ref.py --steps 300 --exc 60 --out ref_small.npz
"""
import argparse, os, random as rnd, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pymonnto_pkg"))
sys.path.insert(0, os.path.join(HERE, "vieth_repo", "Experiments"))

import numpy as np
from PymoNNto import *
from Behavior_Core_Modules import *
from Behavior_Text_Modules import *
from Helper import n_chars, n_unique_chars, get_random_sentences


def build(n_exc, grammar, seed):
    """Experiment_char.py's network, verbatim in structure and parameters."""
    np.random.seed(seed)
    rnd.seed(seed)
    n_inh = n_exc / 10
    target_act = 1 / n_chars(grammar)
    net = Network(tag='char', settings=settings)

    NeuronGroup(net=net, tag='inp_neurons',
                size=Grid(width=10, height=n_unique_chars(grammar), depth=1,
                          centered=False),
                behavior={10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
                          50: Output_TextActivator(),
                          80: TextReconstructor()})

    NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(n_exc), behavior={
        3: Normalization(direction='afferent and efferent', syn_type='DISTAL',
                         exec_every_x_step=200),
        3.1: Normalization(direction='afferent', syn_type='SOMA',
                           exec_every_x_step=200),
        12: SynapseOperation(transmitter='GLU', strength=1.0),
        20: SynapseOperation(transmitter='GABA', strength=-1.0),
        30: IntrinsicPlasticity(target_activity=target_act,
                                strength=0.008735764741458582, init_sensitivity=0),
        40: GABAModulation(transmitter='GABA', strength=6.450234496564654,
                           avg_inh=0.3427857658747104, min=-0.15),
        41: STDP(transmitter='GLU', strength=0.0030597477411211885),
        51: Output_Excitatory(exp=0.7378726012049153, mul=2.353594052973287),
    })

    NeuronGroup(net=net, tag='inh_neurons1', size=getGrid(n_inh), behavior={
        60: SynapseOperation(transmitter='GLUI', strength=1.0),
        70: Output_Inhibitory(avg_inh=0.3427857658747104,
                              target_activity=target_act, duration=2),
    })

    SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1',
                 behavior={1: CreateWeights()})
    SynapseGroup(net=net, tag='EE,GLU,DISTAL', src='exc_neurons1', dst='exc_neurons1',
                 behavior={1: CreateWeights(normalize=False)})
    SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons1', dst='inh_neurons1',
                 behavior={1: CreateWeights()})
    SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons1', dst='exc_neurons1',
                 behavior={1: CreateWeights()})
    net.initialize(info=False)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--exc", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sentences", type=int, default=1)
    ap.add_argument("--trace-steps", type=int, default=0,
                    help="record per-step spikes/weights for the first K steps")
    ap.add_argument("--recovery-steps", type=int, default=0)
    ap.add_argument("--free-steps", type=int, default=0)
    ap.add_argument("--out", default="vieth_ref.npz")
    args = ap.parse_args()

    grammar = get_random_sentences(args.sentences)
    net = build(args.exc, grammar, args.seed)

    W0 = {s.tags[0]: s.W.copy() for s in net.SynapseGroups}
    keys = ["exc_spike", "exc_spike_old", "inp_spike", "inp_spike_old",
            "inh_spike", "inh_avg_act", "li_stdp_mul", "voltage", "input_GABA",
            "sensitivity", "char_index", "recon_index", "recon_act",
            "W_EE", "W_ES", "W_IE", "W_EI"]
    trace = {k: [] for k in keys}
    exc, inp, inh = net.exc_neurons1, net.inp_neurons, net.inh_neurons1

    for it in range(args.steps):
        net.simulate_iteration()
        if it < args.trace_steps:
            # Everything as it stands at the END of the iteration, which is what
            # the next iteration's behaviours read.
            trace["exc_spike"].append(exc.spike.copy())
            trace["exc_spike_old"].append(exc.spike_old.copy())
            trace["inp_spike"].append(inp.spike.copy())
            trace["inp_spike_old"].append(inp.spike_old.copy())
            trace["inh_spike"].append(inh.spike.copy())
            trace["inh_avg_act"].append(np.asarray(inh.Output_Inhibitory.avg_act).copy())
            trace["li_stdp_mul"].append(exc.li_stdp_mul.copy())
            trace["voltage"].append(exc._voltage.copy())
            trace["input_GABA"].append(exc.input_GABA.copy())
            trace["sensitivity"].append(exc.sensitivity.copy())
            trace["char_index"].append(inp.current_char_index)
            trace["recon_index"].append(inp.current_reconstruction_char_index)
            trace["recon_act"].append(inp.rec_act.copy())
            for tag in ("EE", "ES", "IE", "EI"):
                trace[f"W_{tag}"].append(net[tag, 0].W.copy())

    # Upstream's own generation protocol: input off, settle, then free-run.
    free_text = ""
    extra = {}
    if args.free_steps:
        net.deactivate_behaviors('Generator')
        for _ in range(args.recovery_steps):
            net.simulate_iteration()
        # State at the instant free-running begins, so the port can be started
        # from upstream's own endpoint instead of its own training run.
        free_state = {
            "sensitivity": net.exc_neurons1.sensitivity.copy(),
            "exc_spike": net.exc_neurons1.spike.copy(),
            "exc_spike_old": net.exc_neurons1.spike_old.copy(),
            "inh_spike": net.inh_neurons1.spike.copy(),
            "inh_avg_act": np.asarray(net.inh_neurons1.Output_Inhibitory.avg_act).copy(),
            "iteration": np.asarray(net.iteration),
        }
        for k, v in free_state.items():
            extra[f"free_{k}"] = v
        for s_ in net.SynapseGroups:
            extra[f"freeW_{s_.tags[0]}"] = s_.W.copy()
        mark = len(net.inp_neurons.Reconstructor.reconstruction_history)
        rates = []
        for _ in range(args.free_steps):
            net.simulate_iteration()
            rates.append(float(net.exc_neurons1.spike.mean()))
        extra["free_rates"] = np.asarray(rates)
        free_text = net.inp_neurons.Reconstructor.reconstruction_history[mark:]
        tg = net['Generator', 0]
        print(f"free-run  rate={np.mean(rates):.4f} target={1/n_chars(grammar):.4f} "
              f"text_score={tg.get_text_score(free_text):.3f}")
        print("generated:", repr(free_text[:120]))

    out = {f"W0_{k}": v for k, v in W0.items()}
    out["free_text"] = np.asarray(list(free_text))
    out.update(extra)
    out.update({f"W_{s.tags[0]}": s.W.copy() for s in net.SynapseGroups})
    for k, v in trace.items():
        if v:
            out[f"trace_{k}"] = np.asarray(v)
    out["alphabet"] = np.asarray(list(net.inp_neurons.Generator.alphabet))
    out["input_history"] = np.asarray(list(net.inp_neurons.Generator.history))
    out["recon_history"] = np.asarray(list(net.inp_neurons.Reconstructor.reconstruction_history))
    np.savez_compressed(os.path.join(HERE, args.out), **out)

    print(f"steps={args.steps} exc={args.exc} seed={args.seed} "
          f"alphabet={''.join(net.inp_neurons.Generator.alphabet)!r}")
    print("input :", net.inp_neurons.Generator.history[-60:])
    print("recon :", net.inp_neurons.Reconstructor.reconstruction_history[-60:])
    for s in net.SynapseGroups:
        print(f"  {s.tags[0]:6s} {str(s.W.shape):12s} mean={s.W.mean():.6f} max={s.W.max():.6f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
