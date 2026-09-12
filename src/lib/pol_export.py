"""
Export Stable-Baselines3 actors into the files the embedded F77 policy reads.

Two artefacts are produced:

  <name>.pol      the network: shapes, activations, scalings, W and B.
                  Weights are the float32 values stored in the checkpoint,
                  written with 9 significant digits so they round-trip
                  exactly (verified bit-for-bit by utils/pol_selftest.sh).

  drl_policy.in   the run configuration: interaction budget, recording
                  cadence, reward weights and the per-policy control
                  region table.

Both are generated from the run YAML, which stays the single source of
truth -- neither file is meant to be hand-edited.

Used by utils/sb3_to_f77.py (CLI) and by lib/nek_utils.py (prepare step).

@yuningw
"""
import datetime
import io
import json
import os
import zipfile

import numpy as np

# activation codes shared with drl/pol_core.f
ACT_IDENTITY = 0
ACT_RELU = 1
ACT_TANH = 2

# control-side codes shared with inc_src/POLICY
SIDE_ANY = 0
SIDE_SS = 1
SIDE_PS = 2

POL_VERSION = 2

_SIDE_NAMES = {"ANY": SIDE_ANY, "": SIDE_ANY, "NONE": SIDE_ANY,
               "SS": SIDE_SS, "PS": SIDE_PS}


def side_code(side):
    """Map a side label ('SS'/'PS'/'ANY') onto the integer POLICY uses."""
    if isinstance(side, int):
        return side
    key = str(side).strip().upper()
    if key not in _SIDE_NAMES:
        raise ValueError(f"unknown control side '{side}'; "
                         f"expected one of {sorted(_SIDE_NAMES)}")
    return _SIDE_NAMES[key]


def load_state_dict(ckpt):
    import torch
    with zipfile.ZipFile(ckpt) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")),
                        map_location="cpu", weights_only=False)
        data = json.loads(z.read("data").decode())
    return sd, data


def detect_algo(sd, override=None):
    if override:
        return override.upper()
    keys = set(sd.keys())
    if any(k.startswith("actor.latent_pi.") for k in keys):
        return "SAC"
    if any(k.startswith("actor.mu.") for k in keys):
        return "TD3"          # DDPG shares the layout
    if any(k.startswith("action_net.") for k in keys):
        return "PPO"
    raise ValueError("cannot infer the algorithm from the checkpoint; "
                     "pass an explicit algo")


def _collect(sd, prefix):
    """Pull (weight, bias) pairs out of a torch Sequential stored under
    `prefix`, in module-index order. Activation modules carry no
    parameters and are skipped."""
    idx = sorted({int(k[len(prefix):].split(".")[0])
                  for k in sd if k.startswith(prefix)})
    out = []
    for i in idx:
        w = sd.get(f"{prefix}{i}.weight")
        b = sd.get(f"{prefix}{i}.bias")
        if w is None:
            continue
        out.append((w.numpy(), b.numpy()))
    return out


def extract_actor(sd, algo):
    """Return (layers, acts, squash): layers is [(W, b), ...] and acts is
    the activation code applied *after* each layer."""
    if algo in ("TD3", "DDPG"):
        # actor.mu = Sequential(Linear, ReLU, ..., Linear, Tanh)
        layers = _collect(sd, "actor.mu.")
        acts = [ACT_RELU] * (len(layers) - 1) + [ACT_TANH]
        squash = True
    elif algo == "SAC":
        # deterministic action = tanh(mu(latent_pi(obs)))
        layers = _collect(sd, "actor.latent_pi.")
        mu_w, mu_b = sd["actor.mu.weight"].numpy(), sd["actor.mu.bias"].numpy()
        acts = [ACT_RELU] * len(layers) + [ACT_TANH]
        layers = layers + [(mu_w, mu_b)]
        squash = True
    elif algo == "PPO":
        raise NotImplementedError(
            "PPO export is structurally supported but its activation "
            "function comes from policy_kwargs (tanh by default, often "
            "ReLU) and it is not squashed -- SB3 clips to the action "
            "space instead. Wire that through and validate against a "
            "real PPO checkpoint before enabling it.")
    else:
        raise ValueError(f"unsupported algorithm: {algo}")

    if not layers:
        raise ValueError(f"no linear layers found for algo {algo}")
    return layers, acts, squash


def scalings_from_conf(conf):
    """u_tau, action multiplier and action-space bounds, mirroring how
    nek_marl builds them from the runner config."""
    r = conf.runner
    utau = float(r.u_tau)
    if bool(r.rescale_actions):
        # action_space is [-1, 1] and step() multiplies by ctrl_max_amp
        return utau, float(r.ctrl_max_amp), -1.0, 1.0
    return utau, 1.0, float(r.ctrl_min_amp), float(r.ctrl_max_amp)


def obs_perm_for(source_solver, nin):
    """Observation permutation for a policy trained on another solver.

    MetaPolicy._obs_solver_arrange reverses the field axis for a
    Dedalus-trained policy (np.flip(obs, axis=1)), i.e. it feeds (v', u')
    where Nek presents (u', v'). Carrying that as an explicit permutation
    keeps the stored weights bit-identical to the checkpoint, which is
    what the round-trip test checks; folding it into W would not.
    """
    if str(source_solver).lower() == "dedalus":
        return list(range(nin, 0, -1))          # reversed, 1-based
    return list(range(1, nin + 1))              # identity


def write_pol(path, algo, layers, acts, squash, alow, ahigh,
              scl_obs, scl_act, source, obs_perm=None):
    """Write one .pol file. Returns the layer widths."""
    dims = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
    if obs_perm is None:
        obs_perm = list(range(1, dims[0] + 1))
    if sorted(obs_perm) != list(range(1, dims[0] + 1)):
        raise ValueError(f"obs_perm {obs_perm} is not a permutation of "
                         f"1..{dims[0]}")
    with open(path, "w") as f:
        f.write(f"# NEKPOL v{POL_VERSION} -- embedded actor for pol_net.f\n")
        f.write(f"# source     : {source}\n")
        f.write(f"# generated  : "
                f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# algorithm  : {algo}\n")
        f.write(f"# dims       : {' -> '.join(str(d) for d in dims)}\n")
        f.write("# acts       : 0=identity 1=relu 2=tanh\n")
        f.write("#\n")
        f.write(f"# obs perm   : {obs_perm}  (1 = identity order)\n")
        f.write("#\n")
        f.write("# layout (comments start with '#', data is list-directed):\n")
        f.write("#   version / nlayer / dims(nlayer+1) / acts(nlayer)\n")
        f.write("#   squash alow ahigh / scl_obs scl_act / obs_perm(nin)\n")
        f.write("#   then per layer: W (row-major) then B\n")
        f.write("#\n")
        f.write(f"{POL_VERSION}\n")
        f.write(f"{len(layers)}\n")
        f.write(" ".join(str(d) for d in dims) + "\n")
        f.write(" ".join(str(a) for a in acts) + "\n")
        f.write(f"{1 if squash else 0} {alow!r} {ahigh!r}\n")
        f.write(f"{scl_obs!r} {scl_act!r}\n")
        f.write(" ".join(str(p) for p in obs_perm) + "\n")
        for il, (w, b) in enumerate(layers, start=1):
            f.write(f"# ---- layer {il}: W({w.shape[0]}x{w.shape[1]}) "
                    f"then B({b.shape[0]})\n")
            for row in np.asarray(w, dtype=np.float32):
                f.write(" ".join(f"{v:.9e}" for v in row) + "\n")
            f.write(" ".join(f"{v:.9e}" for v in np.asarray(b, dtype=np.float32))
                    + "\n")
    return dims


def export_checkpoint(ckpt, out_path, scl_obs, scl_act,
                      alow=-1.0, ahigh=1.0, algo=None,
                      source_solver="nek"):
    """Read an SB3 checkpoint and write the matching .pol. Returns
    (algo, dims).

    `source_solver` names the solver the policy was TRAINED on; 'dedalus'
    reverses the observation component order, matching what
    MetaPolicy/evaluate.py do for a transferred policy.
    """
    sd, _ = load_state_dict(ckpt)
    algo = detect_algo(sd, algo)
    layers, acts, squash = extract_actor(sd, algo)
    nin = layers[0][0].shape[1]
    dims = write_pol(out_path, algo, layers, acts, squash, alow, ahigh,
                     scl_obs, scl_act, os.path.abspath(ckpt),
                     obs_perm=obs_perm_for(source_solver, nin))
    return algo, dims


def write_analytic_pol(path, kind, scl_obs, scl_act, nin=2, ivel=1,
                       gain=1.0):
    """Write a .pol for a classical (non-learnt) control law.

    These are expressible because the activation and the output squashing
    are data in the file rather than hardcoded, so a linear law is just a
    one-layer network with identity activation.

      'OC' : opposition control, action = -amp * obs[ivel] / u_tau
             identical to lib.AFC.OppoCtrl, which computes
             `-1.0 * ctrl_max_amp * observation[1]` on the u_tau-normalised
             observation. Neither side clips, so the parity holds even
             where |v'| > u_tau.

      'BL' : steady uniform blowing/suction, action = amp

      'OC_U' : streamwise proportional control, action = -amp * u'.
                 It is a one-input law for the ``lc_n7_onlyU`` state.

      'OC_UVCOMB' : oppose the requested two-component combination,
                 action = -amp * (0.5*u' - v') = amp * (v' - 0.5*u').
                 The input order is the channel convention [u', v'].
             identical to lib.AFC.BLCtrl.

    lib.AFC.SinWave is deliberately NOT supported: it is a function of
    position and time rather than of the observation, so it cannot be
    written as a policy over the observation vector.

    `ivel` is the 0-based index of the wall-normal component in the
    observation (1 for the (u',v') layout used by both the channel and
    the wing).
    """
    kind = kind.upper()
    W = np.zeros((1, nin), dtype=np.float32)
    b = np.zeros(1, dtype=np.float32)

    if kind == "OC":
        W[0, ivel] = -float(gain)
    elif kind == "BL":
        b[0] = float(gain)
    elif kind == "OC_U":
        if nin != 1:
            raise ValueError("OC_U requires exactly one u' input")
        W[0, 0] = -float(gain)
    elif kind == "OC_UVCOMB":
        if nin != 2:
            raise ValueError("OC_UVCOMB requires ordered [u', v'] inputs")
        W[0, 0] = -0.5 * float(gain)
        W[0, 1] = float(gain)
    else:
        raise ValueError(f"unsupported analytic policy '{kind}'; "
                         "expected 'OC', 'BL', 'OC_U', or 'OC_UVCOMB'")

    # squash=0: no tanh, and no SB3 unscale_action either -- the classical
    # laws in lib.AFC apply neither.
    return write_pol(path, kind, [(W, b)], [ACT_IDENTITY], False,
                     -1.0, 1.0, scl_obs, scl_act, f"analytic:{kind}")


def build_reference(ckpt, n, scl_obs, scl_act, alow, ahigh, obs_shape, seed):
    """Reference actions from SB3's own predict(), followed by the same
    in-place rescale nek_marl.step applies. This is what the Fortran is
    validated against -- deliberately the real evaluation path rather
    than a re-implementation of it."""
    from gym import spaces
    from stable_baselines3 import TD3, DDPG, SAC, PPO  # noqa: F401

    _, data = load_state_dict(ckpt)
    table = {"TD3": TD3, "DDPG": DDPG, "SAC": SAC, "PPO": PPO}
    blob = str(data.get("policy_class", {})).lower()
    cls = TD3
    for name, algo_cls in table.items():
        if name.lower() in blob:
            cls = algo_cls
            break

    nin = obs_shape[0]
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape,
                           dtype=np.float32)
    act_space = spaces.Box(low=alow, high=ahigh, shape=(1,), dtype=np.float32)
    model = cls.load(ckpt, custom_objects={"observation_space": obs_space,
                                           "action_space": act_space},
                     print_system_info=False)

    rng = np.random.default_rng(seed)
    # deliberately wider than a channel produces, so the table also
    # covers the saturated tails of tanh
    phys = rng.normal(0.0, 4.0 * scl_obs, size=(n, nin)).astype(np.float64)
    obs = (phys / scl_obs).astype(np.float32).reshape((n,) + obs_shape)
    act, _ = model.predict(obs, deterministic=True)
    act = np.asarray(act, dtype=np.float32).reshape(-1)
    act *= np.float32(scl_act)          # exactly nek_marl.step's in-place *=
    return phys, act


def write_chk(path, phys, act, source):
    with open(path, "w") as f:
        f.write("# reference table from SB3 predict(deterministic=True)\n")
        f.write(f"# source : {source}\n")
        f.write(f"# columns: {phys.shape[1]} raw observation components, "
                f"then action\n")
        f.write(f"{phys.shape[0]} {phys.shape[1]}\n")
        for p, a in zip(phys, act):
            f.write(" ".join(f"{v:.17e}" for v in p) + f" {a:.9e}\n")


def write_run_config(path, nb_interactions, rec_freq, rec_bufsize, iprec,
                     reward_mode, dudy_ref, alpha, beta, gamma, policies):
    """Write drl_policy.in, read by pol_cfg_read in drl/pol_net.f.

    `policies` is a list of dicts with keys
    xmin, xmax, side, utau, amp, nupd, file.
    """
    with open(path, "w") as f:
        f.write("# drl_policy.in -- embedded policy run configuration\n")
        f.write(f"# generated  : "
                f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write("# generated from the run YAML; do not hand-edit.\n")
        f.write("#\n")
        f.write("# nb_interactions  rec_freq  rec_bufsize  iprec\n")
        f.write(f"{int(nb_interactions)} {int(rec_freq)} "
                f"{int(rec_bufsize)} {int(iprec)}\n")
        f.write("#\n")
        f.write("# reward_mode(0=dudy,1=net_gain)  dudy_ref  alpha beta gamma\n")
        f.write(f"{int(reward_mode)} {float(dudy_ref)!r} "
                f"{float(alpha)!r} {float(beta)!r} {float(gamma)!r}\n")
        f.write("#\n")
        f.write("# number of policies\n")
        f.write(f"{len(policies)}\n")
        f.write("#\n")
        f.write("# xmin xmax iside utau amp nupd 'file.pol'\n")
        f.write("#   iside: 0 = any, 1 = suction side (y>0), "
                "2 = pressure side (y<0)\n")
        for p in policies:
            f.write(f"{float(p['xmin'])!r} {float(p['xmax'])!r} "
                    f"{int(p['side'])} {float(p['utau'])!r} "
                    f"{float(p['amp'])!r} {int(p['nupd'])} "
                    f"'{p['file']}'\n")
