"""
Collection of NEK5000 usage 
"""
import os
import subprocess
import shutil
from configs import Simulation as nek
from configs import Runner as drl
from pathlib import Path
from lib.writer_int_pos import write_channel
"""
A checklist of dependencies: 

        - CASE.re2
        - CASE.ma2
        - CASE.par
        - SESSION.NAME
        - int_pos (if needed)
        - nek5000 

"""

V17_PARAM_MAPPING = {
    "density": "p001",
    "viscosity": "p002",
    "numSteps": "p011",
    "dt": "p012",
    "writeInterval": "p015",
    "target_cfl": "p026",
    "writeLA2": "p070",
    "p_residualTol": "p021",
    "v_residualTol": "p022",
    "writePTS": "p051",
    "READCHKPT": "p076",
    "CHKPFNUMBER": "p067",
    "CHKPINTERVAL": "p075",
    "AVSTEP": "p087",
    "IOSTEP": "p088",
    "ndrl": "p089",
    "znmf_avg": "p090",
}


class NEK_INIT():
    def __init__(self, nek: nek, drl: drl, rank_folder, emb=None,
                 log=None) -> None:
        """
        A class for initialization of NEK Dependencies
        nek:[dataclass]Simulation config
        drl:[dataclass]DRL config
        rank_folder:[str]target folders to run drl
        emb:[dataclass]Embedded config, or None to leave the embedded
            (Python-free) mode switched off  #[MOD]
        """
        self.nek = nek
        self.drl = drl
        self.emb = emb
        self.log = log
        self.rank_folder = rank_folder
        self.is_done = []

    # -----------------------------------------
    def _embedded_on(self) -> bool:
        """#[MOD] True when the run should be driven by the embedded actor."""
        return self.emb is not None and bool(getattr(self.emb, "enabled", False))
    # -----------------------------------------

    def _derive_embedded_numsteps(self) -> None:
        """Give an embedded run exactly enough solver steps for its budget.

        Unlike coupled evaluation, the embedded control loop owns the
        interaction budget and stops itself after ``nb_interactions``.  Keep
        the solver-level stop condition in lockstep with that budget before
        writing either a v17 .rea or a v19 .par file.
        """
        if not self._embedded_on():
            return

        try:
            nb_interactions = int(self.drl.nb_interactions)
            ndrl = int(self.nek.ndrl)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "[POLICY] embedded mode requires integer runner.nb_interactions "
                "and simulation.ndrl") from exc
        if nb_interactions < 1 or ndrl < 1:
            raise ValueError(
                "[POLICY] embedded mode requires runner.nb_interactions and "
                "simulation.ndrl to both be positive")

        derived = nb_interactions * ndrl
        configured = int(self.nek.numSteps)
        self.nek.numSteps = derived
        if configured == derived:
            print(f"[POLICY] numSteps={derived} matches embedded budget",
                  flush=True)
        else:
            print(f"[POLICY] set numSteps={derived} "
                  f"(nb_interactions={nb_interactions} * ndrl={ndrl}; "
                  f"was {configured})", flush=True)
    # -----------------------------------------

    def _resume_embedded_checkpoint(self) -> None:
        """Select the newest complete local restart set for ``--resume``.

        The v19 checkpoint toolbox alternates two three-file sets.  Its
        ``CHKPFNUMBER`` input is not persisted after a successful output, so
        regenerating a ``.par`` with the YAML's original value made resumes
        reread set 1.  Incomplete writes are deliberately ignored.

        The v17 wing checkpoint implementation maintains its own
        ``<case>.restart`` pointer file, so it needs no parameter rewrite.
        """
        if not (self._embedded_on()
                and bool(getattr(self.emb, 'resume', False))):
            return

        case = str(self.nek.CASENAME)
        if self._is_v17():
            pointer = os.path.join(self.rank_folder, f'{case}.restart')
            if not os.path.isfile(pointer):
                raise FileNotFoundError(
                    f"[POLICY] --resume requested but v17 restart pointer is "
                    f"missing: {pointer}")
            with open(pointer) as f:
                set_no = f.readline().strip()
            print(f"[POLICY] resume v17 checkpoint set {set_no} from {pointer}",
                  flush=True)
            return

        # v19 KTH_Framework checkpoint/mstep: two cyclic sets, each formed by
        # three snapshots.  The ASCII header carries saved time then ISTEP.
        # A fresh solver invocation resets ISTEP to zero, so time—not ISTEP—is
        # the ordering key across separately launched embedded segments.
        import re

        name_re = re.compile(rf'^rs.{re.escape(case)}0\.f(\d{{5}})$')
        files = {}
        for name in os.listdir(self.rank_folder):
            match = name_re.match(name)
            if match:
                files[int(match.group(1))] = os.path.join(self.rank_folder, name)

        candidates = []
        for set_no in (1, 2):
            first = 3 * (set_no - 1) + 1
            members = [files.get(first + offset) for offset in range(3)]
            if any(path is None for path in members):
                continue
            try:
                # Nek's fixed-size ASCII field header is 132 bytes; reading
                # farther reaches binary mesh data and would reject a valid
                # checkpoint under strict decoding.
                with open(members[-1], 'rb') as f:
                    words = f.read(132).decode('ascii', errors='strict').split()
                if not words or words[0] != '#std':
                    continue
                saved_time = float(words[7])
                saved_step = int(words[8])
            except (OSError, UnicodeDecodeError, IndexError, ValueError):
                continue
            candidates.append((saved_time, saved_step, set_no))

        if not candidates:
            raise RuntimeError(
                "[POLICY] --resume requested but no complete v19 checkpoint "
                f"set (rs?{case}0.f00001..00006) exists in {self.rank_folder}")

        saved_time, saved_step, set_no = max(candidates)
        old = int(self.nek.CHKPFNUMBER)
        self.nek.CHKPFNUMBER = set_no
        print(f"[POLICY] resume v19 checkpoint set {set_no} "
              f"(saved TIME={saved_time:.12g}, ISTEP={saved_step}; "
              f"CHKPFNUMBER {old} -> {set_no})",
              flush=True)
    # -----------------------------------------

    def _is_v17(self) -> bool:
        version = getattr(self.nek, "solver_version", "v19")
        return "v17" in str(version).lower()

    # -----------------------------------------
    #[MOD] Dependencies are now resolved from SEVERAL source folders instead of
    #[MOD] compile_path only: the case folder (envs/cases/<case>) first, then the
    #[MOD] shared data folder(s) (data/simulations/<case>). This way a new case
    #[MOD] variant only has to carry what actually differs (source, SIZE, .par),
    #[MOD] while the bulky mesh/mask files stay in ONE place.
    def _source_dirs(self) -> list:
        """Search order for the case dependencies: case folder, then shared data."""
        dirs = [self.nek.compile_path]

        shared = getattr(self.nek, "shared_data_path", "")
        if isinstance(shared, str):
            shared = [shared] if shared else []
        for d in (shared or []):
            if not d:
                continue
            if not os.path.isdir(d):
                print(f"[IO] WARNING: shared data folder NOT EXIST: {d}", flush=True)
                continue
            dirs.append(d)
        return dirs

    # -----------------------------------------
    def _locate(self, fname: str):
        """
        Find fname in the source folders.
        Returns (path, is_shared) or (None, False) if it is nowhere to be found.
        """
        for i, d in enumerate(self._source_dirs()):
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                return candidate, (i > 0)
        return None, False

    # -----------------------------------------
    def _fetch(self, fname: str, overwrite: bool) -> bool:
        """
        Make fname available in the rank folder.

        overwrite=True  (mandatory files) : always refreshed from the source.
        overwrite=False (optional files)  : kept if it is already there.

        Files coming from the shared folder are SYMLINKED rather than copied
        (unless nek.shared_data_link is False), which is what keeps the mesh /
        mask of a big case out of the per-case copy cost.
        """
        to_file = os.path.join(self.rank_folder, fname)

        if os.path.exists(to_file) or os.path.islink(to_file):
            if not overwrite:
                print(f"[IO] {to_file} EXIST", flush=True)
                return True
            os.remove(to_file)
            print(f'[IO] REMOVE EXIST: {to_file}', flush=True)

        from_file, is_shared = self._locate(fname)
        if from_file is None:
            return False

        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        if is_shared and getattr(self.nek, "shared_data_link", True):
            os.symlink(os.path.abspath(from_file), to_file)
            print(f"[IO] {to_file} LINKED <- {from_file}", flush=True)
        else:
            shutil.copy(from_file, to_file)
            print(f"[IO] {to_file} COPIED <- {from_file}", flush=True)
        return True

    # -----------------------------------------
    def get_Case_Files(self):
        """
        Get required case files for running simulation
        IF it is complusory, it will be rewritten no matter if the file exists
        IF it is optional, it will NOT be covered if it Exist.

        Each file is looked up in the case folder first and in the shared data
        folder(s) afterwards; at the end all mandatory files MUST be present in
        the run folder, no matter which source they came from.
        """
        #[MOD] The solver binary follows simulation.exeName instead of being
        # hardcoded, so an embedded run can pull ./nek5000_solo (built against
        # the raw Nek core) out of the same case folder as the coupled
        # ./nek5000. Both binaries coexist there by design.
        exename = getattr(self.nek, 'exeName', 'nek5000') or 'nek5000'

        if self._is_v17():
            checklist = {
                'must': [
                    # Solver
                    exename,
                    # Mesh
                    f"{self.nek.CASENAME}.re2",
                    f"{self.nek.CASENAME}.map",
                    f"{self.nek.CASENAME}.wall",
                    f"{self.nek.CASENAME}.restart",
                    # Time Series Probs
                    "stat_pts.in",
                    # Tripping
                    "forparam.i",
                ],
                'option': [
                    'SIZE',
                    f"mask_{self.nek.CASENAME}0.f00002",
                ]
            }
        else:
            checklist = {
                'must': [
                    # Solver
                    exename,
                    # Mesh
                    f"{self.nek.CASENAME}.re2",
                    f"{self.nek.CASENAME}.ma2",
                    f"{self.nek.CASENAME}.usr",
                    'int_pos',
                ],
                'option': [
                    'SIZE',
                    # Rotation
                    #'int_pos',
                ]
            }
        print(f"[IO] SOURCE FOLDERS: {self._source_dirs()}", flush=True)

        for fname in checklist["must"]:
            if not self._fetch(fname, overwrite=True):
                raise FileNotFoundError(
                    f"[IO] {fname} not FOUND in any of {self._source_dirs()}!")

        for fname in checklist["option"]:
            #[MOD] A missing optional file is a warning, not a crash (it used to
            #[MOD] raise inside shutil.copy).
            if not self._fetch(fname, overwrite=False):
                print(f"[IO] WARNING: optional {fname} not FOUND in "
                      f"{self._source_dirs()}", flush=True)

        return True

    # -----------------------------------------
    def write_SESSION_NAME(self):
        """Write the session name and where the code should be executed"""

        solver_root = Path(self.rank_folder)
        fileName = os.path.join(solver_root, 'SESSION.NAME')
        is_exist = os.path.exists(fileName)
        if is_exist:
            os.remove(fileName)
        command = "cd %s && touch SESSION.NAME" % solver_root
        subprocess.call(command, shell=True)
        command = "cd %s && echo %s > SESSION.NAME" % (
            solver_root, self.nek.CASENAME)
        subprocess.call(command, shell=True)
        command = "cd %s && echo $(pwd) >> SESSION.NAME" % solver_root
        subprocess.call(command, shell=True)
        print('[IO] SESSION NAME WRITTEN', flush=True)
        return True

    # -----------------------------------------
    def rewrite_REA_v17(self):
        """
        Re-Write parameter files for NEK version <= 17.
        For the controllable params, please see config.
        """
        #[MOD] The template .rea is also resolved through the shared folders.
        file_path, _ = self._locate(f"{self.nek.CASENAME}.rea")
        if file_path is None:
            raise FileNotFoundError(
                f"[IO] {self.nek.CASENAME}.rea not FOUND in any of "
                f"{self._source_dirs()}!")
        output_path = os.path.join(self.rank_folder, f'{self.nek.CASENAME}.rea')

        with open(file_path, 'r') as f:
            lines = f.readlines()
            updated_lines = []
            for line in lines:
                if 'p' in line:
                    parts = line.split()
                    #[MOD] Control mode. v17 has no UPARAM at all -- the DRL
                    # code reads PARAM(89)/PARAM(90) straight out of the .rea
                    # -- so the embedded switch takes the free p091 slot
                    # rather than the UPARAM(10) used by the v19 path. It is
                    # handled here instead of through V17_PARAM_MAPPING
                    # because its value comes from the Embedded config, not
                    # from Simulation.
                    if len(parts) > 1 and parts[1] == 'p091':
                        parts[0] = f"{1.0 if self._embedded_on() else 0.0:.6E}"
                        if len(parts) > 2:
                            parts[2:] = ['YW:', 'CTRL_MODE', '(1==EMBEDDED)']
                        else:
                            parts += ['YW:', 'CTRL_MODE', '(1==EMBEDDED)']
                        updated_lines.append('\t'.join(parts) + '\n')
                        continue
                    if len(parts) > 1 and parts[1] in V17_PARAM_MAPPING.values():
                        for attr, pkey in V17_PARAM_MAPPING.items():
                            if parts[1] == pkey:
                                parts[0] = f"{getattr(self.nek, attr):.6E}"
                                line = '\t'.join(parts) + '\n'
                                break
                updated_lines.append(line)

        with open(output_path, 'w') as f:
            f.writelines(updated_lines)

        return True

    # -----------------------------------------
    def rewrite_REA_v19(self):
        """
        Write parameter files for NEK version >= 19
        """
        fname = os.path.join(self.rank_folder, f'{self.nek.CASENAME}.par')
        print(f"[IO] Writting .par file:\n{fname}", flush=True)
        with open(fname, 'w') as fpar:
            fpar.write("# nek parameter file\n")

            # ------------------------
            # General Setup
            # ------------------------
            fpar.write("[GENERAL]\n")
            if self.nek.stopAt is not None:
                fpar.write("stopAt = %s   \n" % self.nek.stopAt)
                fpar.write("numSteps = %d \n" % self.nek.numSteps)

            fpar.write("dt = %.3e           \n" % self.nek.dt)
            fpar.write("timeStepper = %s  \n" % self.nek.timeStepper)
            fpar.write("variableDt = %s   \n" % self.nek.variableDt)
            fpar.write("\n")
            fpar.write('writeControl = %s \n' % self.nek.writeControl)
            fpar.write('writeInterval = %d\n' % self.nek.writeInterval)
            fpar.write("\n")
            fpar.write('dealiasing = %s   \n' % self.nek.dealiasing)
            fpar.write('filtering = %s    \n' % self.nek.filtering)
            fpar.write('filterWeight = %f \n' % self.nek.filterWeight)
            fpar.write('filterCutoffRatio = %f \n' %
                       self.nek.filterCutoffRatio)
            fpar.write("\n")

            # ---------------------------
            # DRL SETUP
            # ---------------------------
            userp = 1
            fpar.write('#------DRL SETUP-------\n')
            fpar.write('userParam%02d = %s \n' % (userp, self.nek.ndrl)) # Number of DRL step
            userp += 1
            fpar.write('userParam%02d = %s \n' % (userp, self.nek.znmf_avg)) # Average Z-mode for DRL
            userp += 1
            fpar.write('userParam%02d = %s \n' % (userp, self.nek.y_sensing)) # Sensing plane location for DRL
            userp += 1
            fpar.write('userParam%02d = %s \n' % (userp, self.nek.retau)) # Reynolds number for reference channel
            userp += 1
            fpar.write('userParam%02d = %s \n' % (userp, self.nek.ys_bdf)) # Sensing plane location for Body-Force
            userp += 1
            fpar.write('userParam%02d = %s \n' % (userp, self.nek.amp_bdf)) # Amplitude for Body-Force
            userp += 1
            fpar.write('userParam%02d = %s \n' % (userp, self.nek.ret_bdf)) # Scale for Body-Force
            userp += 1
            fpar.write('userParam%02d = %s \n' % (userp, self.nek.gll_unique)) # Unique GLL points for DRL (0==No, 1==Yes)
            userp += 1
            # [MOD] Reward mode selector (UPARAM(9) in Fortran).
            # Derived from Runner.reward_fn so the Fortran reward path and the
            # Python MPI-Recv path share ONE source of truth (avoids deadlock).
            # 0 = dudy (drag reduction), 1 = net_gain (net energy saving).
            reward_fn = getattr(self.drl, 'reward_fn', 'dudy')
            if reward_fn not in ('dudy', 'net_gain'):
                raise ValueError(
                    f"[IO] Unknown reward_fn '{reward_fn}'; "
                    "expected 'dudy' or 'net_gain'")
            reward_mode = 1 if reward_fn == 'net_gain' else 0
            fpar.write('userParam%02d = %s \n' % (userp, reward_mode)) # Reward mode: 0=dudy, 1=net_gain
            userp += 1
            #[MOD] Control mode (UPARAM(10) in Fortran).
            # 0 = coupled  : Python drives the solver over MPI (training and
            #                the original evaluation path).
            # 1 = embedded : the actor is evaluated inside Nek5000, no Python.
            # The embedded value only makes sense for a binary built against
            # the raw solver (./nek5000_solo); the DRL-core binary reserves
            # rank 0 for Python and would waste it.
            fpar.write('userParam%02d = %s \n' %
                       (userp, 1 if self._embedded_on() else 0))
            fpar.write('#---------------------\n')
            fpar.write("\n")
            # ------------------------
            # Problem Type
            # ------------------------
            fpar.write("[PROBLEMTYPE]\n")
            fpar.write("stressFormulation = %s\n" % self.nek.stressFormulation)
            fpar.write("variableProperties = %s\n" %
                       self.nek.variableProperties)
            fpar.write("\n")
            # ------------------------
            # PRESSURE
            # ------------------------
            fpar.write("[PRESSURE]\n")
            fpar.write("residualTol = %e\n" % self.nek.p_residualTol)
            fpar.write("residualProj = %s\n" % self.nek.p_residualProj)
            fpar.write("\n")
            # ------------------------
            # VELOCITY
            # ------------------------
            fpar.write("[VELOCITY]\n")
            fpar.write("residualTol = %e\n" % self.nek.v_residualTol)
            fpar.write("residualProj = %s\n" % self.nek.v_residualProj)
            fpar.write("density = %f\n" % self.nek.density)
            fpar.write("viscosity = %f\n" % self.nek.viscosity)
            fpar.write("advection = %s\n" % self.nek.advection)
            fpar.write("\n")
            # ------------------------
            # _RUNPAR
            # ------------------------
            fpar.write("[_RUNPAR]\n")
            fpar.write("PARFWRITE = %s\n" % self.nek.PARFWRITE)
            fpar.write("outparfile = %s\n" % self.nek.PARFNAME)
            fpar.write("\n")
            # ------------------------
            # MOINTOR
            # ------------------------
            fpar.write("[_MONITOR]\n")
            fpar.write("LOGLEVEL = %d\n" % self.nek.LOGLEVEL)
            fpar.write("WALLTIME = %s\n" % self.nek.WALLTIME)
            fpar.write("\n")
            # ------------------------
            # _CHECKPOINT
            # ------------------------
            fpar.write("[_CHKPOINT]\n")
            fpar.write("READCHKPT = %s\n" % self.nek.READCHKPT)
            fpar.write("CHKPFNUMBER = %d\n" % self.nek.CHKPFNUMBER)
            fpar.write("CHKPINTERVAL = %d\n" % self.nek.CHKPINTERVAL)
            fpar.write("\n")
            # ------------------------
            # _STAT
            # ------------------------
            fpar.write("[_STAT]\n")
            fpar.write("AVSTEP = %d\n" % self.nek.AVSTEP)
            fpar.write("IOSTEP = %d\n" % self.nek.IOSTEP)
            fpar.write("\n")
            # ------------------------
            # _STAT
            # ------------------------
            fpar.write("[_TSRS]\n")
            fpar.write("SMPSTEP = %d\n" % self.nek.SMPSTEP)
            fpar.write("\n")
            fpar.close()
            print(f'[IO] WRITTEN .par file: {fname}', flush=True)
        return True
    # -----------------------------------------

    def init_restart(self):
        """Copy the restart file to the target folder only if RSTART NOT EXIST"""

        restart_folder = os.path.join(
            self.nek.restart_folder, f"init_{self.drl.rank}")
        rs_list = os.listdir(restart_folder)
        rs_list = [f for f in rs_list if 'rs' in f]

        rs_exist = os.listdir(self.rank_folder)
        rs_exist = [f for f in rs_exist if 'rs' in f]

        if len(rs_exist) == 0:
            print(f"[INIT] IMPORTING RESTART FILES!", flush=True)
            for rsfile in rs_list:
                rsfile = os.path.join(restart_folder, rsfile)
                shutil.copy(rsfile, dst=self.rank_folder)
                print(f"[STB3] RS6 file RESET: {rsfile}", flush=True)
                # print('[STB3] RS6 file RESET',flush=True)
        else:
            file_example = rs_exist[0]
            loc = file_example.find('rs')
            rsx = int(file_example[loc+2:loc+3])
            print(f"[INIT] {rsx} FILE!", flush=True)
            if len(rs_exist) < rsx//2:
                print(
                    f"[INIT] {rsx} > {len(rs_list)}: IMPORTING RESTART FILES!", flush=True)
                for rsfile in rs_list:
                    rsfile = os.path.join(restart_folder, rsfile)
                    shutil.copy(rsfile, dst=self.rank_folder)
                    print(f"[STB3] RS6 file RESET: {rsfile}", flush=True)
            else:
                print(f"[INIT] File Exists no need to copy!", flush=True)

        return True

    # -----------------------------------------
    def write_timeSeries(self):
        """Write the int_pos file for the case file"""
        #[MOD] y_planes (if set) selects the tsrs interpolation planes; it is a
        #[MOD] diagnostic knob only and leaves the DRL sensing plane untouched.
        y_planes = getattr(self.nek, 'y_planes', None)
        yplus = self.nek.y_sensing if not y_planes else list(y_planes)
        is_done = write_channel(path=self.nek.compile_path,
                                Ret=self.nek.retau, yplus=yplus,
                                Lx=self.nek.Lx, Lz=self.nek.Lz,
                                Nx=self.nek.Nx, Nz=self.nek.Nz,
                                lx1=self.nek.lx1, nproc=self.nek.nproc)
        return is_done
    # -----------------------------------------

    # -----------------------------------------
    def _policy_table(self):
        """#[MOD] Build the per-region policy table for drl_policy.in.

        Two config shapes are supported and auto-detected:

        * SINGLE (configs.py, minimal channel) -- scalar Runner fields, one
          region covering the whole wall unless embedded.* says otherwise.

        * META (configs_meta.py, wing) -- the Runner already carries the
          region table MetaPolicy uses: agent_ctrl_area, agent_ctrl_side,
          u_tau, action_bounds, drl_steps, source_solvers, and lists of
          agent_run_name/policy resolved through logging.policy_dir. That
          is read directly rather than being restated under embedded.*,
          so the two can never disagree.

        Anything set explicitly on the Embedded config overrides both.
        """
        emb, drl = self.emb, self.drl

        #[MOD] OmegaConf hands back ListConfig, which is a Sequence but NOT a
        # list subclass, so a plain isinstance(x, list) check silently reports
        # "not a sequence" and the whole meta branch is skipped. Normalise
        # once, here, rather than sprinkling the check around.
        def _isseq(v):
            return (not isinstance(v, (str, bytes))
                    and hasattr(v, '__len__') and hasattr(v, '__getitem__'))

        def _get(name, default=None):
            v = getattr(drl, name, default)
            if v is None:
                return default
            return list(v) if _isseq(v) else v

        meta_areas = _get('agent_ctrl_area', []) or []
        is_meta = _isseq(meta_areas) and len(meta_areas) > 0

        if emb.ctrl_areas is not None:
            areas = [list(a) for a in emb.ctrl_areas]
        elif is_meta:
            areas = [list(a) for a in meta_areas]
        else:
            areas = [[-1.0e9, 1.0e9]]          # one region, whole wall
        npol = len(areas)

        def _spread(value, default, name):
            """Accept a scalar, a list, or None -> list of length npol."""
            if value is None:
                return [default] * npol
            if _isseq(value):
                vals = list(value)
                if len(vals) < npol:
                    raise ValueError(
                        f"[POLICY] {name} has {len(vals)} entries but there "
                        f"are {npol} control regions")
                #[MOD] Wing configs habitually carry one spare entry (e.g.
                # 5 u_tau values for 4 regions), so extras are trimmed
                # rather than treated as an error.
                return vals[:npol]
            return [value] * npol

        if is_meta:
            d_sides = _get('agent_ctrl_side', ['ANY'] * npol)
            d_utau = _get('u_tau', 1.0)
            d_nupd = _get('drl_steps', 1)
            d_srcs = _get('source_solvers', 'nek')
            bounds = _get('action_bounds', None)
            if bounds is not None:
                # amplitude is the half-range of the action bound
                d_amp = [0.5 * (float(b[1]) - float(b[0]))
                         for b in list(bounds)[:npol]]
            else:
                d_amp = 1.0
        else:
            d_sides = 'ANY'
            d_utau = float(_get('u_tau', 1.0))
            d_nupd = 1
            d_srcs = _get('source_solver', 'nek')
            d_amp = float(_get('ctrl_max_amp', 1.0))

        sides = _spread(emb.ctrl_sides, None, 'ctrl_sides') \
            if emb.ctrl_sides is not None else _spread(d_sides, 'ANY', 'agent_ctrl_side')
        nupds = _spread(emb.nupd, None, 'nupd') \
            if emb.nupd is not None else _spread(d_nupd, 1, 'drl_steps')
        utaus = _spread(emb.u_tau, None, 'u_tau') \
            if emb.u_tau is not None else _spread(d_utau, 1.0, 'u_tau')
        amps = _spread(emb.ctrl_max_amp, None, 'ctrl_max_amp') \
            if emb.ctrl_max_amp is not None else _spread(d_amp, 1.0, 'action_bounds')
        srcs = _spread(d_srcs, 'nek', 'source_solvers')

        # ---- checkpoints
        if emb.policies is not None:
            ckpts = _spread(emb.policies, None, 'policies')
        elif is_meta:
            # MetaPolicy._load_policy: <policy_dir>/<run>/logs/<policy>.zip
            pol_dir = getattr(self.log, 'policy_dir', 'runs') if self.log else 'runs'
            runs = _spread(_get('agent_run_name', ''), '', 'agent_run_name')
            pols = _spread(_get('policy', ''), '', 'policy')
            ckpts = [os.path.join(str(pol_dir), str(r), 'logs', str(p))
                     for r, p in zip(runs, pols)]
        else:
            stem = str(drl.policy) if drl.policy else 'best_model'
            ckpts = [os.path.join('runs', str(drl.agent_run_name),
                                  'logs', stem)] * npol

        table = []
        for il in range(npol):
            table.append({
                'xmin': areas[il][0],
                'xmax': areas[il][1],
                'side': sides[il],
                'utau': float(utaus[il]),
                'amp': float(amps[il]),
                'nupd': int(nupds[il]),
                'src': srcs[il],
                'ckpt': ckpts[il],
            })
        return table

    # -----------------------------------------
    def write_policy_files(self):
        """#[MOD] Emit the embedded-mode artefacts into the run folder:
        one `.pol` per control region plus `drl_policy.in`.

        Both are derived from the run YAML, which stays the single source
        of truth. The solver reads them from its working directory, so
        the `.pol` files are referenced by bare file name.
        """
        from lib.pol_export import (export_checkpoint, side_code,
                                    write_analytic_pol, write_run_config)

        emb, drl = self.emb, self.drl
        table = self._policy_table()
        analytic = getattr(emb, 'analytic_policy', None)
        if analytic is not None:
            analytic = str(analytic).strip().upper()
            if analytic not in ('OC', 'BL', 'OC_U', 'OC_UVCOMB'):
                raise ValueError(
                    f"[POLICY] unsupported analytic_policy '{analytic}'; "
                    "expected OC, BL, OC_U, or OC_UVCOMB")
            nstate = int(getattr(drl, 'npl_state', 2))
            if nstate < 2 and analytic == 'OC':
                raise ValueError(
                    '[POLICY] analytic OC needs npl_state >= 2 for v\'')
            if nstate != 1 and analytic == 'OC_U':
                raise ValueError(
                    "[POLICY] OC_U needs npl_state=1 with val_obs(1)=u'")
            if nstate != 2 and analytic == 'OC_UVCOMB':
                raise ValueError(
                    "[POLICY] OC_UVCOMB needs npl_state=2 ordered as [u', v']")
        else:
            nstate = 0

        # Action-space bounds, mirroring how nek_marl builds them. The meta
        # Runner has no rescale_actions: MetaPolicy._rescale_actions always
        # multiplies a [-1,1] output by the region amplitude.
        if hasattr(drl, 'rescale_actions') and not bool(drl.rescale_actions):
            alow, ahigh = float(drl.ctrl_min_amp), float(drl.ctrl_max_amp)
        else:
            alow, ahigh = -1.0, 1.0

        entries = []
        for il, p in enumerate(table):
            fname = f"actor_{il:03d}.pol"
            out = os.path.join(self.rank_folder, fname)
            if analytic is not None:
                dims = write_analytic_pol(out, analytic, p['utau'], p['amp'],
                                          nin=nstate)
                algo = f'analytic:{analytic}'
            else:
                ckpt = str(p['ckpt'])
                if not ckpt.endswith('.zip'):
                    ckpt += '.zip'
                if not os.path.isfile(ckpt):
                    print(f"[POLICY] checkpoint NOT FOUND: {ckpt}", flush=True)
                    return False
                algo, dims = export_checkpoint(ckpt, out, p['utau'], p['amp'],
                                               alow, ahigh,
                                               source_solver=p.get('src', 'nek'))
            print(f"[POLICY] region {il}: {algo} "
                  f"{' -> '.join(str(d) for d in dims)} "
                  f"utau={p['utau']} amp={p['amp']} nupd={p['nupd']} "
                  f"src={p.get('src','nek')}", flush=True)
            if analytic is None:
                print(f"[POLICY]            {ckpt}", flush=True)

            entries.append({'xmin': p['xmin'], 'xmax': p['xmax'],
                            'side': side_code(p['side']), 'utau': p['utau'],
                            'amp': p['amp'], 'nupd': p['nupd'],
                            'file': fname})

        #[MOD] The recorder writes into <run folder>/drlrec/. Fortran cannot
        # portably create a directory, so it is made here; pol_rec_probe
        # falls back to the run folder itself (with a warning) if it is
        # missing, which keeps hand-staged runs working.
        os.makedirs(os.path.join(self.rank_folder, 'drlrec'), exist_ok=True)

        reward_fn = getattr(drl, 'reward_fn', 'dudy')
        reward_mode = 1 if reward_fn == 'net_gain' else 0

        cfg = os.path.join(self.rank_folder, 'drl_policy.in')
        write_run_config(cfg,
                         nb_interactions=drl.nb_interactions,
                         rec_freq=emb.rec_freq,
                         rec_bufsize=emb.rec_bufsize,
                         iprec=emb.net_precision,
                         reward_mode=reward_mode,
                         dudy_ref=(list(drl.dUdy)[0]
                                   if (not isinstance(drl.dUdy, (str, bytes))
                                       and hasattr(drl.dUdy, '__len__'))
                                   else drl.dUdy),
                         alpha=getattr(drl, 'reward_alpha', 1.0),
                         beta=getattr(drl, 'reward_beta', 1.0),
                         gamma=getattr(drl, 'reward_gamma', 1.0),
                         policies=entries)
        print(f"[POLICY] wrote {cfg}", flush=True)

        return True

    # -----------------------------------------
    def write_coupled_record_config(self):
        """Write the small recorder-only config for a Python-coupled run.

        The binary writer is shared with embedded mode, but a coupled run has
        no ``drl_policy.in`` (and must not require an F77 actor).  This file is
        intentionally emitted only by the explicit coupled-recorder opt-in.
        """
        if self.emb is None or not bool(getattr(self.emb,
                                                 'coupled_recorder', False)):
            return True

        rec_freq = int(getattr(self.emb, 'rec_freq', 1))
        rec_bufsize = int(getattr(self.emb, 'rec_bufsize', 100))
        if rec_freq < 1 or rec_bufsize < 1:
            raise ValueError(
                '[POLREC] coupled recorder requires positive rec_freq and '
                'rec_bufsize')

        reward_fn = getattr(self.drl, 'reward_fn', 'dudy')
        if reward_fn not in ('dudy', 'net_gain'):
            raise ValueError(
                f"[POLREC] unknown reward_fn '{reward_fn}' for coupled recorder")
        reward_mode = 1 if reward_fn == 'net_gain' else 0

        os.makedirs(os.path.join(self.rank_folder, 'drlrec'), exist_ok=True)
        path = os.path.join(self.rank_folder, 'drl_record.in')
        with open(path, 'w') as f:
            f.write('# coupled binary recorder: rec_freq rec_bufsize reward_mode\n')
            f.write(f'{rec_freq} {rec_bufsize} {reward_mode}\n')
        print(f'[POLREC] wrote {path}', flush=True)
        return True

    # -----------------------------------------
    def main(self):
        # Must happen before rewrite_REA_v17/rewrite_REA_v19 serialise the
        # simulation configuration into the solver input file.
        self._derive_embedded_numsteps()
        if self._is_v17():
            self.is_done.append(self.get_Case_Files())
            self.is_done.append(self.write_SESSION_NAME())
            self.is_done.append(self.rewrite_REA_v17())
            self.is_done.append(self.init_restart())
            self._resume_embedded_checkpoint()
        else:
            self.is_done.append(self.write_timeSeries())
            self.is_done.append(self.get_Case_Files())
            self.is_done.append(self.write_SESSION_NAME())
            self.is_done.append(self.init_restart())
            # init_restart supplies a fresh run's input files; resume scanning
            # must happen after that, but before .par writes CHKPFNUMBER.
            self._resume_embedded_checkpoint()
            self.is_done.append(self.rewrite_REA_v19())

        #[MOD] Embedded mode needs its networks and run configuration in
        # the working directory; pol_cfg_read aborts loudly without them.
        if self._embedded_on():
            self.is_done.append(self.write_policy_files())
        else:
            self.is_done.append(self.write_coupled_record_config())

        if False not in self.is_done:
            return True
        else:
            return False


def remove_sch(current_path):
    file_list = os.listdir(current_path)
    file_list = [f for f in file_list if ".sch" in f]
    if len(file_list) > 0:
        for f in file_list:
            os.remove(os.path.join(current_path, f))
    return


def oppo_control(observation, env):
    """Simple policy of applying the opposition CTRL"""
    actions = {}
    # Opposition control
    # -1 ==> v-velocity
    for agent in env.possible_agents:
        actions[agent] = -1.0 * observation[agent][-1, 0, 0]
    return actions


def show_title():
    text_ = """
--------------------------------------------------------
███╗   ██╗███████╗██╗  ██╗    ██████╗ ██████╗ ██╗     
████╗  ██║██╔════╝██║ ██╔╝    ██╔══██╗██╔══██╗██║     
██╔██╗ ██║█████╗  █████╔╝     ██║  ██║██████╔╝██║     
██║╚██╗██║██╔══╝  ██╔═██╗     ██║  ██║██╔══██╗██║     
██║ ╚████║███████╗██║  ██╗    ██████╔╝██║  ██║███████╗
            NEK5000 Reinforcement Learning
                Stable-Baselines3
                    Yuning Wang
--------------------------------------------------------

"""
    print(text_, flush=True)
    return


def show_end():
    text_ = """
▗▖  ▗▖▗▄▄▄▖▗▖ ▗▖    ▗▄▄▄ ▗▄▄▖ ▗▖       ▗▄▄▄▖▗▖  ▗▖▗▄▄▄ 
▐▛▚▖▐▌▐▌   ▐▌▗▞▘    ▐▌  █▐▌ ▐▌▐▌       ▐▌   ▐▛▚▖▐▌▐▌  █
▐▌ ▝▜▌▐▛▀▀▘▐▛▚▖     ▐▌  █▐▛▀▚▖▐▌       ▐▛▀▀▘▐▌ ▝▜▌▐▌  █
▐▌  ▐▌▐▙▄▄▖▐▌ ▐▌    ▐▙▄▄▀▐▌ ▐▌▐▙▄▄▖    ▐▙▄▄▖▐▌  ▐▌▐▙▄▄▀
"""
    print(text_, flush=True)
    return
