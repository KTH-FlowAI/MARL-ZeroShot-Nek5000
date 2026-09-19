# v-only Nek-solo records had one observation field, while generic post-processing requires `[u', v']`

## Summary

`envs/cases/lc_n7_onlyV` deliberately trains and evaluates its actor on the
wall-normal fluctuation `v'` only. That is the correct policy contract:
`NFLDC=1`, `runner.npl_state=1`, the Python MPI exchange contains one field,
and the exported `.pol` network has one input.

The embedded recorder originally wrote that same one-field array directly to
`drlrec`. `post_processing/read_drlrec.py` correctly honours the `nfld` value
in the file header, but the generic plot products in
`post_processing/postlib/drlrec.py` intentionally operate on the conventional
two-dimensional `[u', v']` observation plane. They therefore fail when they
index the missing second field of a v-only record.

## Root cause

Before this change, one Fortran array represented two different contracts:

```text
val_obs(1,:) = v'  -> actor input, Python/MPI state, drlrec payload
```

Using `NFLDC=2` to satisfy the recorder would be wrong. The embedded policy
loader rejects a `.pol` input dimension different from `NFLDC`, and the
v-only checkpoint plus Python/MPI protocol would no longer agree on the
number of state fields.

## Implemented design

The case now keeps those contracts separate:

```text
flow field at sensing points
        |
        +-- val_rec_obs(1:2,:) = [u', v']  -> drlrec diagnostics
        |
        +-- val_obs(1,:)      = v'         -> actor and MPI only
```

`NRECFLD=2` is a recorder-only parameter in `inc_src/DRL`. On every sensing
update, `drl_state.f` interpolates the streamwise and wall-normal
fluctuations at the existing agent sensing points, then copies the recorded
`v'` value into the one-field policy state. `pol_IO.f` writes `NRECFLD` and
`val_rec_obs` into the existing self-describing `NEKPOLR2` record format.

Consequently, a new v-only run has a normal two-component `drlrec` payload
ordered `[u', v']`, so the shared post-processing code works unchanged. The
second record component is exactly the same sampled `v'` passed to the model;
the first is a diagnostic value only. No additional MPI message, policy
input, or model weight is introduced.

## Compatibility and limitations

- Existing v-only `drlrec` files still contain only `v'`. Real `u'` values
  cannot be recovered without rerunning; zero-filling or duplicating `v'`
  would produce misleading two-dimensional plots.
- The binary magic remains `NEKPOLR2`: its header already records `nfld`, and
  the generic reader uses that value to determine record size.
- A two-field record is deliberately a *diagnostic representation*, not a
  statement that the v-only policy changed to two inputs. Configurations
  must continue to use `runner.npl_state: 1` and a one-input exported policy.

## Verification

Build the raw solver and run an embedded evaluation:

```bash
cd envs/cases/lc_n7_onlyV
./compile_script --solver raw --all
python3 ../../../post_processing/read_drlrec.py \
  <run>/eval/env_001
```

The reader summary should report `nfld: 2`; the first component is `u'` and
the second is `v'`. The policy loader should still report that it expects one
input and that `NFLDC = 1`.
