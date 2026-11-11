c==============================================
c Embedded deterministic policy: control driver
c
c Replaces the MPI request loop of DRL_main with a local scheduler. The
c point of this file is that it reproduces the coupled mode's timing
c EXACTLY, because the drag reduction depends on it.
c
c In the coupled mode the sequence per userchk call is driven by the
c requests Python sends. Tracing it through DRL_main gives:
c
c   ISTEP = 0            nek_init calls userchk once: set-up only
c   ISTEP = 1            STATE, CNTRL and EVOLV are all consumed in this
c                        one call, then drl_reward(1) runs BEFORE the
c                        next solve, i.e. on the pre-actuation field
c   ISTEP = 2 .. ndrl    one pass each, i_evolv = 2 .. ndrl
c   ISTEP = ndrl         i_evolv = ndrl, the reward buffers ship
c   ISTEP = ndrl+1       the cycle restarts with a fresh STATE
c
c so the control updates land on ISTEP = 1, ndrl+1, 2*ndrl+1, ... and
c the reward moving average covers samples 1..ndrl of each cycle. That
c is exactly what the scheduler below reproduces:
c
c   i_evolv = mod(ISTEP-1, ndrl) + 1,  update when i_evolv == 1
c
c With the raw (unpatched) Nek5000 core there is no episode restart, so
c ISTEP simply runs 1..nsteps and pol_step0 stays 0.
c
c Yuning Wang
c==============================================


c------------------------------------------------------------------
      subroutine POL_main
c Embedded-mode entry point, called from DRL_main when PARAM(91) = 1.
c (v19/channel uses UPARAM(10); this solver has no UPARAM.)
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      integer ndrl, kstp, i_evolv, ip
      logical ifupd(LPOL)

c---------------------------------------------
c     Set-up (ISTEP = 0 comes from nek_init)
c---------------------------------------------
      if (ISTEP.eq.0) then
         if (NID.eq.0) then
            write(6,*) '==========================================='
            write(6,*) '[POLICY] EMBEDDED MODE: no Python, no MPI'
            write(6,*) '[POLICY] ndrl (PARAM 89) =',nint(PARAM(89))
            write(6,*) '==========================================='
         endif
c        Same initialisation as the coupled path: wall points, sorting,
c        sensing plane and the findpts registration.
         call drl_init
c        Then the actors and the agent -> region map.
         call pol_init
c        Finally the per-rank trajectory recorder.
         call pol_rec_open
         pol_step0 = ISTEP
         pol_icyc  = 0
         return
      endif

      if (.not.pol_ifinit) return

c---------------------------------------------
c     Control schedule
c---------------------------------------------
c     v17: the DRL cadence lives in PARAM(89) of the .rea; this solver
c     has no UPARAM at all (see drl_main.f / drl_reward.f).
      ndrl = nint(PARAM(89))
      if (ndrl.lt.1) ndrl = 1

      kstp    = ISTEP - pol_step0
      i_evolv = mod(kstp-1,ndrl) + 1

      if (i_evolv.eq.1) then
         pol_icyc = pol_icyc + 1

c        Per-policy update interval, mirroring
c        MetaPolicy._partial_predict: region ip refreshes on the first
c        interaction and then every pol_nupd(ip) interactions; in
c        between it keeps sending its held action.
         do ip=1,pol_npol
            ifupd(ip) = (pol_icyc.eq.1) .or.
     $                  (mod(pol_icyc,pol_nupd(ip)).eq.0)
         enddo

c        STATE: identical to the coupled drl_state minus the MPI send.
         call sensing_pts_compute

c        CNTRL: evaluate the actors locally and impose the actuation.
         call pol_control(ifupd)
      endif

c---------------------------------------------
c     REWARD: untouched, every step, as in the coupled mode.
c     drl_reward_out skips its MPI sends when pol_ifsolo is set.
c---------------------------------------------
      call drl_reward(i_evolv)

c---------------------------------------------
c     RECORD: at the close of a cycle, where the observation, the action
c     and the completed reward average are all simultaneously valid.
c---------------------------------------------
      if (i_evolv.eq.ndrl) call pol_rec_write

c---------------------------------------------
c     Stop once the interaction budget is spent. lastep is used rather
c     than exitt0 so the solver leaves through its normal end-of-run
c     path (comment/prepost/nek_end) instead of being killed.
c---------------------------------------------
      if (pol_nbint.gt.0 .and. pol_icyc.ge.pol_nbint
     $    .and. i_evolv.eq.ndrl) then
         lastep = 1
         if (NID.eq.0) then
            write(6,*) '[POLICY] interaction budget reached:',
     $                 pol_icyc,' cycles, stopping'
         endif
      endif

c     lastep is also set by nek_solve when it reaches nsteps, so this
c     covers both the budget stop above and a natural end of run.
      if (lastep.eq.1) call pol_rec_close

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_control(ifupd)
c Embedded counterpart of drl_action: build this rank's action buffer
c from the local networks, scatter it onto the ACTIONS field, then apply
c the same zero-net-mass-flux treatment the coupled path uses.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'TSTEP'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      logical ifupd(LPOL)
      real    act_buffer(totctrl)
      integer i_znmf

      call pol_actions(act_buffer,ifupd)

      call apply_actions(act_buffer)

      i_znmf = nint(PARAM(90))
      if (i_znmf.gt.0) then
c        The wing's drl_action calls znmf_avg only (no znmf_check).
         call znmf_avg()
      endif

      return
      end
c------------------------------------------------------------------
