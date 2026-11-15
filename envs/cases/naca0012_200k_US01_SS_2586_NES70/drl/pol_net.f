c==============================================
c Embedded deterministic policy: Nek5000 side
c
c Wraps the include-free arithmetic in drl/pol_core.f with the solver's
c common blocks: reads the run configuration, loads one actor per
c control region, maps agents onto regions, and evaluates the networks
c locally on every rank.
c
c No MPI is used for the policy itself. The actor is pointwise in the
c agent index, so each rank evaluates its own agents from val_obs and
c writes its own act_buffer -- the observation gather, the action
c scatter and the Python round trip all disappear. The only collective
c here is the one-off broadcast of the weights at initialisation.
c
c Yuning Wang
c==============================================


c------------------------------------------------------------------
      subroutine pol_init
c Load drl_policy.in and every actor it references, then map the local
c agents onto control regions. Call once, after drl_init, so that
c pos_agt/NUMCTRL are already populated.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      integer ierr

      pol_ifinit = .false.
      pol_icyc   = 0
      pol_step0  = 0

      call pol_cfg_read(ierr)
      call pol_chkerr(ierr,'reading drl_policy.in')

      call pol_load_all(ierr)
      call pol_chkerr(ierr,'loading the .pol networks')

      call pol_bcast

      call pol_assign

      call pol_banner

      pol_ifinit = .true.

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_chkerr(ierr,what)
c Abort every rank on a configuration failure. Getting this wrong is a
c setup error, not a recoverable one: continuing would silently run an
c uncontrolled or wrongly scaled simulation.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      integer ierr, ierrg, iglmax
      character*(*) what

      ierrg = iglmax(ierr,1)
      if (ierrg.ne.0) then
         if (NID.eq.0) then
            write(6,*) '[POLICY] FATAL: ',what
            write(6,*) '[POLICY] ierr = ',ierrg
         endif
         call exitt
      endif

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_cfg_read(ierr)
c Rank-0 read of drl_policy.in. Layout (comments start with '#',
c everything else is list-directed):
c
c   nb_interactions  rec_freq  rec_bufsize  iprec
c   reward_mode  dudy_ref  alpha  beta  gamma
c   npol
c   then npol lines:  xmin xmax iside utau amp nupd 'file.pol'
c
c   iside: 0 = any, 1 = suction side (y>0), 2 = pressure side (y<0)
c
c The file is generated from the run YAML by the prepare step; it is not
c meant to be hand-edited.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'POLICY'

      integer ierr, iu, il
      real    r_utau, r_amp
      logical ifexist
      character*32 fcfg
      parameter (fcfg='drl_policy.in')

      ierr = 0
      if (NID.ne.0) return

      inquire(file=fcfg,exist=ifexist)
      if (.not.ifexist) then
         write(6,*) '[POLICY] cannot find ',fcfg
         ierr = 1
         return
      endif

      iu = 31
      open(iu,file=fcfg,status='old',err=900)

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) pol_nbint, pol_recf, pol_recb,
     $                           pol_iprec

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) pol_rwmode, pol_dudyref,
     $                           pol_alpha, pol_beta, pol_gamma

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) pol_npol
      if (pol_npol.lt.1 .or. pol_npol.gt.LPOL) then
         write(6,*) '[POLICY] npol out of range: ',pol_npol,' max ',LPOL
         ierr = 2
         goto 999
      endif

      do il=1,pol_npol
         call pol_skipc(iu,ierr)
         if (ierr.ne.0) goto 901
         read(iu,*,err=901,end=901) pol_xmin(il), pol_xmax(il),
     $        pol_side(il), r_utau, r_amp, pol_nupd(il), pol_file(il)
         pol_sobs(il) = real(r_utau,4)
         pol_sact(il) = real(r_amp,4)
         if (pol_nupd(il).lt.1) then
            write(6,*) '[POLICY] policy',il,' has nupd < 1'
            ierr = 3
            goto 999
         endif
      enddo

      if (pol_iprec.ne.4 .and. pol_iprec.ne.8) then
         write(6,*) '[POLICY] iprec must be 4 or 8, got ',pol_iprec
         ierr = 4
         goto 999
      endif

 999  close(iu)
      return

 900  write(6,*) '[POLICY] cannot open ',fcfg
      ierr = 5
      return
 901  write(6,*) '[POLICY] parse error in ',fcfg
      ierr = 6
      close(iu)
      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_load_all(ierr)
c Rank-0 read of every .pol referenced by the configuration. The
c scalings in drl_policy.in win over the ones baked into the .pol at
c export time; a disagreement is reported rather than applied silently,
c because it usually means the eval config and the training config have
c drifted apart.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      integer ierr, il, ie
      real*4  f_sobs, f_sact

      ierr = 0
      if (NID.ne.0) return

      do il=1,pol_npol
         call pol_read(32,pol_file(il),LNW,LNLAY,
     $                 pol_nlay(il),pol_dim(0,il),pol_act(1,il),
     $                 pol_isq(il),pol_alow(il),pol_ahigh(il),
     $                 f_sobs,f_sact,pol_perm(1,il),
     $                 pol_w(1,1,1,il),pol_b(1,1,il),ie)
         if (ie.ne.0) then
            write(6,*) '[POLICY] pol_read failed for ',
     $                 pol_file(il)(1:60),' ierr=',ie
            ierr = 10 + ie
            return
         endif

         if (abs(f_sobs-pol_sobs(il)).gt.1.0e-6*abs(f_sobs)) then
            write(6,*) '[POLICY] WARNING policy',il,
     $           ' u_tau differs: .pol=',f_sobs,' run=',pol_sobs(il)
         endif
         if (abs(f_sact-pol_sact(il)).gt.1.0e-6*abs(f_sact)) then
            write(6,*) '[POLICY] WARNING policy',il,
     $           ' amp differs: .pol=',f_sact,' run=',pol_sact(il)
         endif

         if (pol_dim(0,il).ne.NFLDC) then
            write(6,*) '[POLICY] policy',il,' expects',pol_dim(0,il),
     $           ' inputs but NFLDC =',NFLDC
            ierr = 7
            return
         endif
         if (pol_dim(pol_nlay(il),il).ne.1) then
            write(6,*) '[POLICY] policy',il,' must have 1 output, got',
     $           pol_dim(pol_nlay(il),il)
            ierr = 8
            return
         endif
      enddo

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_bcast
c Push the configuration and the weights from rank 0 to every rank.
c One-off, at initialisation; the control loop itself is communication
c free. bcast() takes a length in bytes.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'POLICY'

c     One call per member on purpose. Broadcasting whole common blocks by
c     a computed byte count is a standing invitation to silent corruption
c     the next time somebody adds a field, and it cannot be checked by
c     eye. Init happens once, so the extra calls cost nothing.
c
c     Deliberately NOT broadcast: pol_id and pol_hold are per-rank state
c     built locally in pol_assign, and the logicals plus the scheduler
c     counters are set identically on every rank by pol_init/POL_main.

c     ---- run-level scalars
      call bcast(pol_npol  ,ISIZE)
      call bcast(pol_iprec ,ISIZE)
      call bcast(pol_nbint ,ISIZE)
      call bcast(pol_recf  ,ISIZE)
      call bcast(pol_recb  ,ISIZE)
      call bcast(pol_rwmode,ISIZE)

c     ---- reward normalisation
      call bcast(pol_dudyref,WDSIZE)
      call bcast(pol_alpha  ,WDSIZE)
      call bcast(pol_beta   ,WDSIZE)
      call bcast(pol_gamma  ,WDSIZE)

c     ---- per-policy network description
      call bcast(pol_nlay,LPOL*ISIZE)
      call bcast(pol_dim ,(LNLAY+1)*LPOL*ISIZE)
      call bcast(pol_act ,LNLAY*LPOL*ISIZE)
      call bcast(pol_isq ,LPOL*ISIZE)
      call bcast(pol_perm,LNW*LPOL*ISIZE)

c     ---- per-policy region and cadence
      call bcast(pol_xmin,LPOL*WDSIZE)
      call bcast(pol_xmax,LPOL*WDSIZE)
      call bcast(pol_side,LPOL*ISIZE)
      call bcast(pol_nupd,LPOL*ISIZE)

c     ---- scalings and weights, stored as exact float32
      call bcast(pol_alow ,LPOL*4)
      call bcast(pol_ahigh,LPOL*4)
      call bcast(pol_sobs ,LPOL*4)
      call bcast(pol_sact ,LPOL*4)
      call bcast(pol_w    ,LNW*LNW*LNLAY*LPOL*4)
      call bcast(pol_b    ,LNW*LNLAY*LPOL*4)

c     ---- provenance
      call bcast(pol_file,128*LPOL)

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_assign
c Map each local agent onto a control region, mirroring
c MetaPolicy._distribute_agents: x within [xmin,xmax] and, for a wing,
c the sign of y selecting suction or pressure side. First match wins.
c Agents matching nothing keep pol_id = 0 and are never actuated.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      integer il, ip, nass, nuna, iglsum
      real    xa, ya
      logical ifmatch

      call izero(pol_id,totctrl)
      call rzero(pol_hold,totctrl)

      nass = 0
      do il=1,NUMCTRL
         xa = pos_agt(1,il)
         ya = pos_agt(2,il)
         do ip=1,pol_npol
            if (pol_id(il).ne.0) goto 100
            ifmatch = (xa.ge.pol_xmin(ip)) .and. (xa.le.pol_xmax(ip))
            if (ifmatch) then
               if (pol_side(ip).eq.POL_S_SS) then
                  ifmatch = ya.gt.0.0
               elseif (pol_side(ip).eq.POL_S_PS) then
                  ifmatch = ya.lt.0.0
               endif
            endif
            if (ifmatch) then
               pol_id(il) = ip
               nass = nass + 1
            endif
         enddo
 100     continue
      enddo

      nuna = iglsum(NUMCTRL-nass,1)
      nass = iglsum(nass,1)
      if (NID.eq.0) then
         write(6,*) '[POLICY] agents assigned  =',nass
         if (nuna.gt.0) write(6,*)
     $      '[POLICY] WARNING agents with no policy =',nuna
      endif

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_actions(act_buffer,ifupd)
c Fill act_buffer(1:NUMCTRL) with this rank's actions.
c
c   ifupd(ip) = .true.  re-evaluate policy ip and refresh its hold
c             = .false. reuse the held action
c
c The hold reproduces MetaPolicy._partial_predict, where a region whose
c update interval has not elapsed keeps sending its previous action.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      real    act_buffer(totctrl)
      logical ifupd(LPOL)

      integer il, ip, kl, nin
      real    obs(LNW), acti

      call rzero(act_buffer,totctrl)

      do il=1,NUMCTRL
         ip = pol_id(il)
         if (ip.eq.0) goto 200

         if (ifupd(ip)) then
            nin = pol_dim(0,ip)
            do kl=1,nin
               obs(kl) = val_obs(kl,il)
            enddo
            call pol_eval(LNW,LNLAY,pol_nlay(ip),pol_dim(0,ip),
     $                    pol_act(1,ip),pol_isq(ip),
     $                    pol_alow(ip),pol_ahigh(ip),
     $                    pol_sobs(ip),pol_sact(ip),pol_perm(1,ip),
     $                    pol_w(1,1,1,ip),pol_b(1,1,ip),
     $                    pol_iprec,obs,acti)
            pol_hold(il) = acti
         endif

         act_buffer(il) = pol_hold(il)
 200     continue
      enddo

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_banner
c One-off summary so a log makes it obvious which networks were flown.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'POLICY'

      integer il, jl

      if (NID.ne.0) return

      write(6,*) '==========================================='
      write(6,*) '[POLICY] EMBEDDED ACTOR MODE'
      write(6,*) '==========================================='
      write(6,*) '[POLICY] policies      =',pol_npol
      write(6,*) '[POLICY] accumulation  =',pol_iprec,' byte real'
      write(6,*) '[POLICY] interactions  =',pol_nbint
      write(6,*) '[POLICY] record freq   =',pol_recf
      write(6,*) '[POLICY] reward mode   =',pol_rwmode
      write(6,*) '[POLICY] dUdy ref      =',pol_dudyref
      write(6,*) '[POLICY] a/b/g         =',pol_alpha,pol_beta,pol_gamma
      do il=1,pol_npol
         write(6,*) '-------------------------------------------'
         write(6,*) '[POLICY]',il,' : ',pol_file(il)(1:60)
         write(6,*) '[POLICY]   dims     =',
     $        (pol_dim(jl,il),jl=0,pol_nlay(il))
         write(6,*) '[POLICY]   acts     =',
     $        (pol_act(jl,il),jl=1,pol_nlay(il))
         write(6,*) '[POLICY]   x range  =',pol_xmin(il),pol_xmax(il)
         write(6,*) '[POLICY]   side     =',pol_side(il)
         write(6,*) '[POLICY]   nupd     =',pol_nupd(il)
         write(6,*) '[POLICY]   u_tau    =',pol_sobs(il)
         write(6,*) '[POLICY]   amp      =',pol_sact(il)
         write(6,*) '[POLICY]   squash   =',pol_isq(il),
     $                                      pol_alow(il),pol_ahigh(il)
         write(6,*) '[POLICY]   obs perm =',
     $        (pol_perm(jl,il),jl=1,pol_dim(0,il))
      enddo
      write(6,*) '==========================================='

      return
      end
c------------------------------------------------------------------
