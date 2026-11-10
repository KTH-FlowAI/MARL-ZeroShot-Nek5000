c==============================================
c Embedded deterministic policy: on-the-fly binary recorder
c
c Writes the (observation, action, reward) trajectory of every agent to a
c per-rank binary file, read back by post_processing/read_drlrec.py.
c
c Three design points, because they are easy to get wrong:
c
c 1. PER-RANK FILES, NO GATHER. A rank writes only its own agents. There
c    is no collective on the write path and no rank-0 serialisation
c    point, so the recorder stays off the critical path however many
c    ranks are used. The cost is moved to the reader, which merges.
c
c 2. PARTITION-INDEPENDENT IDENTITY. Which rank owns an agent, and in
c    what order, changes with nproc. The set of agents does not: it
c    follows from the mesh and the boundary conditions alone. So every
c    agent is stamped with (ieg, iface, ix, iy, iz) -- global element,
c    face, GLL indices -- which is invariant under repartitioning. The
c    reader sorts on that, so a run on 16 ranks lines up column for
c    column with a run on 10.
c
c 3. SEGMENTS, SO A RESUME NEVER OVERWRITES. Each run claims the next
c    free segment index rather than truncating what is already there.
c    A job killed on wall clock and resumed leaves s00000 intact and
c    writes s00001; the reader stitches them in time order.
c
c Files land in ./drlrec/ when that directory exists (the prepare step
c creates it), otherwise in the working directory with a warning.
c
c Why not TSRS: it re-interpolates onto its own point set, which is
c exactly what must not happen on a wing, where the agents already sit
c on curved-wall GLL nodes. Here the values are sampled at the controller's
c sensing points. The u-only channel case records [u',v'] for generic
c post-processing while its actor still consumes only val_obs(1)=u'.
c
c File layout (stream access, native endianness, marked in the header):
c
c   header
c     char*8  'NEKPOLR2'          magic + format version
c     int32   1234567890          endianness probe
c     int32   nid
c     int32   nagents             agents owned by THIS rank
c     int32   nfld                recorded observation components per agent
c     int32   nrwd                reward components per agent (3)
c     int32   ndrl                solver steps per control cycle
c     int32   rec_freq            cycles between records
c     int32   iseg                segment index of this run
c     real8   dt
c     real8   t0                  time of the first record
c     nagents * (int32 ieg, iface, ix, iy, iz, ipol,
c                real8 x, y, z)
c
c   record, repeated
c     real8   time
c     int32   istep
c     int32   icycle
c     real8   obs(nfld, nagents)
c     real8   act(nagents)
c     real8   rwd(nrwd, nagents)
c
c In dudy reward mode only the first reward slot is filled (rwd_agt); in
c net_gain mode the three slots are tau_w, |p'v| and 0.5|v^3|.
c
c Yuning Wang
c==============================================


c------------------------------------------------------------------
      subroutine pol_rec_probe(idir,iseg)
c Rank-0 only. Decide where the record files go and which segment index
c this run owns.
c
c   idir = 1  ./drlrec/ exists and is writable
c        = 0  fall back to the working directory
c   iseg      lowest index for which no rank-0 file exists yet
c
c The directory test is a trial open rather than INQUIRE, because
c INQUIRE(file=...) on a directory is not portable.
c------------------------------------------------------------------
      implicit none
      integer idir, iseg
      integer ios
      logical ifexist
      character*64 fname

      idir = 0
      iseg = 0

      open(69,file='drlrec/.polprobe',status='unknown',iostat=ios)
      if (ios.eq.0) then
         close(69,status='delete')
         idir = 1
      endif

c     Claim the first free segment. Only rank 0's file is probed: every
c     run has a rank 0, whatever nproc is.
      do iseg = 0, 99999
         call pol_rec_name(fname,idir,iseg,0)
         inquire(file=fname,exist=ifexist)
         if (.not.ifexist) return
      enddo

      iseg = 99999
      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_rec_coupled_init
c Enable the same per-rank binary recorder in a Python-coupled run when
c preparation wrote drl_record.in.  Unlike embedded mode this path never
c reads a policy file: Python remains the action source.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      integer ios, il
      logical ifexist
      character*32 fcfg
      parameter (fcfg='drl_record.in')

      pol_ifrec = .false.
      pol_recf = 0
      pol_recb = 0
      pol_icyc = 0

      inquire(file=fcfg,exist=ifexist)
      if (.not.ifexist) return

      open(33,file=fcfg,status='old',form='formatted',iostat=ios)
      if (ios.ne.0) then
         if (NID.eq.0) write(6,*) '[POLREC] cannot open ',fcfg
         return
      endif
      call pol_skipc(33,ios)
      if (ios.eq.0)
     $   read(33,*,iostat=ios) pol_recf,pol_recb,pol_rwmode
      close(33)
      if (ios.ne.0 .or. pol_recf.lt.1 .or. pol_recb.lt.1 .or.
     $    pol_rwmode.lt.0 .or. pol_rwmode.gt.1) then
         if (NID.eq.0) write(6,*) '[POLREC] invalid ',fcfg
         pol_recf = 0
         return
      endif

c     Coupled actions do not have an F77 policy-region map.  Keep the
c     identity field explicit (0 = Python-coupled / no embedded policy).
      do il=1,NUMCTRL
         pol_id(il) = 0
         pol_hold(il) = 0.0
      enddo
      call pol_rec_open

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_rec_name(fname,idir,iseg,inid)
c Build a record file name. Kept in one place so the writer and the
c segment probe can never disagree about it.
c------------------------------------------------------------------
      implicit none
      character*(*) fname
      integer idir, iseg, inid

      if (idir.eq.1) then
         write(fname,'(A,I5.5,A,I5.5,A)')
     $        'drlrec/drlrec_s', iseg, '_p', inid, '.bin'
      else
         write(fname,'(A,I5.5,A,I5.5,A)')
     $        'drlrec_s', iseg, '_p', inid, '.bin'
      endif

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_rec_open
c Create this rank's record file and write the header. Ranks that own no
c agents write nothing at all.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      integer il, ios
      integer nrwd, idir, iseg
      character*64 fname

      pol_ifrec = .false.
      pol_nrec  = 0
      pol_runit = 61

      if (pol_recf.le.0) return

c     Directory and segment are decided once, on rank 0, then shared:
c     every rank must agree on the segment or the reader sees a
c     half-written run.
      idir = 0
      iseg = 0
      if (NID.eq.0) call pol_rec_probe(idir,iseg)
      call bcast(idir,ISIZE)
      call bcast(iseg,ISIZE)

      if (NID.eq.0 .and. idir.eq.0) then
         write(6,*) '[POLREC] ./drlrec/ not writable, ',
     $              'writing into the run directory instead'
      endif

      if (NUMCTRL.le.0) return

      nrwd = 3
      call pol_rec_name(fname,idir,iseg,NID)

      open(pol_runit,file=fname,form='unformatted',access='stream',
     $     status='replace',iostat=ios)
      if (ios.ne.0) then
c        A missing trajectory is a lost diagnostic, not a reason to kill
c        the run: aborting one rank here would strand the others in the
c        next collective, i.e. deadlock the whole job.
         write(6,*) '[POLREC] cannot open ',fname,' iostat=',ios,
     $              ' -- recording disabled on rank',NID
         return
      endif

      write(pol_runit) 'NEKPOLR2'
      write(pol_runit) 1234567890
      write(pol_runit) NID
      write(pol_runit) NUMCTRL
      write(pol_runit) NRECFLD
      write(pol_runit) nrwd
      write(pol_runit) nint(UPARAM(1))
      write(pol_runit) pol_recf
      write(pol_runit) iseg
      write(pol_runit) DT
      write(pol_runit) TIME

c     The identity block. (ieg, iface, ix, iy, iz) is invariant under
c     repartitioning, so the reader can line up runs made with different
c     nproc; ipol records which policy governed the agent.
      do il=1,NUMCTRL
         write(pol_runit) info_agt(1,il), info_agt(2,il),
     $                    info_agt(3,il), info_agt(4,il),
     $                    info_agt(5,il), pol_id(il)
         write(pol_runit) pos_agt(1,il), pos_agt(2,il), pos_agt(3,il)
      enddo

      pol_ifrec = .true.

      if (NID.eq.0) then
         write(6,*) '[POLREC] segment',iseg,' -> ',fname
         write(6,*) '[POLREC] agents on rank 0 =',NUMCTRL
      endif

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_rec_write
c Append one record. Called at the END of a control cycle, where all
c three quantities are simultaneously valid:
c   val_rec_obs holds [u',v'] sampled with the observation that produced
c            this action (it is only refreshed at the next cycle start)
c   val_obs  remains the policy-only state and may have fewer components
c   pol_hold holds the action actually applied during the cycle
c   rwd_*    hold the completed moving average over the cycle
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'TSTEP'
      include 'PARALLEL'
      include 'DRL'
      include 'POLICY'

      integer il, kl

      if (.not.pol_ifrec) return
      if (mod(pol_icyc,pol_recf).ne.0) return

      write(pol_runit) TIME
      write(pol_runit) ISTEP
      write(pol_runit) pol_icyc

      write(pol_runit) ((val_rec_obs(kl,il),kl=1,NRECFLD),
     $                  il=1,NUMCTRL)
      write(pol_runit) (pol_hold(il),il=1,NUMCTRL)

      if (pol_rwmode.eq.1) then
         write(pol_runit) (rwd_tau(il),rwd_pw(il),rwd_v3(il),
     $                     il=1,NUMCTRL)
      else
c        dudy mode: only the first slot carries information.
         write(pol_runit) (rwd_agt(il),0.0d0,0.0d0,il=1,NUMCTRL)
      endif

      pol_nrec = pol_nrec + 1

c     Flush periodically so a job killed on wall clock still leaves a
c     readable file; the reader tolerates a partial trailing record.
      if (pol_recb.gt.0) then
         if (mod(pol_nrec,pol_recb).eq.0) call flush(pol_runit)
      endif

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_rec_close
c Final flush and close.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'POLICY'

      if (.not.pol_ifrec) return

      close(pol_runit)
      pol_ifrec = .false.

      if (NID.eq.0) then
         write(6,*) '[POLREC] closed after',pol_nrec,' records'
      endif

      return
      end
c------------------------------------------------------------------
