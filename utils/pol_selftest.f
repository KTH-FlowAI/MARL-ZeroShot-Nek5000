c==============================================
c Offline self-test for the embedded policy core (drl/pol_core.f).
c
c Links pol_core.f ONLY -- no Nek5000, no MPI. It reads the .pol network
c and the .chk reference table written by utils/sb3_to_f77.py, replays
c every row through pol_eval in both precisions, and reports how far the
c Fortran actor is from Stable-Baselines3's own predict().
c
c   pol_selftest <file.pol> <file.chk> [bits.out]
c
c The optional third argument dumps the IEEE-754 bit pattern of every
c weight so the caller can prove the ASCII round-trip is exact.
c
c Yuning Wang
c==============================================
      program pol_selftest
      implicit none

      integer LNW, LNLAY, MAXIN
      parameter (LNW=64, LNLAY=6, MAXIN=64)

      integer ndim(0:LNLAY), iact(LNLAY), iperm(LNW)
      integer nlay, isq, ierr
      real*4  alow, ahigh, sobs, sact
      real*4  w(LNW,LNW,LNLAY), b(LNW,LNLAY)

      character*256 polf, chkf, bitf
      integer nrow, ncol, il, jl, kl, iu
      real*8  obs(MAXIN), a4, a8, aref
      real*8  d4, d8, d4max, d8max, d48max
      real*8  s4sq, s8sq, arng
      integer n4bit, n8bit
      real*4  r4, rref

c     EQUIVALENCE gives us the raw bit pattern without leaving F77.
      real*4    rbits
      integer*4 ibits
      equivalence (rbits,ibits)

      call getarg(1,polf)
      call getarg(2,chkf)
      call getarg(3,bitf)

      if (polf.eq.' ' .or. chkf.eq.' ') then
         write(6,*) 'usage: pol_selftest <file.pol> <file.chk> [bits]'
         stop 1
      endif

c---------------------------------------------
c     Load the network
c---------------------------------------------
      iu = 21
      call pol_read(iu,polf,LNW,LNLAY,
     $              nlay,ndim,iact,isq,alow,ahigh,sobs,sact,
     $              iperm,w,b,ierr)
      if (ierr.ne.0) then
         write(6,*) '[SELFTEST] pol_read FAILED, ierr=',ierr
         stop 2
      endif

      write(6,'(A)')     '---------------------------------------------'
      write(6,'(A,I3)')  '[SELFTEST] nlayer   = ',nlay
      write(6,'(A,8I5)') '[SELFTEST] dims     = ',(ndim(il),il=0,nlay)
      write(6,'(A,8I5)') '[SELFTEST] acts     = ',(iact(il),il=1,nlay)
      write(6,'(A,I3,2E15.7)')
     $                   '[SELFTEST] squash   = ',isq,alow,ahigh
      write(6,'(A,2E15.7)')
     $                   '[SELFTEST] scalings = ',sobs,sact
      write(6,'(A,8I5)') '[SELFTEST] obs perm = ',
     $                   (iperm(il),il=1,ndim(0))
      write(6,'(A)')     '---------------------------------------------'

c---------------------------------------------
c     Optional: dump weight bit patterns
c---------------------------------------------
      if (bitf.ne.' ') then
         open(23,file=bitf,status='unknown')
         do il=1,nlay
            do jl=1,ndim(il)
               do kl=1,ndim(il-1)
                  rbits = w(jl,kl,il)
                  write(23,'(A,3I5,1X,I12)') 'W',il,jl,kl,ibits
               enddo
            enddo
            do jl=1,ndim(il)
               rbits = b(jl,il)
               write(23,'(A,2I5,1X,I12)') 'B',il,jl,ibits
            enddo
         enddo
         close(23)
         write(6,'(A,A)') '[SELFTEST] weight bits -> ',bitf(1:60)
      endif

c---------------------------------------------
c     Replay the reference table
c---------------------------------------------
      iu = 22
      open(iu,file=chkf,status='old',err=900)
      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) nrow, ncol
      if (ncol.ne.ndim(0)) then
         write(6,*) '[SELFTEST] input width mismatch:',ncol,ndim(0)
         stop 3
      endif

      d4max  = 0.0d0
      d8max  = 0.0d0
      d48max = 0.0d0
      s4sq   = 0.0d0
      s8sq   = 0.0d0
      n4bit  = 0
      n8bit  = 0

      do il=1,nrow
         call pol_skipc(iu,ierr)
         if (ierr.ne.0) goto 901
         read(iu,*,err=901,end=901) (obs(kl),kl=1,ncol), aref

         call pol_eval(LNW,LNLAY,nlay,ndim,iact,isq,
     $                 alow,ahigh,sobs,sact,iperm,w,b,4,obs,a4)
         call pol_eval(LNW,LNLAY,nlay,ndim,iact,isq,
     $                 alow,ahigh,sobs,sact,iperm,w,b,8,obs,a8)

         d4 = abs(a4-aref)
         d8 = abs(a8-aref)
         if (d4.gt.d4max) d4max = d4
         if (d8.gt.d8max) d8max = d8
         if (abs(a4-a8).gt.d48max) d48max = abs(a4-a8)
         s4sq = s4sq + d4*d4
         s8sq = s8sq + d8*d8

         rref = real(aref,4)
         r4   = real(a4,4)
         if (r4.eq.rref) n4bit = n4bit + 1
         r4 = real(a8,4)
         if (r4.eq.rref) n8bit = n8bit + 1
      enddo
      close(iu)

      s4sq = sqrt(s4sq/dble(nrow))
      s8sq = sqrt(s8sq/dble(nrow))
      arng = dble(abs(sact*(ahigh-alow)))

      write(6,'(A,I8)')    '[SELFTEST] rows compared      = ',nrow
      write(6,'(A,E12.4)') '[SELFTEST] action full scale  = ',arng
      write(6,'(A,E12.4)') '[SELFTEST] f32 max |dact|     = ',d4max
      write(6,'(A,E12.4)') '[SELFTEST] f32 rms |dact|     = ',s4sq
      write(6,'(A,E12.4)') '[SELFTEST] f64 max |dact|     = ',d8max
      write(6,'(A,E12.4)') '[SELFTEST] f64 rms |dact|     = ',s8sq
      write(6,'(A,E12.4)') '[SELFTEST] f32 vs f64 max     = ',d48max
      write(6,'(A,E12.4)') '[SELFTEST] f32 max/fullscale  = ',d4max/arng
      write(6,'(A,E12.4)') '[SELFTEST] f64 max/fullscale  = ',d8max/arng
      write(6,'(A,I8,A,I8)')
     $     '[SELFTEST] f32 bit-exact rows = ',n4bit,' /',nrow
      write(6,'(A,I8,A,I8)')
     $     '[SELFTEST] f64 bit-exact rows = ',n8bit,' /',nrow
      write(6,'(A)')     '---------------------------------------------'

      stop 0

 900  write(6,*) '[SELFTEST] cannot open ',chkf
      stop 4
 901  write(6,*) '[SELFTEST] error reading ',chkf
      stop 5
      end
