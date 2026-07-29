c==============================================
c Embedded deterministic policy: self-contained core
c
c This file has NO includes and NO common blocks, so it compiles and runs
c outside Nek5000. That is deliberate: the offline unit test links this
c file alone and replays the reference table produced by
c utils/sb3_to_f77.py, which validates the network arithmetic before any
c solver is involved.
c
c All floating-point kinds are declared explicitly (real*4 / real*8) so
c that the -fdefault-real-8 flag makenek passes cannot silently change
c the arithmetic.
c
c Activation codes (shared with utils/sb3_to_f77.py):
c   0 = identity   1 = relu   2 = tanh
c
c Yuning Wang
c==============================================


c------------------------------------------------------------------
      subroutine pol_skipc(iu,ierr)
c Advance `iu` to the next line that is neither blank nor a '#' comment,
c then step back so the caller's list-directed READ starts on it.
c   ierr = 0 ok / 1 end-of-file / 2 read error
c------------------------------------------------------------------
      implicit none
      integer iu, ierr, i
      character*512 line

      ierr = 0
 10   read(iu,'(A)',end=90,err=91) line
      do i=1,512
         if (line(i:i).ne.' ') then
            if (line(i:i).eq.'#') goto 10
            backspace(iu)
            return
         endif
      enddo
      goto 10                        ! blank line, keep looking

 90   ierr = 1
      return
 91   ierr = 2
      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_read(iu,fname,lnw,lnlay,
     $                    nlay,ndim,iact,isq,alow,ahigh,sobs,sact,
     $                    iperm,w,b,ierr)
c Read one `.pol` file written by utils/sb3_to_f77.py.
c
c   iu     [in]  unit number to use
c   fname  [in]  file name
c   lnw    [in]  max neurons per layer  (leading dim of w/b)
c   lnlay  [in]  max linear layers
c   nlay   [out] number of linear layers
c   ndim   [out] layer widths, ndim(0) is the input width
c   iact   [out] activation code applied after each layer
c   isq    [out] 1 = apply the SB3 unscale_action mapping
c   alow   [out] action-space bounds used by that mapping
c   ahigh  [out]
c   sobs   [out] observation divisor (u_tau)
c   sact   [out] action multiplier   (ctrl_max_amp)
c   iperm  [out] observation permutation: input k of the network is fed
c                obs(iperm(k)). Identity for a Nek-trained policy; a
c                Dedalus-trained one wants the components reversed, which
c                MetaPolicy._obs_solver_arrange does with np.flip.
c   w,b    [out] weights (row-major: w(out,in,layer)) and biases
c   ierr   [out] 0 ok, >0 failure
c------------------------------------------------------------------
      implicit none
      integer iu, lnw, lnlay, nlay, isq, ierr
      integer ndim(0:lnlay), iact(lnlay), iperm(lnw)
      real*4  alow, ahigh, sobs, sact
      real*4  w(lnw,lnw,lnlay), b(lnw,lnlay)
      character*(*) fname

      integer il, jl, kl, nin, nout, ivers
      logical ifexist

      ierr = 0
      inquire(file=fname,exist=ifexist)
      if (.not.ifexist) then
         ierr = 10
         return
      endif

      open(iu,file=fname,status='old',form='formatted',err=900)

c     Format version, so the layout can be extended without guessing.
      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) ivers
      if (ivers.ne.2) then
         ierr = 16
         goto 999
      endif

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) nlay
      if (nlay.lt.1 .or. nlay.gt.lnlay) then
         ierr = 11
         goto 999
      endif

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) (ndim(il),il=0,nlay)
      do il=0,nlay
         if (ndim(il).lt.1 .or. ndim(il).gt.lnw) then
            ierr = 12
            goto 999
         endif
      enddo

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) (iact(il),il=1,nlay)

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) isq, alow, ahigh

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) sobs, sact
      if (sobs.eq.0.0e0) then
         ierr = 13
         goto 999
      endif

      call pol_skipc(iu,ierr)
      if (ierr.ne.0) goto 901
      read(iu,*,err=901,end=901) (iperm(kl),kl=1,ndim(0))
      do kl=1,ndim(0)
         if (iperm(kl).lt.1 .or. iperm(kl).gt.ndim(0)) then
            ierr = 17
            goto 999
         endif
      enddo

      do il=1,nlay
         nin  = ndim(il-1)
         nout = ndim(il)
         call pol_skipc(iu,ierr)
         if (ierr.ne.0) goto 901
         read(iu,*,err=901,end=901)
     $        ((w(jl,kl,il),kl=1,nin),jl=1,nout)
         call pol_skipc(iu,ierr)
         if (ierr.ne.0) goto 901
         read(iu,*,err=901,end=901) (b(jl,il),jl=1,nout)
      enddo

 999  close(iu)
      return

 900  ierr = 14
      return
 901  ierr = 15
      close(iu)
      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_eval4(lnw,lnlay,nlay,ndim,iact,isq,
     $                     alow,ahigh,sobs,sact,iperm,w,b,obs,act)
c Single-agent forward pass in float32, mirroring the torch actor.
c
c The chain reproduced here is, end to end:
c     o32 = real( obs / sobs , 4)          <- divide in f64, then round
c     h   = act( W h + b )                 <- float32 throughout
c     a   = alow + 0.5*(a+1)*(ahigh-alow)  <- SB3 unscale_action, if isq=1
c     act = real( a * sact , 8)
c
c The unscale step looks like an identity for alow=-1, ahigh=1 but is NOT
c one in float32: (a+1) rounds to a multiple of 2**-23, so SB3's actions
c are quantised at ~1.2e-7. Reproducing it is required for agreement.
c------------------------------------------------------------------
      implicit none
      integer lnw, lnlay, nlay, isq
      integer ndim(0:lnlay), iact(lnlay), iperm(lnw)
      real*4  alow, ahigh, sobs, sact
      real*4  w(lnw,lnw,lnlay), b(lnw,lnlay)
      real*8  obs(*), act

      real*4  x(1024), y(1024), s
      integer il, jl, kl, nin, nout

      nin = ndim(0)
      do kl=1,nin
         x(kl) = real(obs(iperm(kl))/dble(sobs),4)
      enddo

      do il=1,nlay
         nin  = ndim(il-1)
         nout = ndim(il)
         do jl=1,nout
            s = b(jl,il)
            do kl=1,nin
               s = s + w(jl,kl,il)*x(kl)
            enddo
            if (iact(il).eq.1) then
               if (s.lt.0.0e0) s = 0.0e0
            elseif (iact(il).eq.2) then
               s = tanh(s)
            endif
            y(jl) = s
         enddo
         do jl=1,nout
            x(jl) = y(jl)
         enddo
      enddo

      s = x(1)
      if (isq.eq.1) s = alow + (0.5e0*(s+1.0e0)*(ahigh-alow))
      act = dble(s*sact)

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_eval8(lnw,lnlay,nlay,ndim,iact,isq,
     $                     alow,ahigh,sobs,sact,iperm,w,b,obs,act)
c As pol_eval4 but accumulating in float64. The stored weights are still
c the exact float32 values from the checkpoint; only the arithmetic
c differs. Used for A/B testing the effect of accumulation precision.
c------------------------------------------------------------------
      implicit none
      integer lnw, lnlay, nlay, isq
      integer ndim(0:lnlay), iact(lnlay), iperm(lnw)
      real*4  alow, ahigh, sobs, sact
      real*4  w(lnw,lnw,lnlay), b(lnw,lnlay)
      real*8  obs(*), act

      real*8  x(1024), y(1024), s
      integer il, jl, kl, nin, nout

      nin = ndim(0)
      do kl=1,nin
         x(kl) = obs(iperm(kl))/dble(sobs)
      enddo

      do il=1,nlay
         nin  = ndim(il-1)
         nout = ndim(il)
         do jl=1,nout
            s = dble(b(jl,il))
            do kl=1,nin
               s = s + dble(w(jl,kl,il))*x(kl)
            enddo
            if (iact(il).eq.1) then
               if (s.lt.0.0d0) s = 0.0d0
            elseif (iact(il).eq.2) then
               s = tanh(s)
            endif
            y(jl) = s
         enddo
         do jl=1,nout
            x(jl) = y(jl)
         enddo
      enddo

      s = x(1)
      if (isq.eq.1) s = dble(alow)
     $                + (0.5d0*(s+1.0d0)*dble(ahigh-alow))
      act = s*dble(sact)

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine pol_eval(lnw,lnlay,nlay,ndim,iact,isq,
     $                    alow,ahigh,sobs,sact,iperm,w,b,iprec,obs,act)
c Dispatch on the requested accumulation precision (4 or 8).
c------------------------------------------------------------------
      implicit none
      integer lnw, lnlay, nlay, isq, iprec
      integer ndim(0:lnlay), iact(lnlay), iperm(lnw)
      real*4  alow, ahigh, sobs, sact
      real*4  w(lnw,lnw,lnlay), b(lnw,lnlay)
      real*8  obs(*), act

      if (iprec.eq.4) then
         call pol_eval4(lnw,lnlay,nlay,ndim,iact,isq,
     $                  alow,ahigh,sobs,sact,iperm,w,b,obs,act)
      else
         call pol_eval8(lnw,lnlay,nlay,ndim,iact,isq,
     $                  alow,ahigh,sobs,sact,iperm,w,b,obs,act)
      endif

      return
      end
c------------------------------------------------------------------
