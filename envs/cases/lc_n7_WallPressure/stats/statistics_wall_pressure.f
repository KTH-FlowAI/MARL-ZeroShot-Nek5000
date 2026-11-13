c-----------------------------------------------------------------------
      subroutine wall_pressure_stat_compute(p2,scratch,work,
     $                                      lnvar,alpha,beta)
c [WALL_PRESSURE] Append variance as field 45, after original KTH moments.
c For RMS, finish any spatial/time averaging of field 45 BEFORE sqrt.
      implicit none
      include 'SIZE'
      include 'STATD'
      real p2(lx1,ly1,lz1,lelt),scratch(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt),alpha,beta
      integer lnvar,ntot
      ntot=lx1*ly1*lz1*nelv
      call wall_pressure_squared(p2,scratch,work)
      if (stat_rdim.eq.1) then
         call stat_reshufflev(work,p2,nelv)
      else
         call copy(work,p2,ntot)
      endif
      lnvar=lnvar+1
      call stat_compute_1Dav1(work,lnvar,alpha,beta)
      return
      end
