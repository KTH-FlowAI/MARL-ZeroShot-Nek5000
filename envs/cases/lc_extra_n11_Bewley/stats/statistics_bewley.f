c-----------------------------------------------------------------------
      subroutine bewley_stat_compute(costp,costk,work,lnvar,alpha,beta)
c Append costs 45/46 after the 44 original KTH statistics. Reuse scratch
c only after all original moments have been evaluated.
      implicit none
      include 'SIZE'
      include 'STATD'
      real costp(lx1,ly1,lz1,lelt),costk(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt),alpha,beta
      integer lnvar,ntot
      ntot=lx1*ly1*lz1*nelv
      call bewley_cost_fields(costp,costk)
      if (stat_rdim.eq.1) then
         call stat_reshufflev(work,costp,nelv)
      else
         call copy(work,costp,ntot)
      endif
      lnvar=lnvar+1
      call stat_compute_1Dav1(work,lnvar,alpha,beta)
      if (stat_rdim.eq.1) then
         call stat_reshufflev(work,costk,nelv)
      else
         call copy(work,costk,ntot)
      endif
      lnvar=lnvar+1
      call stat_compute_1Dav1(work,lnvar,alpha,beta)
      return
      end
