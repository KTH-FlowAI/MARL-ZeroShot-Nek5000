c-----------------------------------------------------------------------
      subroutine bewley_stat_compute(costp,costk,work,lnvar,alpha,beta)
c Append two scalar costs after the 66 original statistics. Reuse caller
c scratch after the filter terms; never average the DRL cycle buffers.
      implicit none
      include 'SIZE'
      include 'INPUT'
      real costp(lx1,ly1,lz1,lelt),costk(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt),alpha,beta
      integer lnvar,ntot
      ntot=lx1*ly1*lz1*nelv

      call bewley_cost_fields(costp,costk)
      if (if3d) then
         call stat_reshufflev(work,costp,nelv)
      else
         call copy(work,costp,ntot)
      endif
      lnvar=lnvar+1
      call stat_compute_1Dav1(work,lnvar,alpha,beta)

      if (if3d) then
         call stat_reshufflev(work,costk,nelv)
      else
         call copy(work,costk,ntot)
      endif
      lnvar=lnvar+1
      call stat_compute_1Dav1(work,lnvar,alpha,beta)
      return
      end
