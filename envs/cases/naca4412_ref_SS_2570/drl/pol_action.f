c------------------------------------------------------------------
      subroutine apply_actions(act_buffer)
c Scatter a per-rank action buffer onto the ACTIONS field and rebuild the
c actuation mask.  This routine is used by the embedded policy controller;
c coupled cases retain the same scatter through recv_Actions.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      include 'PARALLEL'
      include 'DRL'
      include 'SOLN'
      integer il, jl, ntot
      integer glbid,fceid,lclid
      integer ix,iy,iz
      real act_buffer(totctrl), act_i
      real act_buff(LX1,LY1,LZ1,LELT)
      character*4 str,str1

      ntot=LX1*LY1*LZ1*LELT
      call rzero(act_buff(1,1,1,1),ntot)
      call ifill(msk_act(1,1,1,1),0,ntot)
      do il=1,NUMCTRL
         glbid=info_agt(1,il)
         lclid=gllel(glbid)
         fceid=info_agt(2,il)
         ix=info_agt(3,il)
         iy=info_agt(4,il)
         iz=info_agt(5,il)
         act_i=act_buffer(il)
c        The face is represented by the per-node action layout.
         act_buff(ix,iy,iz,lclid)=act_i
         if (int(PARAM(90)).le.0) then
            msk_act(ix,iy,iz,lclid)=1
         endif
      enddo

      call copy(ACTIONS(1,1,1,1),act_buff(1,1,1,1),NTOT)

#ifdef YWDEBUG
      if (ISTEP.le.2 .and. NUMCTRL.gt.0) then
         write(str,"(i4.4)") NID
         write(str1,"(i4.4)") ISTEP
         open(10001,file="RECV-ACTION.txt"//str//str1)
         write(10001,*) "IGL, ", "X, ", "Y, ", "Z, ",
     $                  "ACT, "
         do il = 1,NUMCTRL
            lclid=info_agt(1,il)
            lclid=gllel(lclid)
            ix=info_agt(3,il)
            iy=info_agt(4,il)
            iz=info_agt(5,il)
            write(10001,*) lclid,
     $         (pos_agt(jl,il), jl=1,NDIM),
     $         ACTIONS(ix,iy,iz,lclid)
         enddo
         close(10001)
      endif
#endif

      if (NID.eq.0) then
         print *, "-------------------------"
         print *, "[ACTION] UPDATED"
         print *, "-------------------------"
      endif

      return
      end
c------------------------------------------------------------------
