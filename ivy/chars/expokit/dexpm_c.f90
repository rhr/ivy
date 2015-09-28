module dexpm_wrap

  implicit none

contains

  ! Calculate exp(H*t) for a dense nstates-by-nstates matrix H using Expokit.
  subroutine f_dexpm(nstates, H, t, expH) bind(c)
    use, intrinsic :: ISO_C_BINDING
    integer(c_int), intent(in), value :: nstates
    real(c_double), intent(in) :: H(nstates*nstates)
    real(c_double), intent(in), value :: t
    real(c_double), intent(out) :: expH(nstates*nstates)

    ! Expokit variables
    external :: DGPADM
    integer, parameter :: ideg = 6
    double precision, dimension(4*nstates*nstates + ideg + 1) :: wsp
    integer, dimension(nstates) :: iwsp
    integer :: iexp, ns, iflag, n

    ! if (size(H,1) /= size(H,2)) then
    !    stop 'dexpm: matrix must be square'
    ! end if

    n = nstates
    call DGPADM(ideg, n, t, H, n, wsp, size(wsp,1), iwsp, iexp, ns, iflag)
    !expH = reshape(wsp(iexp:iexp+n*n-1), [n,n])
    expH = wsp(iexp:iexp+n*n-1)
    ! write(*,*) H
    ! write(*,*) expH
  end subroutine f_dexpm

  subroutine f_dexpm_wsp(nstates, H, t, i, wsp, expH) bind(c)
    use, intrinsic :: ISO_C_BINDING
    integer(c_int), intent(in), value :: nstates
    real(c_double), intent(in) :: H(nstates*nstates)
    real(c_double), intent(in), value :: t
    integer(c_int), intent(in), value :: i
    real(c_double), intent(out) :: expH(nstates*nstates)

    ! Expokit variables
    external :: DGPADM
    real(c_double), intent(in) :: wsp(4*nstates*nstates + i + 1)
    integer, dimension(nstates) :: iwsp
    integer :: ideg, iexp, ns, iflag, n
    ideg = i
    n = nstates
    call DGPADM(ideg, n, t, H, n, wsp, size(wsp,1), iwsp, iexp, ns, iflag)
    expH = wsp(iexp:iexp+n*n-1)
    ! write(*,*) H
    ! write(*,*) expH
  end subroutine f_dexpm_wsp

end module dexpm_wrap
