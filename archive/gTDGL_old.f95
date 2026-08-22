program solve_pde
#ifdef _OPENMP
    use omp_lib
#endif
    implicit none

    include 'mpif.h'
    include 'dmumps_struc.h'
    type(DMUMPS_STRUC) :: id
    integer :: ierr, nnz
    integer, parameter :: dim = 60
    integer, parameter :: n = dim**2
    real(8), parameter :: xymax = 30d0
    integer, parameter :: n_t = 10000
    real(8), parameter :: tmax = 300d0
    integer, parameter :: n_relax = 1000, n_ramp = 2000
    integer, parameter :: H_relax = 0, H_ramp = n_relax
    integer, parameter :: stride = 100
    real(8), parameter :: kappa = 2.1d0, sigma_n = 1d0, gam = 10d0
    real(8), parameter :: H_ext = 0.1d0, current_max = 0.23d0
    real(8) :: h, dt, u
    complex(8), allocatable :: f_init(:), f_new(:), f_old(:), f_laplac(:)
    complex(8), allocatable :: G(:)
    integer :: t, i, j
    integer :: idx, psi_count
    real(8), allocatable :: phi(:)
    real(8) :: current_now, H_now
    logical :: is_it
    real(8), allocatable :: Jx(:), Jy(:), div_Js(:), div_Js_LHS(:), div_Js_diff(:)
    real(8) :: avg_abs_psi, sum_abs_psi
    integer :: n_open, curr
    integer, allocatable :: rows(:), cols(:)
    real(8), allocatable :: vals(:)
    real(8), allocatable :: arr_real(:), arr_imag(:)
    real(8), allocatable :: Ax(:), Ay(:)
    real(8) :: sum_abs_Js, div_Js_diff_mean
    integer :: Js_count

    ! time keeping
    real(8) :: t0, twall0
    real(8) :: t_Js, t_div, t_phi, t_lap, t_io, t_A, t_total
    integer :: nthreads

    allocate(f_init(n), f_new(n), f_old(n), f_laplac(n), G(n))
    allocate(phi(n), Jx(n), Jy(n), arr_real(n), arr_imag(n))
    allocate(div_Js_diff(n), div_Js(n), div_Js_LHS(n))
    allocate(Ax(4*n), Ay(4*n))

    call MPI_INIT(ierr)

    id%COMM = MPI_COMM_WORLD
    id%SYM = 2
    id%PAR = 1
    id%JOB = -1
    call DMUMPS(id)

    id%ICNTL(2) = 0
    id%ICNTL(3) = 0
    id%ICNTL(4) = 1

    h = xymax / (dim - 1)
    dt = tmax / n_t
    u = 5.79d0
    phi = 0d0
    
    Ax = 0d0; Ay = 0d0
    t_Js = 0d0; t_div = 0d0; t_phi = 0d0; t_lap = 0d0; t_A = 0d0 ; t_io = 0d0

    call random_number(arr_real)
    call random_number(arr_imag)

    nthreads = 1
#ifdef _OPENMP
    !$omp parallel
    !$omp masked
    nthreads = omp_get_num_threads()
    !$omp end masked
    !$omp end parallel
    write(*,'(A,I0,A,I0,A,I0)') "OpenMP build | threads = ", nthreads, &
        "   grid = ", dim, "x", dim
#else
    write(*,'(A,I0,A,I0)') "SERIAL build (no OpenMP) | grid = ", dim, "x", dim
#endif

    twall0 = wtime()

    f_init = (1d0, 0d0)
    f_init = f_init + 0.01d0 * cmplx(arr_real - 0.5d0, arr_imag - 0.5d0, 8)

    !$omp parallel do collapse(2) private(idx, is_it) schedule(static)
    do j = 1, dim
        do i = 1, dim
            idx = i + (j-1) * dim
            is_it = is_blockade(i, j, dim)
            if (is_it) f_init(idx) = (0d0, 0d0)
        end do
    end do
    !$omp end parallel do

    open(unit=10, file="data/gTDGL_psi.bin", status="replace", form="unformatted", access="stream")
    open(unit=20, file="data/gTDGL_Js.bin", status="replace", form="unformatted", access="stream")
    open(unit=30, file="data/gTDGL_div_Js.bin", status="replace", form="unformatted", access="stream")
    open(unit=40, file="data/gTDGL_phi.bin", status="replace", form="unformatted", access="stream")
    open(unit=50, file="data/gTDGL_vec_pot.bin", status="replace", form="unformatted", access="stream")

    f_old = f_init
    write(10) f_old

    allocate(rows(5*n), cols(5*n), vals(5*n))

    curr = 1
    n_open = 0
    rows = 0
    cols = 0
    vals = 0

    ! make sparse matrix
    do j = 1, dim
        do i = 1, dim
            idx = i + (j-1) * dim
            n_open = 0

            if (is_blockade(i, j, dim)) then
                rows(curr) = idx ; cols(curr) = idx ; vals(curr) = 1d0
                curr = curr + 1
                cycle
            end if

            ! count all open neighbors (for the diagonal value)
            if (i > 1 .and. .not. is_blockade(i-1, j, dim)) then
                n_open = n_open + 1
            end if
            if (i < dim .and. .not. is_blockade(i+1, j, dim)) then
                n_open = n_open + 1
            end if
            if (j > 1 .and. .not. is_blockade(i, j-1, dim)) then
                n_open = n_open + 1
            end if
            if (j < dim .and. .not. is_blockade(i, j+1, dim)) then
                n_open = n_open + 1
            end if

            ! lower triangle off-diagonals
            if (i > 1 .and. .not. is_blockade(i-1, j, dim)) then
                rows(curr) = idx ; cols(curr) = idx-1 ; vals(curr) = -1d0
                curr = curr + 1
            end if
            if (j > 1 .and. .not. is_blockade(i, j-1, dim)) then
                rows(curr) = idx ; cols(curr) = idx-dim ; vals(curr) = -1d0
                curr = curr + 1
            end if

            ! store diagonal values
            rows(curr) = idx
            cols(curr) = idx
            vals(curr) = real(n_open, 8)
            curr = curr + 1
        end do
    end do

    nnz = curr - 1
    id%N = dim**2
    id%NNZ = nnz
    allocate(id%IRN(nnz), id%JCN(nnz), id%A(nnz), id%RHS(dim**2))
    id%IRN = rows(1:nnz)
    id%JCN = cols(1:nnz)
    id%A = vals(1:nnz)
    id%JOB = 1 ; call DMUMPS(id)
    id%JOB = 2 ; call DMUMPS(id)

    ! time loop
    do t = 1, n_t

        f_new = (0d0, 0d0)

        ! linear external mag field increase
        if (t <= H_relax) then
            H_now = 0d0
        else if (t <= H_relax + H_ramp) then
            H_now = H_ext * real(t - H_relax, 8) / real(H_ramp, 8)
        else
            H_now = H_ext
        end if

        t0 = wtime()
        call calculate_Js(dim, h, H_now, f_old, Ax, Ay, Jx, Jy)
        t_Js = t_Js + (wtime() - t0)

        t0 = wtime()
        call calculate_div_Js(dim, h, Jx, Jy, div_Js)
        t_div = t_div + (wtime() - t0)

        ! linear current increase
        if (t <= n_relax) then
            current_now = 0d0
        else if (t <= n_relax + n_ramp) then
            current_now = current_max * real(t - n_relax, 8) / real(n_ramp, 8)
        else
            current_now = current_max
        end if

        t0 = wtime()
        call calculate_phi(dim, h, div_Js, sigma_n, current_now, phi)
        t_phi = t_phi + (wtime() - t0)

        t0 = wtime()
        call calculate_laplacian(dim, h, H_now, f_old, Ax, Ay, f_laplac)
        t_lap = t_lap + (wtime() - t0)

        t0 = wtime()
        ! call calculate_A(dim, h, dt, sigma_n, kappa, Jx, Jy, phi, Ax, Ay)
        t_A = t_A + (wtime() - t0)

        ! time step
        !$omp parallel do private(idx) schedule(static)
        do idx = 1, n
            G(idx) = ( sqrt(1d0 + gam**2 * abs(f_old(idx))**2) / u ) &
                   * ( f_laplac(idx) / h**2 + (1d0 - abs(f_old(idx))**2) * f_old(idx) ) &
                   - (0d0, 1d0) * phi(idx) * f_old(idx)

            f_new(idx) = ( f_old(idx) + dt * ( &
                G(idx) - (gam**2 * f_old(idx) / (1d0 + abs(f_old(idx))**2 * gam**2)) &
                * real(conjg(f_old(idx)) * G(idx)) &
            ) )! * exp(-(0d0, 1d0) * phi(idx) * dt)
        end do
        !$omp end parallel do

        ! re-zero blockade sites
        !$omp parallel do collapse(2) private(idx) schedule(static)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1)*dim
                if (is_blockade(i, j, dim)) f_new(idx) = (0d0, 0d0)
            end do
        end do
        !$omp end parallel do

        t0 = wtime()
        if (mod(t, stride) == 0) write(10) f_new
        if (mod(t, stride) == 0) write(20) Jx, Jy
        if (mod(t, stride) == 0) write(30) div_Js
        if (mod(t, stride) == 0) write(40) phi
        if (mod(t, stride) == 0) write(50) Ax, Ay
        t_io = t_io + (wtime() - t0)

        ! check for current conservation in div Js
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                div_Js_LHS(idx) = (u / sqrt(1d0 + gam**2 * abs(f_old(idx))**2)) * &
                                  ( aimag(conjg(f_old(idx)) * (f_new(idx) - f_old(idx))) / dt &
                                  + phi(idx) * abs(f_old(idx))**2 )

                div_Js_diff(idx) = div_Js(idx) - div_Js_LHS(idx)
            end do
        end do
        !$omp end parallel do

        sum_abs_Js = 0d0
        Js_count = 0
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                if (.not. is_blockade(i, j, dim)) then
                    sum_abs_Js = sum_abs_Js + abs(div_Js_diff(idx))
                    Js_count = Js_count + 1
                end if

            end do
        end do
        !$omp end parallel do

        div_Js_diff_mean = sum_abs_Js / real(Js_count, 8)

        ! diagnostics
        if (mod(t, 100) == 0) then
            sum_abs_psi = 0d0
            psi_count = 0
            do j = 1, dim
                do i = 1, dim
                    idx = i + (j-1)*dim
                    if (.not. is_blockade(i, j, dim)) then
                        sum_abs_psi = sum_abs_psi + abs(f_old(idx))
                        psi_count = psi_count + 1
                    end if
                end do
            end do
            avg_abs_psi = sum_abs_psi / real(psi_count, 8)

            write(*, '(A,I6, A,F8.5, A,F7.4, A,F9.4, A,F7.4, A,F7.4, A,F7.4)') &
                "t=",        t,                       &
                "  I=",      current_now,             &
                "  I/Jd=",   current_now/0.3849d0,    &
                "  |phi|=",  maxval(abs(phi)),        &
                "  |psi|=",  avg_abs_psi,             &
                ! "  max psi=",maxval(abs(f_old)),      &
                "  div Js=", maxval(abs(div_Js_diff)),&
                "  |Js|=",   div_Js_diff_mean

        end if

        f_old = f_new

    end do

    close(10)
    close(20)
    close(30)
    close(40)
    close(50)

    deallocate(id%IRN, id%JCN, id%A, id%RHS)
    id%JOB = -2 ; call DMUMPS(id)
    call MPI_FINALIZE(ierr)

    t_total = wtime() - twall0

    write(*,'(A)') "------------------------------------------------------------"
    write(*,'(A)') " TIMING (wall-clock seconds)"
    write(*,'(A,F12.4)')           "   total runtime        : ", t_total
    write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_Js         : ", t_Js,  "  (", 100d0*t_Js /t_total, " %)"
    write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_div_Js     : ", t_div, "  (", 100d0*t_div/t_total, " %)"
    write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_phi (MUMPS): ", t_phi, "  (", 100d0*t_phi/t_total, " %)"
    write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_laplacian  : ", t_lap, "  (", 100d0*t_lap/t_total, " %)"
    write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_A          : ", t_A,   "  (", 100d0*t_A  /t_total, " %)"
    write(*,'(A,F12.4,A,F6.2,A)')  "   file output          : ", t_io,  "  (", 100d0*t_io /t_total, " %)"
    write(*,'(A)') "------------------------------------------------------------"

contains

    function wtime() result(tt)
        real(8) :: tt
#ifdef _OPENMP
        tt = omp_get_wtime()
#else
        integer(8) :: c, r
        call system_clock(c, r)
        tt = real(c, 8) / real(r, 8)
#endif
    end function wtime

    function is_blockade(i, j, dim) result(out)
        implicit none
        integer, intent(in) :: i, j, dim
        logical :: out
        real(8) :: ci, cj, x, half_w
        real(8) :: w0_half, lead_half, R

        ci = (dim + 1) / 2d0
        cj = (dim + 1) / 2d0
        lead_half = 0.45d0 * dim
        w0_half = 0.57d0 * lead_half
        R = 0.20d0 * dim

        x = real(j, 8) - cj
        half_w = lead_half - (lead_half - w0_half) / (1d0 + (x/R)**2)
        out = abs(real(i, 8) - ci) > half_w
    end function is_blockade

    subroutine calculate_laplacian(dim, h, H_ext, psi, Ax, Ay, f_laplac)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: h, H_ext, Ax(4*dim**2), Ay(4*dim**2)
        complex(8), intent(in) :: psi(dim**2)
        complex(8), intent(out) :: f_laplac(dim**2)
        integer :: i, j, idx, A_idx, fourth
        logical :: is_it
        complex(8) :: Uy, Uy_conj, Ux, Ux_conj

        f_laplac = (0d0, 0d0)
        fourth = dim / 2

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, is_it, Uy, Uy_conj, Ux, Ux_conj)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                is_it = is_blockade(i, j, dim)
                if (is_it) cycle

                f_laplac(idx) = -4d0 * psi(idx)

                ! LINK VARIABLE <-------------------------------------------------------------------------------
                A_idx = (i + fourth) + (j + fourth - 1) * 2*dim

                Uy = exp(-(0d0,1d0) * (H_ext * (real(j,8) - (dim+1)/2d0)*h + Ay(A_idx)) * h)
                Uy_conj = exp((0d0,1d0) * (H_ext * (real(j,8) - (dim+1)/2d0)*h + Ay(A_idx)) * h)

                Ux = exp(-(0d0,1d0) * Ax(A_idx) * h)
                Ux_conj = exp((0d0,1d0) * Ax(A_idx) * h)

                if (i == 1) then
                    f_laplac(idx) = f_laplac(idx) + psi(idx+1) * Uy
                else if (is_blockade(i-1, j, dim)) then
                    f_laplac(idx) = f_laplac(idx) + psi(idx+1) * Uy
                else
                    f_laplac(idx) = f_laplac(idx) + psi(idx-1) * Uy_conj
                end if

                if (i == dim) then
                    f_laplac(idx) = f_laplac(idx) + psi(idx-1) * Uy_conj
                else if (is_blockade(i+1, j, dim)) then
                    f_laplac(idx) = f_laplac(idx) + psi(idx-1) * Uy_conj
                else
                    f_laplac(idx) = f_laplac(idx) + psi(idx+1) * Uy
                end if

                if (j == 1) then
                    f_laplac(idx) = f_laplac(idx) + psi(idx) * Ux
                else if (is_blockade(i, j-1, dim)) then
                    f_laplac(idx) = f_laplac(idx) + psi(idx) * Ux
                else
                    f_laplac(idx) = f_laplac(idx) + psi(idx-dim) * Ux_conj
                end if

                if (j == dim) then
                    f_laplac(idx) = f_laplac(idx) + psi(idx) * Ux_conj
                else if (is_blockade(i, j+1, dim)) then
                    f_laplac(idx) = f_laplac(idx) + psi(idx) * Ux_conj
                else
                    f_laplac(idx) = f_laplac(idx) + psi(idx+dim) * Ux
                end if
            end do
        end do
        !$omp end parallel do

    end subroutine calculate_laplacian

    subroutine calculate_phi(dim, h, div_Js, sigma_n, current, phi)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: div_Js(dim**2), h, current, sigma_n
        real(8), intent(inout) :: phi(dim**2)
        integer :: i, j, idx, n_valid
        real(8), allocatable :: src(:)
        real(8) :: src_mean, phi_mean

        allocate(src(dim**2))

        ! multiply constants into div Js
        !$omp parallel do private(idx) schedule(static)
        do idx = 1, dim**2
            src(idx) = div_Js(idx) / sigma_n * -(h**2)
        end do
        !$omp end parallel do

        ! find div Js mean
        src_mean = 0d0
        n_valid = 0
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1)*dim
                if (.not. is_blockade(i, j, dim)) then
                    src_mean = src_mean + src(idx)
                    n_valid = n_valid + 1
                end if
            end do
        end do
        src_mean = src_mean / real(n_valid, 8)

        ! set div Js mean to 0
        !$omp parallel do collapse(2) private(idx) schedule(static)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1)*dim
                if (.not. is_blockade(i, j, dim)) src(idx) = src(idx) - src_mean
            end do
        end do
        !$omp end parallel do

        ! inject current
        do i = 1, dim
            if (.not. is_blockade(i, 1, dim)) then
                src(i) = src(i) + current * h / sigma_n
            end if
            if (.not. is_blockade(i, dim, dim)) then
                src(i + (dim-1) * dim) = src(i + (dim-1) * dim) - current * h / sigma_n
            end if
        end do

        id%RHS = src
        id%JOB = 3 ; call DMUMPS(id)
        phi = id%RHS

        ! set phi mean to 0
        phi_mean = 0d0
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim
                if (.not. is_blockade(i, j, dim)) phi_mean = phi_mean + phi(idx)
            end do
        end do
        phi_mean = phi_mean / real(n_valid, 8)

        !$omp parallel do collapse(2) private(idx) schedule(static)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim
                if (.not. is_blockade(i, j, dim)) phi(idx) = phi(idx) - phi_mean
            end do
        end do
        !$omp end parallel do

        deallocate(src)

    end subroutine calculate_phi

    subroutine calculate_Js(dim, h, H_ext, psi, Ax, Ay, Jx, Jy)
        implicit none
        integer, intent(in) :: dim
        complex(8), intent(in) :: psi(dim**2)
        real(8), intent(in) :: h, H_ext, Ax(4*dim**2), Ay(4*dim**2)
        real(8), intent(out) :: Jx(dim**2), Jy(dim**2)
        complex(8) :: f_grad_x, f_grad_y
        complex(8) :: Uy, Uy_conj, Ux, Ux_conj
        integer :: i, j, idx, A_idx, fourth
        logical :: is_it
        real(8) :: one_over_h, one_over_two_times_h

        Jx = 0d0
        Jy = 0d0

        one_over_h = 1d0 / h
        one_over_two_times_h = 1d0 / (2d0*h)
        fourth = dim / 2

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, is_it, Uy, Uy_conj, Ux, Ux_conj, f_grad_x, f_grad_y, A_idx)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                ! LINK VARIABLE <-------------------------------------------------------------------------------
                A_idx = (i + fourth) + (j + fourth - 1) * 2*dim

                Uy = exp(-(0d0,1d0) * (H_ext * (real(j,8) - (dim+1)/2d0)*h + Ay(A_idx)) * h)
                Uy_conj = exp((0d0,1d0) * (H_ext * (real(j,8) - (dim+1)/2d0)*h + Ay(A_idx)) * h)

                Ux = exp(-(0d0,1d0) * Ax(A_idx) * h)
                Ux_conj = exp((0d0,1d0) * Ax(A_idx) * h)

                is_it = is_blockade(i, j, dim)
                if (is_it) then
                    Jx(idx) = 0d0
                    Jy(idx) = 0d0
                    cycle
                end if

                if (j > 1 .and. j < dim .and. &
                    .not. is_blockade(i, j+1, dim) .and. &
                    .not. is_blockade(i, j-1, dim)) then
                    f_grad_x = (psi(idx+dim) * Ux - psi(idx-dim) * Ux_conj) * one_over_two_times_h
                else if (j < dim .and. .not. is_blockade(i, j+1, dim)) then
                    f_grad_x = (psi(idx+dim) * Ux - psi(idx)) * one_over_h
                else if (j > 1 .and. .not. is_blockade(i, j-1, dim)) then
                    f_grad_x = (psi(idx) - psi(idx-dim) * Ux_conj) * one_over_h
                else
                    f_grad_x = (0d0, 0d0)
                end if

                if (i > 1 .and. i < dim .and. &
                    .not. is_blockade(i+1, j, dim) .and. &
                    .not. is_blockade(i-1, j, dim)) then
                    f_grad_y = (psi(idx+1) * Uy - psi(idx-1) * Uy_conj) * one_over_two_times_h
                else if (i < dim .and. .not. is_blockade(i+1, j, dim)) then
                    f_grad_y = (psi(idx+1) * Uy - psi(idx)) * one_over_h
                else if (i > 1 .and. .not. is_blockade(i-1, j, dim)) then
                    f_grad_y = (psi(idx) - psi(idx-1) * Uy_conj) * one_over_h
                else
                    f_grad_y = (0d0, 0d0)
                end if

                Jx(idx) = aimag(conjg(psi(idx)) * f_grad_x)
                Jy(idx) = aimag(conjg(psi(idx)) * f_grad_y)

            end do
        end do
        !$omp end parallel do

    end subroutine calculate_Js

    subroutine calculate_div_Js(dim, h, Jx, Jy, div_Js)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: h
        real(8), intent(in) :: Jx(dim**2), Jy(dim**2)
        real(8), intent(out) :: div_Js(dim**2)
        integer :: i, j, idx
        real(8) :: dJx_dx, dJy_dy, one_over_h, one_over_two_times_h
        logical :: is_it

        div_Js = 0d0
        one_over_h = 1d0 / h
        one_over_two_times_h = 1d0 / (2d0*h)

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, is_it, dJx_dx, dJy_dy)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                is_it = is_blockade(i, j, dim)
                if (is_it) cycle

                if (j > 1 .and. j < dim .and. &
                    .not. is_blockade(i, j+1, dim) .and. &
                    .not. is_blockade(i, j-1, dim)) then
                    dJx_dx = (Jx(idx+dim) - Jx(idx-dim)) * one_over_two_times_h
                else if (j < dim .and. .not. is_blockade(i, j+1, dim)) then
                    dJx_dx = (Jx(idx+dim) - Jx(idx)) * one_over_h
                else if (j > 1 .and. .not. is_blockade(i, j-1, dim)) then
                    dJx_dx = (Jx(idx) - Jx(idx-dim)) * one_over_h
                else
                    dJx_dx = 0d0
                end if

                if (i > 1 .and. i < dim .and. &
                    .not. is_blockade(i+1, j, dim) .and. &
                    .not. is_blockade(i-1, j, dim)) then
                    dJy_dy = (Jy(idx+1) - Jy(idx-1)) * one_over_two_times_h
                else if (i < dim .and. .not. is_blockade(i+1, j, dim)) then
                    dJy_dy = (Jy(idx+1) - Jy(idx)) * one_over_h
                else if (i > 1 .and. .not. is_blockade(i-1, j, dim)) then
                    dJy_dy = (Jy(idx) - Jy(idx-1)) * one_over_h
                else
                    dJy_dy = 0d0
                end if

                div_Js(idx) = dJx_dx + dJy_dy

            end do
        end do
        !$omp end parallel do

    end subroutine calculate_div_Js

    subroutine calculate_A(dim, h, dt, sigma_n, kappa, Jx, Jy, phi, Ax, Ay)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: Jx(dim**2), Jy(dim**2), h, phi(dim**2), sigma_n, dt, kappa
        real(8), intent(inout) :: Ax(4*dim**2), Ay(4*dim**2)
        integer :: i, j, idx, fourth
        integer :: ip, jp, pidx
        real(8) :: grad_x_phi(dim**2), grad_y_phi(dim**2), denom
        real(8), allocatable :: Ax_laplac(:), Ay_laplac(:), Ax_new(:), Ay_new(:)
        real(8) :: one_over_h, one_over_two_times_h
        logical :: is_it

        allocate(Ax_laplac(5*dim**2), Ay_laplac(5*dim**2), Ax_new(5*dim**2), Ay_new(5*dim**2))

        one_over_h = 1d0 / h
        one_over_two_times_h = 1d0 / (2d0*h)
        denom = 1 / h**2
        grad_x_phi = 0d0 ; grad_y_phi = 0d0
        Ax_laplac = 0d0 ; Ay_laplac = 0d0

        ! calculate the gradient of phi
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, is_it)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                is_it = is_blockade(i, j, dim)
                if (is_it) then
                    grad_x_phi(idx) = 0d0
                    grad_y_phi(idx) = 0d0
                    cycle
                end if

                if (j > 1 .and. j < dim .and. &
                    .not. is_blockade(i, j+1, dim) .and. &
                    .not. is_blockade(i, j-1, dim)) then
                    grad_x_phi(idx) = (phi(idx+dim) - phi(idx-dim)) * one_over_two_times_h
                else if (j < dim .and. .not. is_blockade(i, j+1, dim)) then
                    grad_x_phi(idx) = (phi(idx+dim) - phi(idx)) * one_over_h
                else if (j > 1 .and. .not. is_blockade(i, j-1, dim)) then
                    grad_x_phi(idx) = (phi(idx) - phi(idx-dim)) * one_over_h
                else
                    grad_x_phi(idx) = 0d0
                end if

                if (i > 1 .and. i < dim .and. &
                    .not. is_blockade(i+1, j, dim) .and. &
                    .not. is_blockade(i-1, j, dim)) then
                    grad_y_phi(idx) = (phi(idx+1) - phi(idx-1)) * one_over_two_times_h
                else if (i < dim .and. .not. is_blockade(i+1, j, dim)) then
                    grad_y_phi(idx) = (phi(idx+1) - phi(idx)) * one_over_h
                else if (i > 1 .and. .not. is_blockade(i-1, j, dim)) then
                    grad_y_phi(idx) = (phi(idx) - phi(idx-1)) * one_over_h
                else
                    grad_y_phi(idx) = 0d0
                end if
            end do
        end do
        !$omp end parallel do

        ! calculate laplacian of A
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 2, 2*dim-1
            do i = 2, 2*dim-1
                idx = i + (j-1) * 2*dim

                Ax_laplac(idx) = -4d0 * Ax(idx)
                Ay_laplac(idx) = -4d0 * Ay(idx)

                Ax_laplac(idx) = Ax_laplac(idx) + Ax(idx-1)
                Ax_laplac(idx) = Ax_laplac(idx) + Ax(idx+1)
                Ax_laplac(idx) = Ax_laplac(idx) + Ax(idx-2*dim)
                Ax_laplac(idx) = Ax_laplac(idx) + Ax(idx+2*dim)

                Ay_laplac(idx) = Ay_laplac(idx) + Ay(idx-1)
                Ay_laplac(idx) = Ay_laplac(idx) + Ay(idx+1)
                Ay_laplac(idx) = Ay_laplac(idx) + Ay(idx-2*dim)
                Ay_laplac(idx) = Ay_laplac(idx) + Ay(idx+2*dim)
            end do
        end do
        !$omp end parallel do

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 1, 2*dim
            do i = 1, 2*dim
                idx = i + (j-1) * 2*dim
                Ax_laplac(idx) = Ax_laplac(idx) * denom
                Ay_laplac(idx) = Ay_laplac(idx) * denom
            end do
        end do
        !$omp end parallel do

        ! enforce Dirichlet boundary conditions
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 1, 2*dim
            do i = 1, 2*dim
                if (i == 1 .or. i == 2*dim .or. &
                    j == 1 .or. j == 2*dim) then

                    idx = i + (j-1) * 2*dim
                    
                    Ax(idx) = 0d0 ; Ay(idx) = 0d0
                    Ax_laplac(idx) = 0d0 ; Ay_laplac(idx) = 0d0
                end if
            end do
        end do
        !$omp end parallel do

        ! time step
        fourth = dim / 2
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, pidx, ip, jp)
        do j = 1, 2*dim
            do i = 1, 2*dim
                idx = i + (j-1) * 2*dim
                    
                if (i > fourth .and. i <= 2*dim-fourth .and. j > fourth .and. j <= 2*dim-fourth) then
                    ip = i - fourth ; jp = j - fourth
                    pidx = ip + (jp-1)*dim

                    Ax_new(idx) = Ax(idx) + dt * &
                    ((kappa**2 * Ax_laplac(idx) + Jx(pidx)) / sigma_n - grad_x_phi(pidx))
                    Ay_new(idx) = Ay(idx) + dt * &
                    ((kappa**2 * Ay_laplac(idx) + Jy(pidx)) / sigma_n - grad_y_phi(pidx))
                else
                    Ax_new(idx) = Ax(idx) + dt * ((kappa**2 * Ax_laplac(idx)) / sigma_n)
                    Ay_new(idx) = Ay(idx) + dt * ((kappa**2 * Ay_laplac(idx)) / sigma_n)
                end if
                
            end do
        end do
        !$omp end parallel do

        Ax = Ax_new
        Ay = Ay_new

        deallocate(Ax_laplac, Ay_laplac, Ax_new, Ay_new)

    end subroutine calculate_A

end program solve_pde