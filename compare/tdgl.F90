program solve_tdgl
#ifdef _OPENMP
    use omp_lib
#endif
    implicit none

    include 'mpif.h'
    include 'dmumps_struc.h'
    type(DMUMPS_STRUC) :: id
    integer :: ierr, nnz
    integer, parameter :: dimx = 16
    integer, parameter :: dimy = 16
    integer, parameter :: n = dimx*dimy
    integer, parameter :: nA = dimy**2
    real(8), parameter :: ymax = 30d0
    integer, parameter :: n_t = 10000
    real(8), parameter :: tmax = 200d0
    integer, parameter :: n_relax = 0, n_ramp = 1000
    integer, parameter :: H_relax = 0, H_ramp = n_relax
    integer, parameter :: stride = nint(n_t / tmax)
    integer, parameter :: stride_power = stride
    integer, parameter :: stride_mumps = 1
    real(8), parameter :: kappa = 2.1d0, sigma_n = 1d0, gam = 0d0
    real(8), parameter :: H_ext = 0.0d0, current_max = 0.25d0
    integer, parameter :: print_stride = 500
    logical, parameter :: A_on = .false.
    real(8) :: h, dt, u, eps
    complex(8), allocatable :: psi_init(:), psi_new(:), psi_old(:), psi_laplac(:)
    complex(8), allocatable :: G(:)
    integer :: t, i, j
    integer :: idx, psi_count
    real(8), allocatable :: phi(:)
    real(8) :: current_now, H_now
    logical :: is_it
    real(8), allocatable :: Jsx(:), Jsy(:), div_Js(:), Jnx(:), Jny(:)
    real(8), allocatable :: Jsx_plot(:), Jsy_plot(:), Jnx_plot(:), Jny_plot(:), Jtotx(:), Jtoty(:)
    real(8), allocatable :: div_Js_LHS(:), div_Js_RHS(:), div_Js_diff_LHS(:), div_Js_diff_RHS(:)
    real(8) :: avg_abs_psi, sum_abs_psi
    integer :: n_open, curr
    integer, allocatable :: rows(:), cols(:)
    real(8), allocatable :: vals(:)
    real(8), allocatable :: dreal(:), dimag(:)
    real(8), allocatable :: Ax(:), Ay(:)
    real(8) :: sum_abs_Js, div_Js_diff_mean
    integer :: Js_count
    real(8) :: max_div_A, mean_div_A
    integer :: div_i, div_j
    real(8), allocatable :: div_A(:), div_Jtot(:), Jn_wall(:)
    real(8), allocatable :: Ex(:), Ey(:), Pow(:)

    ! time keeping
    real(8) :: t0, twall0
    real(8) :: t_Js, t_div, t_phi, t_lap, t_io, t_A, t_total
    integer :: nthreads

    allocate(psi_init(n), psi_new(n), psi_old(n), psi_laplac(n), G(n))
    allocate(phi(n), Jsx(n), Jsy(n), Jtotx(n), Jtoty(n), dreal(n), dimag(n))
    allocate(div_Js_diff_LHS(n), div_Js_diff_RHS(n), div_Js(n), div_Js_LHS(n), div_Js_RHS(n))
    allocate(Ax(9*nA), Ay(9*nA), Jnx(n), Jny(n))
    allocate(Jnx_plot(n), Jny_plot(n), Jsx_plot(n), Jsy_plot(n))
    allocate(div_A(9*nA), div_Jtot(n), Jn_wall(dimy), Ex(n), Ey(n), Pow(n))

    call MPI_INIT(ierr)

    id%COMM = MPI_COMM_WORLD
    id%SYM = 1
    id%PAR = 1
    id%JOB = -1
    call DMUMPS(id)

    id%ICNTL(2) = 0
    id%ICNTL(3) = 0
    id%ICNTL(4) = 1

    h = ymax / (dimy - 1)
    dt = tmax / n_t
    u = 5.79d0
    phi = 0d0
    eps = 1.0d0
    
    Ax = 0d0; Ay = 0d0; Jnx = 0d0; Jny = 0d0; Jn_wall = 0d0; Ex = 0d0; Ey = 0d0; Pow = 0d0
    t_Js = 0d0; t_div = 0d0; t_phi = 0d0; t_lap = 0d0; t_A = 0d0 ; t_io = 0d0

    call random_number(dreal)
    call random_number(dimag)

    nthreads = 1
#ifdef _OPENMP
        !$omp parallel
        !$omp masked
        nthreads = omp_get_num_threads()
        !$omp end masked
        !$omp end parallel
        write(*,'(A,I0, A,I0, A,I0, A,F7.4, A,F7.4)') "OpenMP build | threads = ", nthreads, &
            "   grid = ", dimx, "x", dimy, " Mag", H_ext, " Curr", current_max
#else
        write(*,'(A,I0, A,I0)') "SERIAL build (no OpenMP) | grid = ", dimx, "x", dimy
#endif

    twall0 = wtime()

    psi_init = (1d0, 0d0)
    ! psi_init = psi_init + 0.01d0 * cmplx(dreal - 0.5d0, dimag - 0.5d0, 8)

    !$omp parallel do collapse(2) private(idx, is_it) schedule(static)
    do j = 1, dimx
        do i = 1, dimy
            idx = i + (j-1) * dimy
            is_it = is_blockade(i, j, dimx, dimy)
            if (is_it) psi_init(idx) = (0d0, 0d0)
        end do
    end do
    !$omp end parallel do

    call execute_command_line("mkdir -p data")
    open(unit=10, file="data/psi.bin", status="replace", form="unformatted", access="stream")
    open(unit=20, file="data/Js.bin", status="replace", form="unformatted", access="stream")
    open(unit=30, file="data/div_Js.bin", status="replace", form="unformatted", access="stream")
    open(unit=40, file="data/phi.bin", status="replace", form="unformatted", access="stream")
    open(unit=50, file="data/vec_pot.bin", status="replace", form="unformatted", access="stream")
    open(unit=60, file="data/Jn.bin", status="replace", form="unformatted", access="stream")
    open(unit=70, file="data/Jtot.bin", status="replace", form="unformatted", access="stream")
    open(unit=80, file="data/div_A.bin", status="replace", form="unformatted", access="stream")
    open(unit=90, file="data/div_Jtot.bin", status="replace", form="unformatted", access="stream")
    open(unit=100, file="data/power.bin", status="replace", form="unformatted", access="stream")

    psi_old = psi_init
    write(10) psi_old

    allocate(rows(5*n), cols(5*n), vals(5*n))

    curr = 1
    n_open = 0
    rows = 0
    cols = 0
    vals = 0

    ! make sparse matrix
    do j = 1, dimx
        do i = 1, dimy
            idx = i + (j-1) * dimy
            n_open = 0

            if (is_blockade(i, j, dimx, dimy)) then
                rows(curr) = idx ; cols(curr) = idx ; vals(curr) = 1d0
                curr = curr + 1
                cycle
            end if

            ! count all open neighbors (for the diagonal value)
            if (i > 1 .and. .not. is_blockade(i-1, j, dimx, dimy)) then
                n_open = n_open + 1
            end if
            if (i < dimy .and. .not. is_blockade(i+1, j, dimx, dimy)) then
                n_open = n_open + 1
            end if
            if (j > 1 .and. .not. is_blockade(i, j-1, dimx, dimy)) then
                n_open = n_open + 1
            end if
            if (j < dimx .and. .not. is_blockade(i, j+1, dimx, dimy)) then
                n_open = n_open + 1
            end if

            ! lower triangle off-diagonals
            if (i > 1 .and. .not. is_blockade(i-1, j, dimx, dimy)) then
                rows(curr) = idx ; cols(curr) = idx-1 ; vals(curr) = -1d0
                curr = curr + 1
            end if
            if (j > 1 .and. .not. is_blockade(i, j-1, dimx, dimy)) then
                rows(curr) = idx ; cols(curr) = idx-dimy ; vals(curr) = -1d0
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
    id%N = n
    id%NNZ = nnz
    allocate(id%IRN(nnz), id%JCN(nnz), id%A(nnz), id%RHS(n))
    id%IRN = rows(1:nnz)
    id%JCN = cols(1:nnz)
    id%A = vals(1:nnz)
    id%JOB = 1 ; call DMUMPS(id)
    id%JOB = 2 ; call DMUMPS(id)

    ! time loop
    do t = 1, n_t

        psi_new = (0d0, 0d0)

        ! linear external mag field increase
        if (t <= H_relax) then
            H_now = 0d0
        else if (t <= H_relax + H_ramp) then
            H_now = H_ext * real(t - H_relax, 8) / real(H_ramp, 8)
        else
            H_now = H_ext
        end if

        ! linear current increase
        if (t <= n_relax) then
            current_now = 0d0
        else if (t <= n_relax + n_ramp) then
            current_now = current_max * real(t - n_relax, 8) / real(n_ramp, 8)
        else
            current_now = current_max
        end if

        t0 = wtime()
        call calculate_Js(dimx, dimy, h, H_now, psi_old, Ax, Ay, Jsx, Jsy)
        t_Js = t_Js + (wtime() - t0)

        t0 = wtime()
        call calculate_div_Js(dimx, dimy, h, H_now, psi_old, Ax, Ay, div_Js)
        t_div = t_div + (wtime() - t0)

        if (mod(t, stride_mumps) == 0) then
            t0 = wtime()
            call calculate_phi(dimx, dimy, h, div_Js, sigma_n, current_now, phi)
            t_phi = t_phi + (wtime() - t0)
        end if

        t0 = wtime()
        call calculate_laplacian(dimx, dimy, h, H_now, psi_old, Ax, Ay, psi_laplac)
        t_lap = t_lap + (wtime() - t0)

        t0 = wtime()
        if (A_on) call calculate_A_Jn_E(dimx, dimy, h, dt, sigma_n, kappa, current_now, &
                                        Jsx, Jsy, phi, Ax, Ay, Jnx, Jny, Jn_wall, Ex, Ey, Pow)
        t_A = t_A + (wtime() - t0)

        ! time step
        !$omp parallel do private(idx) schedule(static)
        do idx = 1, n
            G(idx) = (sqrt(1d0 + gam**2 * abs(psi_old(idx))**2) / u ) &
                   * (psi_laplac(idx) / h**2 + (eps - abs(psi_old(idx))**2) * psi_old(idx))

            psi_new(idx) = (psi_old(idx) + dt * ( &
                G(idx) - (gam**2 * psi_old(idx) / (1d0 + abs(psi_old(idx))**2 * gam**2)) &
                * real(conjg(psi_old(idx)) * G(idx)) &
            )) * exp(-(0d0, 1d0) * phi(idx) * dt)
        end do
        !$omp end parallel do

        ! re-zero blockade sites
        !$omp parallel do collapse(2) private(idx) schedule(static)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1)*dimy
                if (is_blockade(i, j, dimx, dimy)) psi_new(idx) = (0d0, 0d0)
            end do
        end do
        !$omp end parallel do

        ! for plotting
        if (mod(t, stride) == 0) then
            call compute_div_A_field(dimx, dimy, h, Ax, Ay, div_A)
            call compute_div_Jtot_field(dimx, dimy, h, Jn_wall, Jsx, Jsy, Jnx, Jny, div_Jtot)
            call post_processing(dimx, dimy, Jsx, Jsy, Jnx, Jny, &
                                 Jsx_plot, Jsy_plot, Jnx_plot, Jny_plot, Jtotx, Jtoty)
        end if

        t0 = wtime()
        if (mod(t, stride) == 0) then
            write(10) psi_new
            write(20) Jsx_plot, Jsy_plot
            write(30) div_Js
            write(40) phi
            write(50) Ax, Ay
            write(60) Jnx_plot, Jny_plot
            write(70) Jtotx, Jtoty
            write(80) div_A
            write(90) div_Jtot
        end if
        if (mod(t, stride_power) == 0) then
            write(100) Pow
        end if
        t_io = t_io + (wtime() - t0)

        ! check for current conservation in div Js
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1) * dimy

                div_Js_LHS(idx) = (u / sqrt(1d0 + gam**2 * abs(psi_old(idx))**2)) * &
                                  (aimag(conjg(psi_old(idx)) * (psi_new(idx) - psi_old(idx))) / dt &
                                  + phi(idx) * abs(psi_old(idx))**2)

                div_Js_RHS(idx) = aimag(conjg(psi_old(idx)) * psi_laplac(idx)) / h**2

                div_Js_diff_LHS(idx) = div_Js(idx) - div_Js_LHS(idx)
                div_Js_diff_RHS(idx) = div_Js(idx) - div_Js_RHS(idx)
            end do
        end do
        !$omp end parallel do

        ! diagnostic prints
        if (mod(t, print_stride) == 0) then
            sum_abs_psi = 0d0
            psi_count = 0
            do j = 1, dimx
                do i = 1, dimy
                    idx = i + (j-1)*dimy
                    if (.not. is_blockade(i, j, dimx, dimy)) then
                        sum_abs_psi = sum_abs_psi + abs(psi_old(idx))
                        psi_count = psi_count + 1
                    end if
                end do
            end do
            avg_abs_psi = sum_abs_psi / real(psi_count, 8)

            call compute_div_Jtot_field(dimx, dimy, h, Jn_wall, Jsx, Jsy, Jnx, Jny, div_Jtot)

            call measure_div_A(dimx, dimy, h, Ax, Ay, max_div_A, mean_div_A, div_i, div_j)
            write(*, '(A,ES11.3, A,ES11.3, A,I4,A,I4,A)') &
                "   max|divA|=", max_div_A, &
                "  mean|divA|=", mean_div_A, &
                "  at (", div_i, ",", div_j, ")"

            write(*, '(A,I6, A,F8.5, A,F9.4, A,F7.4, A,F7.4, A,ES11.3)') &
                "t=",        t,                       &
                "  I=",      current_now,             &
                "  max phi=",  maxval(abs(phi)),      &
                "  |psi|=",  avg_abs_psi,             &
                ! "  max psi=",maxval(abs(psi_old)),    &
                "  L divJs=", maxval(abs(div_Js_diff_LHS)),&
                ! "  R divJs=", maxval(abs(div_Js_diff_RHS)),&
                "  divJtot=", maxval(abs(div_Jtot))

            print *

        end if

        psi_old = psi_new

    end do

    close(10)
    close(20)
    close(30)
    close(40)
    close(50)
    close(60)
    close(70)
    close(80)
    close(90)
    close(100)

    deallocate(id%IRN, id%JCN, id%A, id%RHS)
    id%JOB = -2 ; call DMUMPS(id)
    call MPI_FINALIZE(ierr)

    deallocate(psi_init, psi_new, psi_old, psi_laplac, G)
    deallocate(phi, Jsx, Jsy, dreal, dimag, Jtotx, Jtoty)
    deallocate(div_Js_diff_LHS, div_Js_diff_RHS, div_Js, div_Js_LHS, div_Js_RHS)
    deallocate(Ax, Ay, Jnx, Jny)
    deallocate(Jnx_plot, Jny_plot, Jsx_plot, Jsy_plot)
    deallocate(div_A, div_Jtot, Jn_wall, Ex, Ey, Pow)

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

    function is_blockade(i, j, dimx, dimy) result(out)
        implicit none
        integer, intent(in) :: i, j, dimx, dimy
        logical :: out
        real(8) :: ci, cj, x, half_w
        real(8) :: w0_half, lead_half, R

        out = .false.
    end function is_blockade

    ! function is_blockade(i, j, dimx, dimy) result(out)
    !     implicit none
    !     integer, intent(in) :: i, j, dimx, dimy
    !     logical :: out
    !     real(8) :: ci, cj, x, half_w
    !     real(8) :: w0_half, lead_half, R

    !     ci = (dimy + 1) / 2d0
    !     cj = (dimx + 1) / 2d0
    !     lead_half = 0.45d0 * dimy
    !     w0_half = 0.57d0 * lead_half
    !     R = 0.20d0 * dimy

    !     x = real(j, 8) - cj
    !     half_w = lead_half - (lead_half - w0_half) / (1d0 + (x/R)**2)
    !     out = abs(real(i, 8) - ci) > half_w
    ! end function is_blockade

    ! function is_blockade(i, j, dimx, dimy) result(out)
    !     implicit none
    !     integer, intent(in) :: i, j, dimx, dimy
    !     logical :: out
    !     real(8) :: ci, cj
    !     real(8) :: rect_half_w, gap_half

    !     ci = (dimy + 1) / 2d0
    !     cj = (dimx + 1) / 2d0

    !     rect_half_w = 0.05d0 * dimx        ! half width of blockade
    !     gap_half = 0.7 * 0.5d0 * dimy ! half height of open channel

    !     if (abs(real(j,8) - cj) <= rect_half_w .and. &
    !         abs(real(i,8) - ci) >  gap_half) then
    !         out = .true.
    !     else
    !         out = .false.
    !     end if
    ! end function is_blockade

    subroutine calculate_laplacian(dimx, dimy, h, H_ext, psi, Ax, Ay, psi_laplac)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: h, H_ext, Ax(9*dimy*dimy), Ay(9*dimy*dimy)
        complex(8), intent(in) :: psi(dimx*dimy)
        complex(8), intent(out) :: psi_laplac(dimx*dimy)
        integer :: i, j, idx, A_idx, padx, pady, n_open
        real(8) :: H_term
        complex(8) :: Uy, Uy_conj, Ux, Ux_conj, psi_val
        logical :: is_it

        psi_laplac = (0d0, 0d0)
        padx = (3*dimy - dimx) / 2
        pady = dimy

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, A_idx, is_it, Uy, Uy_conj, Ux, Ux_conj, psi_val, n_open, H_term)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1) * dimy

                is_it = is_blockade(i, j, dimx, dimy)
                if (is_it) cycle

                A_idx = (i + pady) + (j + padx - 1) * 3*dimy
                H_term = H_ext * (real(j,8) - (dimx+1)/2d0) * h

                Uy = exp(-(0d0,1d0) * (H_term + Ay(A_idx)) * h)
                Uy_conj = exp((0d0,1d0) * (H_term + Ay(A_idx-1)) * h)
                Ux = exp(-(0d0,1d0) * Ax(A_idx) * h)
                Ux_conj = exp((0d0,1d0) * Ax(A_idx-3*dimy) * h)

                psi_val = (0d0, 0d0)
                n_open = 0

                if (i > 1 .and. .not. is_blockade(i-1, j, dimx, dimy)) then
                    psi_val = psi_val + psi(idx-1) * Uy_conj
                    n_open = n_open + 1
                end if
                if (i < dimy .and. .not. is_blockade(i+1, j, dimx, dimy)) then
                    psi_val = psi_val + psi(idx+1) * Uy
                    n_open = n_open + 1
                end if
                if (j > 1 .and. .not. is_blockade(i, j-1, dimx, dimy)) then
                    psi_val = psi_val + psi(idx-dimy) * Ux_conj
                    n_open = n_open + 1
                end if
                if (j < dimx .and. .not. is_blockade(i, j+1, dimx, dimy)) then
                    psi_val = psi_val + psi(idx+dimy) * Ux
                    n_open = n_open + 1
                end if

                psi_laplac(idx) = psi_val - real(n_open, 8) * psi(idx)
            end do
        end do
        !$omp end parallel do

    end subroutine calculate_laplacian

    subroutine calculate_phi(dimx, dimy, h, div_Js, sigma_n, current, phi)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: div_Js(dimx*dimy), h, current, sigma_n
        real(8), intent(inout) :: phi(dimx*dimy)
        integer :: i, j, idx, n_valid
        real(8), allocatable :: src(:)
        real(8) :: src_mean, phi_mean

        allocate(src(dimx*dimy))

        ! multiply constants into div Js
        !$omp parallel do private(idx) schedule(static)
        do idx = 1, dimx*dimy
            src(idx) = div_Js(idx) / sigma_n * -(h**2)
        end do
        !$omp end parallel do

        ! find div Js mean
        src_mean = 0d0
        n_valid = 0
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1)*dimy
                if (.not. is_blockade(i, j, dimx, dimy)) then
                    src_mean = src_mean + src(idx)
                    n_valid = n_valid + 1
                end if
            end do
        end do
        src_mean = src_mean / real(n_valid, 8)

        ! set div Js mean to 0
        !$omp parallel do collapse(2) private(idx) schedule(static)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1)*dimy
                if (.not. is_blockade(i, j, dimx, dimy)) src(idx) = src(idx) - src_mean
            end do
        end do
        !$omp end parallel do

        ! inject current
        do i = 1, dimy
            if (.not. is_blockade(i, 1, dimx, dimy)) then
                src(i) = src(i) + current * h / sigma_n
            end if
            if (.not. is_blockade(i, dimx, dimx, dimy)) then
                src(i + (dimx-1) * dimy) = src(i + (dimx-1) * dimy) - current * h / sigma_n
            end if
        end do

        id%RHS = src
        id%JOB = 3 ; call DMUMPS(id)
        phi = id%RHS

        ! set phi mean to 0
        phi_mean = 0d0
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1) * dimy
                if (.not. is_blockade(i, j, dimx, dimy)) phi_mean = phi_mean + phi(idx)
            end do
        end do
        phi_mean = phi_mean / real(n_valid, 8)

        !$omp parallel do collapse(2) private(idx) schedule(static)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1) * dimy
                if (.not. is_blockade(i, j, dimx, dimy)) phi(idx) = phi(idx) - phi_mean
            end do
        end do
        !$omp end parallel do

        deallocate(src)

    end subroutine calculate_phi

    subroutine calculate_Js(dimx, dimy, h, H_ext, psi, Ax, Ay, Jsx, Jsy)
        implicit none
        integer, intent(in) :: dimx, dimy
        complex(8), intent(in) :: psi(dimx*dimy)
        real(8), intent(in) :: h, H_ext, Ax(9*dimy*dimy), Ay(9*dimy*dimy)
        real(8), intent(out) :: Jsx(dimx*dimy), Jsy(dimx*dimy)
        integer :: i, j, idx, A_idx, padx, pady
        real(8) :: H_term, inv_h, g_i, g_j
        complex(8) :: U
        logical :: is_it

        Jsx = 0d0 ; Jsy = 0d0
        padx = (3*dimy - dimx) / 2
        pady = dimy
        inv_h = 1d0 / h

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, A_idx, H_term, U, g_i, g_j, is_it)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1) * dimy

                is_it = is_blockade(i, j, dimx, dimy)
                if (is_it) cycle

                A_idx = (i + pady) + (j + padx - 1) * 3*dimy
                H_term = H_ext * (real(j,8) - (dimx+1)/2d0) * h

                g_i = 0d0 ; g_j = 0d0

                ! calculate Jsy
                if (i < dimy .and. .not. is_blockade(i+1, j, dimx, dimy)) then
                    U = exp(-(0d0,1d0) * (H_term + Ay(A_idx)) * h)
                    g_i = aimag(conjg(psi(idx)) * U * psi(idx+1)) * inv_h
                end if

                ! calculate Jsx
                if (j < dimx .and. .not. is_blockade(i, j+1, dimx, dimy)) then
                    U = exp(-(0d0,1d0) * Ax(A_idx) * h)
                    g_j = aimag(conjg(psi(idx)) * U * psi(idx+dimy)) * inv_h
                end if

                Jsy(idx) = g_i
                Jsx(idx) = g_j

            end do
        end do
        !$omp end parallel do

    end subroutine calculate_Js

    subroutine calculate_div_Js(dimx, dimy, h, H_ext, psi, Ax, Ay, div_Js)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: h, H_ext, Ax(9*dimy*dimy), Ay(9*dimy*dimy)
        complex(8), intent(in) :: psi(dimx*dimy)
        real(8), intent(out) :: div_Js(dimx*dimy)
        integer :: i, j, idx, A_idx, padx, pady
        real(8) :: div, Ay_e, Ax_e, H_term, inv_h2
        complex(8) :: U
        logical :: is_it

        div_Js = 0d0
        padx = (3*dimy - dimx) / 2
        pady = dimy
        inv_h2 = 1d0 / h**2

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, A_idx, div, Ay_e, Ax_e, H_term, U, is_it)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1) * dimy

                is_it = is_blockade(i, j, dimx, dimy)
                if (is_it) cycle

                A_idx = (i + pady) + (j + padx - 1) * 3*dimy
                H_term = H_ext * (real(j,8) - (dimx+1)/2d0) * h
                div = 0d0

                if (i < dimy .and. .not. is_blockade(i+1, j, dimx, dimy)) then
                    U = exp(-(0d0, 1d0) * (H_term + Ay(A_idx)) * h)
                    div = div + aimag(conjg(psi(idx)) * U * psi(idx+1))
                end if
                if (i > 1 .and. .not. is_blockade(i-1, j, dimx, dimy)) then
                    U = exp((0d0, 1d0) * (H_term + Ay(A_idx - 1)) * h)
                    div = div + aimag(conjg(psi(idx)) * U * psi(idx-1))
                end if

                if (j < dimx .and. .not. is_blockade(i, j+1, dimx, dimy)) then
                    U = exp(-(0d0, 1d0) * Ax(A_idx) * h)
                    div = div + aimag(conjg(psi(idx)) * U * psi(idx+dimy))
                end if
                if (j > 1 .and. .not. is_blockade(i, j-1, dimx, dimy)) then
                    U = exp((0d0, 1d0) * Ax(A_idx - 3*dimy) * h)
                    div = div + aimag(conjg(psi(idx)) * U * psi(idx-dimy))
                end if

                div_Js(idx) = div * inv_h2
            end do
        end do
        !$omp end parallel do

    end subroutine calculate_div_Js

    subroutine calculate_A_Jn_E(dimx, dimy, h, dt, sigma_n, kappa, current, Jsx, Jsy, phi, Ax, Ay, Jnx, Jny, Jn_wall, Ex, Ey, Pow)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: Jsx(dimx*dimy), Jsy(dimx*dimy), h, phi(dimx*dimy), sigma_n, dt, kappa, current
        real(8), intent(inout) :: Ax(9*dimy*dimy), Ay(9*dimy*dimy), Jnx(dimx*dimy), Jny(dimx*dimy)
        real(8), intent(out) :: Jn_wall(dimy), Ex(dimx*dimy), Ey(dimx*dimy), Pow(dimx*dimy)
        integer :: i, j, idx, padx, pady
        integer :: ip, jp, pidx
        real(8) :: gx_phi, gy_phi, inv_h2, inv_h
        real(8), allocatable :: Ax_laplac(:), Ay_laplac(:), Ax_new(:), Ay_new(:)
        logical :: is_it

        allocate(Ax_laplac(9*dimy*dimy), Ay_laplac(9*dimy*dimy), Ax_new(9*dimy*dimy), Ay_new(9*dimy*dimy))

        inv_h = 1d0 / h
        inv_h2 = 1 / h**2
        gx_phi = 0d0 ; gy_phi = 0d0
        Ax_laplac = 0d0 ; Ay_laplac = 0d0
        padx = (3*dimy - dimx) / 2
        pady = dimy

        ! calculate laplacian of A
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 2, 3*dimy-1
            do i = 2, 3*dimy-1
                idx = i + (j-1) * 3*dimy

                Ax_laplac(idx) = (Ax(idx-1) + Ax(idx+1) &
                                 + Ax(idx-3*dimy) + Ax(idx+3*dimy) &
                                 - 4d0 * Ax(idx)) * inv_h2

                Ay_laplac(idx) = (Ay(idx-1) + Ay(idx+1) &
                                 + Ay(idx-3*dimy) + Ay(idx+3*dimy) &
                                 - 4d0 * Ay(idx)) * inv_h2
            end do
        end do
        !$omp end parallel do

        ! total current through the left injection site
        do ip = 1, dimy
            idx = (ip + pady) + (padx - 1) * 3*dimy
            Jn_wall(ip) = -kappa**2 * Ax_laplac(idx)
        end do

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, pidx, ip, jp, gx_phi, gy_phi)
        do j = 2, 3*dimy-1
            do i = 2, 3*dimy-1
                idx = i + (j-1) * 3*dimy

                ! solves for the superconducting domain
                if (i > pady .and. i <= 3*dimy-pady .and. j > padx .and. j <= padx+dimx) then
                    ip = i - pady ; jp = j - padx
                    pidx = ip + (jp-1)*dimy

                    ! gradient of phi
                    if (is_blockade(ip, jp, dimx, dimy)) then
                        gx_phi = 0d0
                        gy_phi = 0d0
                    else
                        if (jp < dimx .and. .not. is_blockade(ip, jp+1, dimx, dimy)) then
                            gx_phi = (phi(pidx+dimy) - phi(pidx)) * inv_h
                        else if (jp == dimx) then
                            gx_phi = -current / sigma_n
                        else
                            gx_phi = 0d0
                        end if
                        if (ip < dimy .and. .not. is_blockade(ip+1, jp, dimx, dimy)) then
                            gy_phi = (phi(pidx+1) - phi(pidx)) * inv_h
                        else
                            gy_phi = 0d0
                        end if
                    end if

                    ! time step
                    Ax_new(idx) = Ax(idx) + dt * &
                        ((kappa**2 * Ax_laplac(idx) + Jsx(pidx)) / sigma_n - gx_phi)
                    Ay_new(idx) = Ay(idx) + dt * &
                        ((kappa**2 * Ay_laplac(idx) + Jsy(pidx)) / sigma_n - gy_phi)

                    ! calculate Jn
                    Jnx(pidx) = -kappa**2 * Ax_laplac(idx) - Jsx(pidx)
                    Jny(pidx) = -kappa**2 * Ay_laplac(idx) - Jsy(pidx)

                    ! calculate E
                    Ex(pidx) = Jnx(pidx) / sigma_n
                    Ey(pidx) = Jny(pidx) / sigma_n

                    ! calculate dissipation
                    Pow(pidx) = Jnx(pidx)*Ex(pidx) + Jny(pidx)*Ey(pidx)

                ! solves for the vacuum domain
                else
                    Ax_new(idx) = Ax(idx) + dt * ((kappa**2 * Ax_laplac(idx)) / sigma_n)
                    Ay_new(idx) = Ay(idx) + dt * ((kappa**2 * Ay_laplac(idx)) / sigma_n)
                end if
            end do
        end do
        !$omp end parallel do

        ! stretch injection into the vacuum domain
        do ip = 1, dimy
            if (.not. is_blockade(ip, 1, dimx, dimy)) then
                do j = 1, padx
                    idx = (ip + pady) + (j-1) * 3*dimy
                    Ax_new(idx) = Ax_new(idx) + dt * (current / sigma_n)
                end do
            end if
            if (.not. is_blockade(ip, dimx, dimx, dimy)) then
                do j = padx+dimx+1, 3*dimy-1
                    idx = (ip + pady) + (j-1) * 3*dimy
                    Ax_new(idx) = Ax_new(idx) + dt * (current / sigma_n)
                end do
            end if
        end do

        ! neumann boundary condition
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 1, 3*dimy
            do i = 1, 3*dimy
                idx = i + (j-1) * 3*dimy
                if (i == 1) then
                    Ax_new(idx) = Ax_new(idx+1) ; Ay_new(idx) = Ay_new(idx+1)
                else if (i == 3*dimy) then
                    Ax_new(idx) = Ax_new(idx-1) ; Ay_new(idx) = Ay_new(idx-1)
                else if (j == 1) then
                    Ax_new(idx) = Ax_new(idx+3*dimy) ; Ay_new(idx) = Ay_new(idx+3*dimy)
                else if (j == 3*dimy) then
                    Ax_new(idx) = Ax_new(idx-3*dimy) ; Ay_new(idx) = Ay_new(idx-3*dimy)
                end if
            end do
        end do
        !$omp end parallel do

        Ax = Ax_new
        Ay = Ay_new

        deallocate(Ax_laplac, Ay_laplac, Ax_new, Ay_new)

    end subroutine calculate_A_Jn_E

    subroutine measure_div_A(dimx, dimy, h, Ax, Ay, max_div, mean_div, max_div_i, max_div_j)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: h, Ax(9*dimy*dimy), Ay(9*dimy*dimy)
        real(8), intent(out) :: max_div, mean_div
        integer, intent(out) :: max_div_i, max_div_j
        integer :: i, j, idx, n_int
        real(8) :: dAx_dx, dAy_dy, divA, inv_h, sum_abs

        inv_h = 1d0 / h
        max_div = 0d0
        sum_abs = 0d0
        n_int = 0
        max_div_i = 0
        max_div_j = 0

        do j = 2, 3*dimy-1
            do i = 2, 3*dimy-1
                idx = i + (j-1) * 3*dimy

                ! staggered grid divergence
                dAx_dx = (Ax(idx) - Ax(idx-3*dimy)) * inv_h
                dAy_dy = (Ay(idx) - Ay(idx-1)) * inv_h

                divA = dAx_dx + dAy_dy

                sum_abs = sum_abs + abs(divA)
                n_int = n_int + 1
                if (abs(divA) > max_div) then
                    max_div = abs(divA)
                    max_div_i = i
                    max_div_j = j
                end if
            end do
        end do

        mean_div = sum_abs / real(n_int, 8)

    end subroutine measure_div_A

    subroutine post_processing(dimx, dimy, Jsx, Jsy, Jnx, Jny, &
                               Jsx_plot, Jsy_plot, Jnx_plot, Jny_plot, Jtotx, Jtoty)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: Jsx(dimx*dimy), Jsy(dimx*dimy), Jnx(dimx*dimy), Jny(dimx*dimy)
        real(8), intent(out) :: Jsx_plot(dimx*dimy), Jsy_plot(dimx*dimy), Jnx_plot(dimx*dimy), Jny_plot(dimx*dimy)
        real(8), intent(out) :: Jtotx(dimx*dimy), Jtoty(dimx*dimy)
        integer :: i, j, idx
        real(8) :: sxR, sxL, syD, syU, nxR, nxL, nyD, nyU
        logical :: is_it

        Jsx_plot = 0d0 ; Jsy_plot = 0d0 ; Jnx_plot = 0d0 ; Jny_plot = 0d0
        Jtotx = 0d0 ; Jtoty = 0d0

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, is_it, sxR, sxL, syD, syU, nxR, nxL, nyD, nyU)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1) * dimy

                is_it = is_blockade(i, j, dimx, dimy)
                if (is_it) cycle

                ! average of +x and -x
                sxR = Jsx(idx) ; nxR = Jnx(idx)
                if (j > 1 .and. .not. is_blockade(i, j-1, dimx, dimy)) then
                    sxL = Jsx(idx-dimy) ; nxL = Jnx(idx-dimy)
                else
                    sxL = sxR ; nxL = nxR
                end if
                Jsx_plot(idx) = 0.5d0 * (sxR + sxL)
                Jnx_plot(idx) = 0.5d0 * (nxR + nxL)

                ! average of +y and -y
                syD = Jsy(idx) ; nyD = Jny(idx)
                if (i > 1 .and. .not. is_blockade(i-1, j, dimx, dimy)) then
                    syU = Jsy(idx-1) ; nyU = Jny(idx-1)
                else
                    syU = syD ; nyU = nyD
                end if
                Jsy_plot(idx) = 0.5d0 * (syD + syU)
                Jny_plot(idx) = 0.5d0 * (nyD + nyU)

                ! super and normal mean for total current at nodes
                Jtotx(idx) = 0.5d0 * ((Jsx(idx) + Jnx(idx)) + (sxL+nxL))
                Jtoty(idx) = 0.5d0 * ((Jsy(idx) + Jny(idx)) + (syU+nyU))
            end do
        end do
        !$omp end parallel do

    end subroutine post_processing

    subroutine compute_div_A_field(dimx, dimy, h, Ax, Ay, div_A)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: h, Ax(9*dimy*dimy), Ay(9*dimy*dimy)
        real(8), intent(out) :: div_A(9*dimy*dimy)
        integer :: i, j, idx
        real(8) :: dAx_dx, dAy_dy, inv_h

        inv_h = 1d0 / h
        div_A = 0d0

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, dAx_dx, dAy_dy)
        do j = 2, 3*dimy-1
            do i = 2, 3*dimy-1
                idx = i + (j-1) * 3*dimy
                dAx_dx = (Ax(idx) - Ax(idx - 3*dimy)) * inv_h
                dAy_dy = (Ay(idx) - Ay(idx - 1)) * inv_h
                div_A(idx) = dAx_dx + dAy_dy
            end do
        end do
        !$omp end parallel do

    end subroutine compute_div_A_field

    subroutine compute_div_Jtot_field(dimx, dimy, h, Jn_wall, Jsx, Jsy, Jnx, Jny, div_Jtot)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: h, Jsx(dimx*dimy), Jsy(dimx*dimy), Jnx(dimx*dimy), Jny(dimx*dimy), Jn_wall(dimy)
        real(8), intent(out) :: div_Jtot(dimx*dimy)
        integer :: i, j, idx
        real(8) :: dJtotx_dx, dJtoty_dy, inv_h
        real(8) :: Jtotx_temp(dimx*dimy), Jtoty_temp(dimx*dimy)

        inv_h = 1d0 / h
        div_Jtot = 0d0

        !$omp parallel do schedule(static)
        do idx = 1, dimx*dimy
            Jtotx_temp(idx) = Jsx(idx) + Jnx(idx)
            Jtoty_temp(idx) = Jsy(idx) + Jny(idx)
        end do
        !$omp end parallel do

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, dJtotx_dx, dJtoty_dy)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1)*dimy
                if (is_blockade(i, j, dimx, dimy)) cycle

                if (j == 1) then
                    dJtotx_dx = Jtotx_temp(idx) - Jn_wall(i)
                else
                    dJtotx_dx = Jtotx_temp(idx) - Jtotx_temp(idx-dimy)
                end if

                if (i == 1) then
                    dJtoty_dy = Jtoty_temp(idx)
                else
                    dJtoty_dy = Jtoty_temp(idx) - Jtoty_temp(idx-1)
                end if

                div_Jtot(idx) = (dJtotx_dx + dJtoty_dy) * inv_h
            end do
        end do
        !$omp end parallel do

    end subroutine compute_div_Jtot_field

end program solve_tdgl