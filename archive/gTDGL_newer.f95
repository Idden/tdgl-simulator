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
    integer, parameter :: n_t = 30000
    real(8), parameter :: tmax = 300d0
    integer, parameter :: n_relax = 5000, n_ramp = 5000
    integer, parameter :: H_relax = 0, H_ramp = n_relax
    integer, parameter :: stride = 200
    real(8), parameter :: kappa = 2.1d0, sigma_n = 1d0, gam = 1d0
    real(8), parameter :: H_ext = 0.0353d0, current_max = 0.15d0
    integer, parameter :: print_stride = 500
    logical, parameter :: A_on = .true.
    real(8) :: h, dt, u
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

    real(8), allocatable :: div_A(:)

    ! time keeping
    real(8) :: t0, twall0
    real(8) :: t_Js, t_div, t_phi, t_lap, t_io, t_A, t_total
    integer :: nthreads

    allocate(psi_init(n), psi_new(n), psi_old(n), psi_laplac(n), G(n))
    allocate(phi(n), Jsx(n), Jsy(n), Jtotx(n), Jtoty(n), dreal(n), dimag(n))
    allocate(div_Js_diff_LHS(n), div_Js_diff_RHS(n), div_Js(n), div_Js_LHS(n), div_Js_RHS(n))
    allocate(Ax(4*n), Ay(4*n), Jnx(n), Jny(n))
    allocate(Jnx_plot(n), Jny_plot(n), Jsx_plot(n), Jsy_plot(n))
    allocate(div_A(4*n))

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

    call random_number(dreal)
    call random_number(dimag)

    nthreads = 1
#ifdef _OPENMP
    !$omp parallel
    !$omp masked
    nthreads = omp_get_num_threads()
    !$omp end masked
    !$omp end parallel
    write(*,'(A,I0,A,I0,A,I0)') "OpenMP build | threads = ", nthreads, &
        "   grid = ", dim, "x", dim!, "Amp", A_on, "Mag", H_ext, "Curr", current_max
#else
    write(*,'(A,I0,A,I0)') "SERIAL build (no OpenMP) | grid = ", dim, "x", dim
#endif

    twall0 = wtime()

    psi_init = (1d0, 0d0)
    psi_init = psi_init + 0.01d0 * cmplx(dreal - 0.5d0, dimag - 0.5d0, 8)

    !$omp parallel do collapse(2) private(idx, is_it) schedule(static)
    do j = 1, dim
        do i = 1, dim
            idx = i + (j-1) * dim
            is_it = is_blockade(i, j, dim)
            if (is_it) psi_init(idx) = (0d0, 0d0)
        end do
    end do
    !$omp end parallel do

    open(unit=10, file="data/psi.bin", status="replace", form="unformatted", access="stream")
    open(unit=20, file="data/Js.bin", status="replace", form="unformatted", access="stream")
    open(unit=30, file="data/div_Js.bin", status="replace", form="unformatted", access="stream")
    open(unit=40, file="data/phi.bin", status="replace", form="unformatted", access="stream")
    open(unit=50, file="data/vec_pot.bin", status="replace", form="unformatted", access="stream")
    open(unit=60, file="data/Jn.bin", status="replace", form="unformatted", access="stream")
    open(unit=70, file="data/Jtot.bin", status="replace", form="unformatted", access="stream")
    open(unit=80, file="data/div_A.bin", status="replace", form="unformatted", access="stream")

    psi_old = psi_init
    write(10) psi_old

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
        call calculate_Js(dim, h, H_now, psi_old, Ax, Ay, Jsx, Jsy)
        t_Js = t_Js + (wtime() - t0)

        t0 = wtime()
        call calculate_div_Js(dim, h, H_now, psi_old, Ax, Ay, div_Js)
        t_div = t_div + (wtime() - t0)

        t0 = wtime()
        call calculate_phi(dim, h, div_Js, sigma_n, current_now, phi)
        t_phi = t_phi + (wtime() - t0)

        t0 = wtime()
        call calculate_laplacian(dim, h, H_now, psi_old, Ax, Ay, psi_laplac)
        t_lap = t_lap + (wtime() - t0)

        t0 = wtime()
        if (A_on) call calculate_A_Jn(dim, h, dt, sigma_n, kappa, Jsx, Jsy, phi, Ax, Ay, Jnx, Jny)
        t_A = t_A + (wtime() - t0)

        ! time step
        !$omp parallel do private(idx) schedule(static)
        do idx = 1, n
            G(idx) = (sqrt(1d0 + gam**2 * abs(psi_old(idx))**2) / u ) &
                   * (psi_laplac(idx) / h**2 + (1d0 - abs(psi_old(idx))**2) * psi_old(idx)) !&
                   !- (0d0, 1d0) * phi(idx) * psi_old(idx)

            psi_new(idx) = (psi_old(idx) + dt * ( &
                G(idx) - (gam**2 * psi_old(idx) / (1d0 + abs(psi_old(idx))**2 * gam**2)) &
                * real(conjg(psi_old(idx)) * G(idx)) &
            )) * exp(-(0d0, 1d0) * phi(idx) * dt)
        end do
        !$omp end parallel do

        ! re-zero blockade sites
        !$omp parallel do collapse(2) private(idx) schedule(static)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1)*dim
                if (is_blockade(i, j, dim)) psi_new(idx) = (0d0, 0d0)
            end do
        end do
        !$omp end parallel do

        ! save link values as node values for plotting on grid
        call post_processing(dim, Jsx, Jsy, Jnx, Jny, &
                             Jsx_plot, Jsy_plot, Jnx_plot, Jny_plot, Jtotx, Jtoty)

        t0 = wtime()
        if (mod(t, stride) == 0) write(10) psi_new
        if (mod(t, stride) == 0) write(20) Jsx_plot, Jsy_plot
        if (mod(t, stride) == 0) write(30) div_Js
        if (mod(t, stride) == 0) write(40) phi
        if (mod(t, stride) == 0) write(50) Ax, Ay
        if (mod(t, stride) == 0) write(60) Jnx_plot, Jny_plot
        if (mod(t, stride) == 0) write(70) Jtotx, Jtoty
        if (mod(t, stride) == 0) then
            call compute_div_A_field(dim, h, Ax, Ay, div_A)
            write(80) div_A
        end if
        t_io = t_io + (wtime() - t0)

        ! check for current conservation in div Js
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                div_Js_LHS(idx) = (u / sqrt(1d0 + gam**2 * abs(psi_old(idx))**2)) * &
                                  (aimag(conjg(psi_old(idx)) * (psi_new(idx) - psi_old(idx))) / dt &
                                  + phi(idx) * abs(psi_old(idx))**2)

                div_Js_RHS(idx) = aimag(conjg(psi_old(idx)) * psi_laplac(idx)) / h**2

                div_Js_diff_LHS(idx) = div_Js(idx) - div_Js_LHS(idx)
                div_Js_diff_RHS(idx) = div_Js(idx) - div_Js_RHS(idx)
            end do
        end do
        !$omp end parallel do

        ! diagnostics
        if (mod(t, print_stride) == 0) then
            sum_abs_psi = 0d0
            psi_count = 0
            do j = 1, dim
                do i = 1, dim
                    idx = i + (j-1)*dim
                    if (.not. is_blockade(i, j, dim)) then
                        sum_abs_psi = sum_abs_psi + abs(psi_old(idx))
                        psi_count = psi_count + 1
                    end if
                end do
            end do
            avg_abs_psi = sum_abs_psi / real(psi_count, 8)

            call measure_div_A(dim, h, Ax, Ay, max_div_A, mean_div_A, div_i, div_j)
            write(*, '(A,ES11.3, A,ES11.3, A,I4,A,I4,A)') &
                "   max|divA|=", max_div_A, &
                "  mean|divA|=", mean_div_A, &
                "  at (", div_i, ",", div_j, ")"

            write(*, '(A,I6, A,F8.5, A,F7.4, A,F9.4, A,F7.4, A,F7.4, A,F7.4)') &
                "t=",        t,                       &
                "  I=",      current_now,             &
                "  I/Jd=",   current_now/0.3849d0,    &
                "  max phi=",  maxval(abs(phi)),      &
                "  |psi|=",  avg_abs_psi,             &
                ! "  max psi=",maxval(abs(psi_old)),    &
                "  L divJs=", maxval(abs(div_Js_diff_LHS)),&
                "  R divJs=", maxval(abs(div_Js_diff_RHS))

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

    deallocate(id%IRN, id%JCN, id%A, id%RHS)
    id%JOB = -2 ; call DMUMPS(id)
    call MPI_FINALIZE(ierr)

    deallocate(psi_init, psi_new, psi_old, psi_laplac, G)
    deallocate(phi, Jsx, Jsy, dreal, dimag, Jtotx, Jtoty)
    deallocate(div_Js_diff_LHS, div_Js_diff_RHS, div_Js, div_Js_LHS, div_Js_RHS)
    deallocate(Ax, Ay, Jnx, Jny)
    deallocate(Jnx_plot, Jny_plot, Jsx_plot, Jsy_plot)
    deallocate(div_A)

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

    subroutine calculate_laplacian(dim, h, H_ext, psi, Ax, Ay, psi_laplac)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: h, H_ext, Ax(4*dim**2), Ay(4*dim**2)
        complex(8), intent(in) :: psi(dim**2)
        complex(8), intent(out) :: psi_laplac(dim**2)
        integer :: i, j, idx, A_idx, pad, n_open
        real(8) :: H_term
        complex(8) :: Uy, Uy_conj, Ux, Ux_conj, psi_val
        logical :: is_it

        psi_laplac = (0d0, 0d0)
        pad = dim / 2

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, A_idx, is_it, Uy, Uy_conj, Ux, Ux_conj, psi_val, n_open, H_term)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                is_it = is_blockade(i, j, dim)
                if (is_it) cycle

                A_idx = (i + pad) + (j + pad - 1) * 2*dim
                H_term = H_ext * (real(j,8) - (dim+1)/2d0) * h

                Uy = exp(-(0d0,1d0) * (H_term + Ay(A_idx)) * h)
                Uy_conj = exp((0d0,1d0) * (H_term + Ay(A_idx-1)) * h)
                Ux = exp(-(0d0,1d0) * Ax(A_idx) * h)
                Ux_conj = exp((0d0,1d0) * Ax(A_idx-2*dim) * h)

                psi_val = (0d0, 0d0)
                n_open = 0

                if (i > 1 .and. .not. is_blockade(i-1, j, dim)) then
                    psi_val = psi_val + psi(idx-1) * Uy_conj
                    n_open = n_open + 1
                end if
                if (i < dim .and. .not. is_blockade(i+1, j, dim)) then
                    psi_val = psi_val + psi(idx+1) * Uy
                    n_open = n_open + 1
                end if
                if (j > 1 .and. .not. is_blockade(i, j-1, dim)) then
                    psi_val = psi_val + psi(idx-dim) * Ux_conj
                    n_open = n_open + 1
                end if
                if (j < dim .and. .not. is_blockade(i, j+1, dim)) then
                    psi_val = psi_val + psi(idx+dim) * Ux
                    n_open = n_open + 1
                end if

                psi_laplac(idx) = psi_val - real(n_open, 8) * psi(idx)
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

    subroutine calculate_Js(dim, h, H_ext, psi, Ax, Ay, Jsx, Jsy)
        implicit none
        integer, intent(in) :: dim
        complex(8), intent(in) :: psi(dim**2)
        real(8), intent(in) :: h, H_ext, Ax(4*dim**2), Ay(4*dim**2)
        real(8), intent(out) :: Jsx(dim**2), Jsy(dim**2)
        integer :: i, j, idx, A_idx, pad
        real(8) :: H_term, inv_h, g_i, g_j
        complex(8) :: U
        logical :: is_it

        Jsx = 0d0 ; Jsy = 0d0
        pad = dim / 2
        inv_h = 1d0 / h

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, A_idx, H_term, U, g_i, g_j, is_it)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                is_it = is_blockade(i, j, dim)
                if (is_it) cycle

                A_idx = (i + pad) + (j + pad - 1) * 2*dim
                H_term = H_ext * (real(j,8) - (dim+1)/2d0) * h

                g_i = 0d0 ; g_j = 0d0

                ! calculate Jsy
                if (i < dim .and. .not. is_blockade(i+1, j, dim)) then
                    U = exp(-(0d0,1d0) * (H_term + Ay(A_idx)) * h)
                    g_i = aimag(conjg(psi(idx)) * U * psi(idx+1)) * inv_h
                end if

                ! calculate Jsx
                if (j < dim .and. .not. is_blockade(i, j+1, dim)) then
                    U = exp(-(0d0,1d0) * Ax(A_idx) * h)
                    g_j = aimag(conjg(psi(idx)) * U * psi(idx+dim)) * inv_h
                end if

                Jsy(idx) = g_i
                Jsx(idx) = g_j

            end do
        end do
        !$omp end parallel do

    end subroutine calculate_Js

    subroutine calculate_div_Js(dim, h, H_ext, psi, Ax, Ay, div_Js)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: h, H_ext, Ax(4*dim**2), Ay(4*dim**2)
        complex(8), intent(in) :: psi(dim**2)
        real(8), intent(out) :: div_Js(dim**2)
        integer :: i, j, idx, A_idx, pad
        real(8) :: div, Ay_e, Ax_e, H_term, inv_h2
        complex(8) :: U
        logical :: is_it

        div_Js = 0d0
        pad = dim / 2
        inv_h2 = 1d0 / h**2

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, A_idx, div, Ay_e, Ax_e, H_term, U, is_it)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                is_it = is_blockade(i, j, dim)
                if (is_it) cycle

                A_idx = (i + pad) + (j + pad - 1) * 2*dim
                H_term = H_ext * (real(j,8) - (dim+1)/2d0) * h
                div = 0d0

                if (i < dim .and. .not. is_blockade(i+1, j, dim)) then
                    U = exp(-(0d0, 1d0) * (H_term + Ay(A_idx)) * h)
                    div = div + aimag(conjg(psi(idx)) * U * psi(idx+1))
                end if
                if (i > 1 .and. .not. is_blockade(i-1, j, dim)) then
                    U = exp((0d0, 1d0) * (H_term + Ay(A_idx - 1)) * h)
                    div = div + aimag(conjg(psi(idx)) * U * psi(idx-1))
                end if

                if (j < dim .and. .not. is_blockade(i, j+1, dim)) then
                    U = exp(-(0d0, 1d0) * Ax(A_idx) * h)
                    div = div + aimag(conjg(psi(idx)) * U * psi(idx+dim))
                end if
                if (j > 1 .and. .not. is_blockade(i, j-1, dim)) then
                    U = exp((0d0, 1d0) * Ax(A_idx - 2*dim) * h)
                    div = div + aimag(conjg(psi(idx)) * U * psi(idx-dim))
                end if

                div_Js(idx) = div * inv_h2
            end do
        end do
        !$omp end parallel do

    end subroutine calculate_div_Js

    subroutine calculate_A_Jn(dim, h, dt, sigma_n, kappa, Jsx, Jsy, phi, Ax, Ay, Jnx, Jny)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: Jsx(dim**2), Jsy(dim**2), h, phi(dim**2), sigma_n, dt, kappa
        real(8), intent(inout) :: Ax(4*dim**2), Ay(4*dim**2), Jnx(dim**2), Jny(dim**2)
        integer :: i, j, idx, pad
        integer :: ip, jp, pidx
        real(8) :: gx_phi, gy_phi, inv_h2, inv_h
        real(8), allocatable :: Ax_laplac(:), Ay_laplac(:), Ax_new(:), Ay_new(:)
        logical :: is_it

        allocate(Ax_laplac(4*dim**2), Ay_laplac(4*dim**2), Ax_new(4*dim**2), Ay_new(4*dim**2))

        inv_h = 1d0 / h
        inv_h2 = 1 / h**2
        gx_phi = 0d0 ; gy_phi = 0d0
        Ax_laplac = 0d0 ; Ay_laplac = 0d0
        pad = dim / 2

        ! calculate laplacian of A
        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx)
        do j = 2, 2*dim-1
            do i = 2, 2*dim-1
                idx = i + (j-1) * 2*dim

                Ax_laplac(idx) = (Ax(idx-1) + Ax(idx+1) &
                                 + Ax(idx-2*dim) + Ax(idx+2*dim) &
                                 - 4d0 * Ax(idx)) * inv_h2

                Ay_laplac(idx) = (Ay(idx-1) + Ay(idx+1) &
                                 + Ay(idx-2*dim) + Ay(idx+2*dim) &
                                 - 4d0 * Ay(idx)) * inv_h2
            end do
        end do
        !$omp end parallel do

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, pidx, ip, jp, gx_phi, gy_phi)
        do j = 2, 2*dim-1
            do i = 2, 2*dim-1
                idx = i + (j-1) * 2*dim

                ! solves for the superconducting domain
                if (i > pad .and. i <= 2*dim-pad .and. j > pad .and. j <= 2*dim-pad) then
                    ip = i - pad ; jp = j - pad
                    pidx = ip + (jp-1)*dim

                    ! gradient of phi
                    if (jp < dim .and. .not. is_blockade(ip, jp+1, dim)) then
                        gx_phi = (phi(pidx+dim) - phi(pidx)) * inv_h
                    else
                        gx_phi = 0d0
                    end if
                    if (ip < dim .and. .not. is_blockade(ip+1, jp, dim)) then
                        gy_phi = (phi(pidx+1) - phi(pidx)) * inv_h
                    else
                        gy_phi = 0d0
                    end if

                    ! time step
                    Ax_new(idx) = Ax(idx) + dt * &
                        ((kappa**2 * Ax_laplac(idx) + Jsx(pidx)) / sigma_n - gx_phi)
                    Ay_new(idx) = Ay(idx) + dt * &
                        ((kappa**2 * Ay_laplac(idx) + Jsy(pidx)) / sigma_n - gy_phi)

                    Jnx(pidx) = -kappa**2 * Ax_laplac(idx) - Jsx(pidx)
                    Jny(pidx) = -kappa**2 * Ay_laplac(idx) - Jsy(pidx)

                ! solves for the vacuum domain
                else
                    Ax_new(idx) = Ax(idx) + dt * ((kappa**2 * Ax_laplac(idx)) / sigma_n)
                    Ay_new(idx) = Ay(idx) + dt * ((kappa**2 * Ay_laplac(idx)) / sigma_n)
                end if
            end do
        end do
        !$omp end parallel do

        ! enforce dirichlet boundary conditions for vacuum domain
        !$omp parallel do collapse(2) schedule(static) private(idx)
        do j = 1, 2*dim
            do i = 1, 2*dim
                if (i == 1 .or. i == 2*dim .or. j == 1 .or. j == 2*dim) then
                    idx = i + (j-1) * 2*dim
                    Ax_new(idx) = 0d0 ; Ay_new(idx) = 0d0
                end if
            end do
        end do
        !$omp end parallel do

        Ax = Ax_new
        Ay = Ay_new

        deallocate(Ax_laplac, Ay_laplac, Ax_new, Ay_new)

    end subroutine calculate_A_Jn

    subroutine measure_div_A(dim, h, Ax, Ay, max_div, mean_div, max_div_i, max_div_j)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: h, Ax(4*dim**2), Ay(4*dim**2)
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

        do j = 2, 2*dim-1
            do i = 2, 2*dim-1
                idx = i + (j-1) * 2*dim

                ! staggered grid divergence
                dAx_dx = (Ax(idx) - Ax(idx - 2*dim)) * inv_h
                dAy_dy = (Ay(idx) - Ay(idx - 1)) * inv_h

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

    subroutine post_processing(dim, Jsx, Jsy, Jnx, Jny, &
                               Jsx_plot, Jsy_plot, Jnx_plot, Jny_plot, Jtotx, Jtoty)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: Jsx(dim**2), Jsy(dim**2), Jnx(dim**2), Jny(dim**2)
        real(8), intent(out) :: Jsx_plot(dim**2), Jsy_plot(dim**2), Jnx_plot(dim**2), Jny_plot(dim**2)
        real(8), intent(out) :: Jtotx(dim**2), Jtoty(dim**2)
        integer :: i, j, idx
        real(8) :: sxR, sxL, syD, syU, nxR, nxL, nyD, nyU
        logical :: is_it

        Jsx_plot = 0d0 ; Jsy_plot = 0d0 ; Jnx_plot = 0d0 ; Jny_plot = 0d0
        Jtotx = 0d0 ; Jtoty = 0d0

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, is_it, sxR, sxL, syD, syU, nxR, nxL, nyD, nyU)
        do j = 1, dim
            do i = 1, dim
                idx = i + (j-1) * dim

                is_it = is_blockade(i, j, dim)
                if (is_it) cycle

                ! average of +x link (idx) and -x link (idx-dim)
                sxR = Jsx(idx) ; nxR = Jnx(idx)
                if (j > 1 .and. .not. is_blockade(i, j-1, dim)) then
                    sxL = Jsx(idx-dim) ; nxL = Jnx(idx-dim)
                else
                    sxL = sxR ; nxL = nxR
                end if
                Jsx_plot(idx) = 0.5d0 * (sxR + sxL)
                Jnx_plot(idx) = 0.5d0 * (nxR + nxL)

                ! average of +y link (idx) and -y link (idx-1)
                syD = Jsy(idx) ; nyD = Jny(idx)
                if (i > 1 .and. .not. is_blockade(i-1, j, dim)) then
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

    subroutine compute_div_A_field(dim, h, Ax, Ay, div_A)
        implicit none
        integer, intent(in) :: dim
        real(8), intent(in) :: h, Ax(4*dim**2), Ay(4*dim**2)
        real(8), intent(out) :: div_A(4*dim**2)
        integer :: i, j, idx
        real(8) :: dAx_dx, dAy_dy, inv_h

        inv_h = 1d0 / h
        div_A = 0d0

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(idx, dAx_dx, dAy_dy)
        do j = 2, 2*dim-1
            do i = 2, 2*dim-1
                idx = i + (j-1) * 2*dim
                dAx_dx = (Ax(idx) - Ax(idx - 2*dim)) * inv_h
                dAy_dy = (Ay(idx) - Ay(idx - 1)) * inv_h
                div_A(idx) = dAx_dx + dAy_dy
            end do
        end do
        !$omp end parallel do

    end subroutine compute_div_A_field

end program solve_pde