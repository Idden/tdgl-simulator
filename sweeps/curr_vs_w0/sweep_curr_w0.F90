program solve_pde
#ifdef _OPENMP
    use omp_lib
#endif
    implicit none

    include 'mpif.h'
    include 'dmumps_struc.h'

    type(DMUMPS_STRUC) :: id
    integer :: ierr, nnz
    integer, parameter :: dimx = 60, dimy = 60
    integer, parameter :: n = dimx*dimy
    integer :: i, j, idx, n_open, curr
    integer, allocatable :: rows(:), cols(:)
    real(8), allocatable :: vals(:)

    ! sweep controls
    integer, parameter :: nW = 251
    integer, parameter :: nI = 201
    real(8), parameter :: W_min = 0.2d0, W_max = 0.9d0
    real(8), parameter :: H_fixed = 0.0d0
    real(8), parameter :: I_min = 0.07d0, I_max = 0.15d0
    real(8) :: W_list(nW)
    integer :: iW
    real(8) :: I_list(nI)
    integer :: iI
    real(8) :: w0_frac
    real(8) :: I_c
    integer :: t_c
    logical :: found
    integer :: task_id, n_tasks, ios
    character(len=32) :: arg
    character(len=64) :: res_file

    ! linspace for W
    do iW = 1, nW
        if (nW == 1) then
            W_list(iW) = W_min
        else
            W_list(iW) = W_min + (W_max - W_min) * real(iW-1, 8) / real(nW-1, 8)
        end if
    end do

    ! linspace for I
    do iI = 1, nI
        if (nI == 1) then
            I_list(iI) = I_min
        else
            I_list(iI) = I_min + (I_max - I_min) * real(iI-1, 8) / real(nI-1, 8)
        end if
    end do

    ! job array slicing
    task_id = 0 ; n_tasks = 1
    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg) ; read(arg, *, iostat=ios) task_id
        if (ios /= 0) task_id = 0
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg) ; read(arg, *, iostat=ios) n_tasks
        if (ios /= 0 .or. n_tasks < 1) n_tasks = 1
    end if

    ! set w0_frac
    iW = task_id + 1
    if (iW < 1 .or. iW > nW) then
        write(*,*) "task_id ", task_id, " out of range for nW = ", nW
        stop 1
    end if
    w0_frac = W_list(iW)

    ! make sure output dir exists
    call execute_command_line('mkdir -p data')

    ! turn on MPI + MUMPS
    call MPI_INIT(ierr)

    id%COMM = MPI_COMM_WORLD
    id%SYM = 1
    id%PAR = 1
    id%JOB = -1
    call DMUMPS(id)

    id%ICNTL(2) = 0
    id%ICNTL(3) = 0
    id%ICNTL(4) = 1

    ! build the sparse matrix once
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

    ! give the sparse matrix to MUMPS and factorize once
    nnz = curr - 1
    id%N = n
    id%NNZ = nnz
    allocate(id%IRN(nnz), id%JCN(nnz), id%A(nnz), id%RHS(n))
    id%IRN = rows(1:nnz)
    id%JCN = cols(1:nnz)
    id%A = vals(1:nnz)
    id%JOB = 1 ; call DMUMPS(id)
    id%JOB = 2 ; call DMUMPS(id)

    ! results table (one file per array task so tasks never overlap)
    write(res_file, '(A,I0,A)') "data/grid_task", task_id, ".txt"
    open(unit=100, file=trim(res_file), status="replace", action="write")
    write(100, '(A)') "#  iW    iI        w0_frac          I_curr      vortex   t_c"

    ! the sweep
    do iI = 1, nI
        call main_loop(dimx, dimy, iW, H_fixed, I_list(iI), I_c, t_c, found)
        write(100, '(I5,2X,I5,2X,ES15.7,2X,ES15.7,2X,I3,2X,I8)') &
            iW, iI, w0_frac, I_list(iI), merge(1, 0, found), t_c
        flush(100)
    end do
    close(100)

    ! turn off MUMPS + MPI
    deallocate(id%IRN, id%JCN, id%A, id%RHS)
    id%JOB = -2 ; call DMUMPS(id)
    call MPI_FINALIZE(ierr)
    deallocate(rows, cols, vals)

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
        real(8) :: ci, cj
        real(8) :: rect_half_w, gap_half

        ci = (dimy + 1) / 2d0
        cj = (dimx + 1) / 2d0

        rect_half_w = 0.05d0 * dimx       ! half width of blockade
        gap_half = w0_frac * 0.5d0 * dimy ! half height of open channel

        if (abs(real(j,8) - cj) <= rect_half_w .and. &
            abs(real(i,8) - ci) >  gap_half) then
            out = .true.
        else
            out = .false.
        end if
    end function is_blockade

    subroutine main_loop(dimx, dimy, iW, H_ext, current_max, I_c_out, t_c_out, found_out)
        real(8), parameter :: ymax = 30d0
        integer, parameter :: n_t = 40000
        real(8), parameter :: tmax = 400d0
        integer, parameter :: n_relax = 0, n_ramp = 7500
        integer, parameter :: H_relax = 0, H_ramp = n_relax
        integer, intent(in) :: iW, dimx, dimy
        real(8), intent(in) :: H_ext, current_max
        real(8), intent(out) :: I_c_out
        integer, intent(out) :: t_c_out
        logical, intent(out) :: found_out
        integer :: n, nA
        real(8), parameter :: kappa = 2.1d0, sigma_n = 1d0, gam = 0d0
        integer, parameter :: print_stride = 5000
        logical, parameter :: A_on = .true.
        real(8) :: h, dt, u
        complex(8), allocatable :: psi_init(:), psi_new(:), psi_old(:), psi_laplac(:)
        complex(8), allocatable :: G(:)
        integer :: t, i, j
        integer :: idx, psi_count
        real(8), allocatable :: phi(:)
        real(8) :: current_now, H_now
        logical :: is_it
        real(8), allocatable :: Jsx(:), Jsy(:), Jnx(:), Jny(:), Jn_wall(:), div_Js(:)
        real(8) :: avg_abs_psi, sum_abs_psi
        real(8), allocatable :: dreal(:), dimag(:)
        real(8), allocatable :: Ax(:), Ay(:)
        integer :: nv
        integer, parameter :: detect_stride = 200
        integer, parameter :: detect_start = n_ramp

        ! time keeping
        real(8) :: t0, twall0
        real(8) :: t_Js, t_div, t_phi, t_lap, t_A, t_total
        integer :: nthreads

        n = dimx*dimy
        nA = dimy*dimy

        allocate(psi_init(n), psi_new(n), psi_old(n), psi_laplac(n), G(n))
        allocate(phi(n), Jsx(n), Jsy(n), dreal(n), dimag(n), div_Js(n))
        allocate(Ax(9*nA), Ay(9*nA), Jnx(n), Jny(n), Jn_wall(dimy))

        ! default values if no vortex found
        found_out = .false. ; I_c_out = -1d0 ; t_c_out = -1

        h = ymax / (dimy - 1)
        dt = tmax / n_t
        u = 5.79d0
        phi = 0d0

        Ax = 0d0; Ay = 0d0; Jnx = 0d0; Jny = 0d0; Jn_wall = 0d0
        t_Js = 0d0; t_div = 0d0; t_phi = 0d0; t_lap = 0d0; t_A = 0d0

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
            "   grid = ", dimx, "x", dimy, " W", w0_frac, " Curr", current_max
#else
        write(*,'(A,I0, A,I0)') "SERIAL build (no OpenMP) | grid = ", dimx, "x", dimy
#endif

        twall0 = wtime()

        psi_init = (1d0, 0d0)
        psi_init = psi_init + 0.01d0 * cmplx(dreal - 0.5d0, dimag - 0.5d0, 8)

        !$omp parallel do collapse(2) private(idx, is_it) schedule(static)
        do j = 1, dimx
            do i = 1, dimy
                idx = i + (j-1) * dimy
                is_it = is_blockade(i, j, dimx, dimy)
                if (is_it) psi_init(idx) = (0d0, 0d0)
            end do
        end do
        !$omp end parallel do

        psi_old = psi_init

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

            t0 = wtime()
            call calculate_phi(dimx, dimy, h, div_Js, sigma_n, current_now, phi)
            t_phi = t_phi + (wtime() - t0)

            t0 = wtime()
            call calculate_laplacian(dimx, dimy, h, H_now, psi_old, Ax, Ay, psi_laplac)
            t_lap = t_lap + (wtime() - t0)

            t0 = wtime()
            if (A_on) call calculate_A_Jn(dimx, dimy, h, dt, sigma_n, kappa, current_now, &
                                          Jsx, Jsy, phi, Ax, Ay, Jnx, Jny, Jn_wall)
            t_A = t_A + (wtime() - t0)

            ! time step
            !$omp parallel do private(idx) schedule(static)
            do idx = 1, n
                G(idx) = (sqrt(1d0 + gam**2 * abs(psi_old(idx))**2) / u ) &
                    * (psi_laplac(idx) / h**2 + (1d0 - abs(psi_old(idx))**2) * psi_old(idx))

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

            ! check if vortex is found
            if (.not. found_out .and. t >= detect_start .and. mod(t, detect_stride) == 0) then
                nv = count_vortices(dimx, dimy, h, H_now, psi_new, Ax, Ay)
                if (nv >= 1) then
                    found_out = .true.
                    I_c_out = current_now
                    t_c_out = t
                    call write_snapshot(iW, current_max, dimx, dimy, psi_new)
                    write(*,'(A,I3,A,F9.5,A,F10.6,A,I7)') &
                        "  >> W#", iW, "  w0_frac=", w0_frac, "  first vortex at I=", current_now, "  t=", t
                    exit
                end if
            end if

            ! diagnostics
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

                write(*, '(A,I6, A,F8.5, A,F9.4, A,F7.4)') &
                    "t=",        t,                       &
                    "  I=",      current_now,             &
                    "  max phi=",  maxval(abs(phi)),      &
                    "  |psi|=",  avg_abs_psi
                print *

            end if

            psi_old = psi_new

        end do

        deallocate(psi_init, psi_new, psi_old, psi_laplac, G)
        deallocate(phi, Jsx, Jsy, dreal, dimag)
        deallocate(div_Js, Jn_wall)
        deallocate(Ax, Ay, Jnx, Jny)

        t_total = wtime() - twall0

        write(*,'(A)') "------------------------------------------------------------"
        write(*,'(A)') " TIMING (wall-clock seconds)"
        write(*,'(A,F12.4)')           "   total runtime        : ", t_total
        write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_Js         : ", t_Js,  "  (", 100d0*t_Js /t_total, " %)"
        write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_div_Js     : ", t_div, "  (", 100d0*t_div/t_total, " %)"
        write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_phi (MUMPS): ", t_phi, "  (", 100d0*t_phi/t_total, " %)"
        write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_laplacian  : ", t_lap, "  (", 100d0*t_lap/t_total, " %)"
        write(*,'(A,F12.4,A,F6.2,A)')  "   calculate_A          : ", t_A,   "  (", 100d0*t_A  /t_total, " %)"
        write(*,'(A)') "------------------------------------------------------------"
    end subroutine main_loop

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

    subroutine calculate_A_Jn(dimx, dimy, h, dt, sigma_n, kappa, current, Jsx, Jsy, phi, Ax, Ay, Jnx, Jny, Jn_wall)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: Jsx(dimx*dimy), Jsy(dimx*dimy), h, phi(dimx*dimy), sigma_n, dt, kappa, current
        real(8), intent(inout) :: Ax(9*dimy*dimy), Ay(9*dimy*dimy), Jnx(dimx*dimy), Jny(dimx*dimy)
        real(8), intent(out) :: Jn_wall(dimy)
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

    end subroutine calculate_A_Jn

    function count_vortices(dimx, dimy, h, H_ext, psi, Ax, Ay) result(nv)
        implicit none
        integer, intent(in) :: dimx, dimy
        real(8), intent(in) :: h, H_ext, Ax(9*dimy*dimy), Ay(9*dimy*dimy)
        complex(8), intent(in) :: psi(dimx*dimy)
        integer :: nv
        integer :: i, j, padx, pady, A1, A2, A4
        real(8) :: wrapping, Hj, Hj1
        complex(8) :: z1, z2, z3, z4, U12, U23, U34, U41
        real(8), parameter :: twopi = 6.283185307179586d0
        real(8), parameter :: psi_min = 0.05d0

        nv = 0
        padx = (3*dimy - dimx) / 2
        pady = dimy

        !$omp parallel do collapse(2) schedule(static) &
        !$omp private(z1,z2,z3,z4,U12,U23,U34,U41,wrapping,Hj,Hj1,A1,A2,A4) &
        !$omp reduction(+:nv)
        do j = 1, dimx-1
            do i = 1, dimy-1
                if (is_blockade(i, j, dimx, dimy)) cycle
                if (is_blockade(i+1, j, dimx, dimy)) cycle
                if (is_blockade(i+1, j+1, dimx, dimy)) cycle
                if (is_blockade(i, j+1, dimx, dimy)) cycle

                z1 = psi(i + (j-1)*dimy)    ! (i, j)
                z2 = psi(i+1 + (j-1)*dimy)  ! (i+1, j)
                z3 = psi(i+1 + j*dimy)      ! (i+1, j+1)
                z4 = psi(i + j*dimy)        ! (i, j+1)

                if (abs(z1) < psi_min .or. abs(z2) < psi_min .or. &
                    abs(z3) < psi_min .or. abs(z4) < psi_min) cycle

                A1 = (i + pady) + (j + padx - 1) * 3*dimy     ! A index at (i, j)
                A2 = (i+1 + pady) + (j + padx - 1) * 3*dimy   ! A index at (i+1, j)
                A4 = (i + pady) + (j+1 + padx - 1) * 3*dimy   ! A index at (i, j+1)

                Hj = H_ext * (real(j, 8) - (dimx+1)/2d0) * h
                Hj1 = H_ext * (real(j+1, 8) - (dimx+1)/2d0) * h

                U12 = exp(-(0d0,1d0) * (Hj + Ay(A1)) * h)    ! (i,j) -> (i+1,j)
                U23 = exp(-(0d0,1d0) * Ax(A2) * h)           ! (i+1,j) -> (i+1,j+1)
                U34 = exp((0d0,1d0) * (Hj1 + Ay(A4)) * h)    ! (i+1,j+1) -> (i,j+1)
                U41 = exp((0d0,1d0) * Ax(A1) * h)            ! (i,j+1) -> (i,j)

                wrapping = atan2(aimag(conjg(z1)*z2*U12), real(conjg(z1)*z2*U12)) &
                         + atan2(aimag(conjg(z2)*z3*U23), real(conjg(z2)*z3*U23)) &
                         + atan2(aimag(conjg(z3)*z4*U34), real(conjg(z3)*z4*U34)) &
                         + atan2(aimag(conjg(z4)*z1*U41), real(conjg(z4)*z1*U41))

                if (abs(nint(wrapping / twopi)) >= 1) nv = nv + 1
            end do
        end do
        !$omp end parallel do
    end function count_vortices

    subroutine write_snapshot(iW, current, dimx, dimy, psi)
        implicit none
        integer, intent(in) :: iW, dimx, dimy
        real(8), intent(in) :: current
        complex(8), intent(in) :: psi(dimx*dimy)
        character(len=64) :: fname

        call execute_command_line('mkdir -p data/snap')
        write(fname, '(A,I0,A,I0,A)') "data/snap/psi_W", iW, "_I", nint(current*1d4), ".bin"
        open(unit=67, file=trim(fname), status="replace", form="unformatted", access="stream")
        write(67) psi
        close(67)
    end subroutine write_snapshot

end program solve_pde