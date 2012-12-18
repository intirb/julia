libBLAS = dlopen("libopenblas")
load("qr_blas.jl")
load("blastmp.jl")
load("pqr.jl")
# julia/matlab column major
# c row major
# pb*qb


# 0. Initialization
id_test=4;
if id_test==0
    # test blas functions (3 iterations） 
    A = float([4  2 ;  2  6]);
    tile_sz=2;
elseif id_test==1
    # test blas functions (3 iterations） 
    A = float([3  2 1 2;1 2 2 3;4 1 1 5; 4 2 3 6]);
    tile_sz=2;
elseif id_test==2
    # test block structure (transpose?row major?） 
    A = rand(10,10);
    tile_sz=2;
elseif id_test==3
    # test 5*5 matrix (additional row/column)
    A = float([4  2  5  3 1;  2  6  2  5 2 ;5  2 12  2 1;  3  5  2 14 2;1 2 1 2 10]);
    tile_sz=2;
else
    # test large matrix
    A = float(randn(16000,16000));
    A = A*A';
    tile_sz = 4000;
end
println("Initialization:")
println("A = ")
println(A)


println("Output:")
qr(A)
println("julia qr: qr(A) = ")
tic()
println(qr(A))
toc()

qr_blas(A,tile_sz)
println("openblas qr: qr(A) = ")
tic()
println(qr_blas(A,tile_sz))
toc()

pqr(A,tile_sz)
println("parallel openblas qr: qr(A) = ")
tic()
println(qr_blas(A,tile_sz))
toc()


#http://www.netlib.org/lapack/lapack_routine/dgeqrt.f
#DTSQRT:
#a) PLASMA/build/plasma_2.4.6/core_blas/core_dtsqrt.c
#b) dtpqrt with L=0 [1]
#http://www.netlib.org/lapack/double/dormqr.f                                   
#DSSMQR:
#a) PLASMA/build/plasma_1.4.6/core_blas/core_dssmqr.c
#b) dtpqrt with L=0 [1]
                                   
#[1] (http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg01269.html)

#[2] www.netlib.org/lapack/lawnspdf/lawn190.pdf
#[3] www.netlib.org/lapack/lawnspdf/lawn191.pdf
