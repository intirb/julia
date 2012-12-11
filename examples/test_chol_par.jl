@everywhere libBLAS = dlopen("libopenblas")
# julia/matlab column major
# c row major

load("pchol.jl")
load("chol_blas.jl")
@everywhere load("blastmp.jl")

# 0. Initialization
id_test=4;
if id_test==1
    # test blas functions (3 iterations)
    A = float([4  2  5  ;  2  6  2  ;5  2 12]);
    tile_sz=1;
elseif id_test==2
    # test block structure (transpose?row major?) 
    A = float([4  2  5  3;  2  6  2  5  ;5  2 12  2;  3  5  2 14]);
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

B = A;
C = A;
D = A;

println("Initialization:")
println("A = ")
println(A)

println("Output:")
println("julia chol: chol(A) = ")
chol(A); #I'm running it twice because the second time always runs faster for some reason
tic()
println(chol(B)')
toc()

println("openblas chol: chol(A) = ")
chol_blas(A,tile_sz);
tic()
println(chol_blas(C,tile_sz))
toc()

println("parallel openblas chol: chol(A) = ")
pchol(A,tile_sz);
tic()
println(pchol(D,tile_sz))
toc()
#http://www.netlib.org/clapack/clapack-3.2.1-CMAKE/SRC/VARIANTS/cholesky/TOP/dpotrf.c
#http://www.netlib.org/clapack/cblas/dtrsm.c
#http://www.netlib.org/clapack/cblas/dsyrk.c
#http://www.netlib.org/clapack/cblas/dgemm.c
