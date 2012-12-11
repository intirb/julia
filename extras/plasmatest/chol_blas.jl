function chol_blas(A,tile_sz)

# 1. Set up Data Structure
tile_sz2 =	tile_sz^2;
mat_sz   =	size(A,1);
bd_sz    =	tile_sz;
num_tile =	int(ceil(mat_sz/tile_sz));
upper = repmat(linspace(1,tile_sz,tile_sz)',tile_sz,1).>repmat(linspace(1,tile_sz,tile_sz),1,tile_sz);
tile_upper =	find(upper);
bd_upper   =	tile_upper;

#A_para is a cell, where each cell is the corresponding tile of matrix A
A_para = cell(num_tile,num_tile);
for i=1:num_tile-1
    for j=1:i
        A_para[i,j]=reshape(A[(i-1)*tile_sz+(1:tile_sz),(j-1)*tile_sz+(1:tile_sz)],1,tile_sz2);
    end
end

if tile_sz*num_tile>mat_sz
    bd_sz = mat_sz-(num_tile-1)*tile_sz;
    bd_upper=find(upper[1:bd_sz,1:bd_sz]);
    for j=1:num_tile-1
        A_para[num_tile,j]=reshape(A[(mat_sz-bd_sz+1):mat_sz,(j-1)*tile_sz+(1:tile_sz)],1,tile_sz*bd_sz);
    end
    A_para[num_tile,num_tile]=reshape(A[(mat_sz-bd_sz+1):mat_sz,(mat_sz-bd_sz+1):mat_sz],1,bd_sz*bd_sz);
else
    for j=1:num_tile
        A_para[num_tile,j]=reshape(A[(num_tile-1)*tile_sz+(1:tile_sz),(j-1)*tile_sz+(1:tile_sz)],1,tile_sz2);
    end
end

c_uplo   =	'L'; 
c_side   =	'R';
c_transt =	'T';
c_transn =	'N';
c_diag   =	'N';
c_alphap =	1.0;
c_alpham =	-1.0;
c_major  =	102;
c_info   =	0;

# some maths...
# A= LL'where L is lower diagnoal
# let L =[L11,0;L21,L22], then A=[L11*L11',L11*L21';L21*L11',L21*L21'+L22*L22']
# k=1;  2.1 solve L11 (with A11); 2.2 solve L21 (with L11,A21); 2.3/2.4 solve L22 (second iter)

# 2. Computation
# 2.1 DPOTRF

for k=1:num_tile-1
    ccall(dlsym(libBLAS, :dpotrf_),Void,
	(Ptr{Uint8}, Ptr{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
	&c_uplo, &tile_sz, A_para[k,k], &tile_sz, &c_info)
    A_para[k,k][tile_upper]=0;
#    println(A_para[k,k]);

    # 2.2 DTRSM

    for m=k+1:num_tile-1
        ccall(dlsym(libBLAS, :dtrsm_),Void,
	   (Ptr{Uint8},Ptr{Uint8},Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Float64},
		Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Int32}),
	 &c_side, &c_uplo,&c_transt,&c_diag, &tile_sz,&tile_sz,&c_alphap,A_para[k,k],&tile_sz,A_para[m,k], &tile_sz)
    end
    ccall(dlsym(libBLAS, :dtrsm_),Void,
	(Ptr{Uint8},Ptr{Uint8},Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Float64},
		Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Int32}),
	&c_side, &c_uplo,&c_transt,&c_diag, &bd_sz,&tile_sz,&c_alphap,A_para[k,k],&tile_sz,A_para[num_tile,k], &bd_sz)

    # 2.3 DSYRK

    for n=k+1:num_tile-1
        ccall(dlsym(libBLAS, :dsyrk_),Void,
	    (Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Float64},Ptr{Float64},Ptr{Int32},
		Ptr{Float64},Ptr{Float64},Ptr{Int32}),
	    &c_uplo,&c_transn, &tile_sz,&tile_sz,&c_alpham,A_para[n,k],&tile_sz,&c_alphap,A_para[n,n], &tile_sz)

        # 2.4 DGEMM

        for m=n+1:num_tile-1
            ccall(dlsym(libBLAS, :dgemm_),Void,(
		Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Float64},Ptr{Float64},
		    Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Float64},Ptr{Int32}),
		&c_transn,&c_transt,&tile_sz, &tile_sz,&tile_sz,&c_alpham,A_para[m,k],&tile_sz,
		    A_para[n,k],&tile_sz,&c_alphap,A_para[m,n], &tile_sz)
        end
        ccall(dlsym(libBLAS, :dgemm_),Void,
		(Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Float64},
			Ptr{Float64}, Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Float64},Ptr{Int32}),
		&c_transn,&c_transt,&bd_sz, &tile_sz,&tile_sz,&c_alpham,A_para[num_tile,k],
			&bd_sz,A_para[n,k],&tile_sz,&c_alphap,A_para[num_tile,n], &bd_sz)
    end
    ccall(dlsym(libBLAS, :dsyrk_),Void,
	(Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Float64},Ptr{Float64},Ptr{Int32},
		Ptr{Float64},Ptr{Float64},Ptr{Int32}),
	&c_uplo,&c_transn,&bd_sz,&tile_sz,&c_alpham,A_para[num_tile,k],&bd_sz,&c_alphap,A_para[num_tile,num_tile],&bd_sz)
end

# last diagonal block
ccall(dlsym(libBLAS, :dpotrf_),Void,
	(Ptr{Uint8}, Ptr{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
	&c_uplo, &bd_sz, A_para[num_tile,num_tile], &bd_sz, &c_info)
A_para[num_tile,num_tile][bd_upper]=0;

A_out=zeros(size(A))
for i=1:num_tile-1
    for j=1:i#num_tile
        A_out[(i-1)*tile_sz+(1:tile_sz),(j-1)*tile_sz+(1:tile_sz)]=reshape(A_para[i,j],tile_sz,tile_sz);
    end
end
if bd_sz!=tile_sz
    for j=1:num_tile-1
        A_out[(mat_sz-bd_sz+1):mat_sz,(j-1)*tile_sz+(1:tile_sz)]=reshape(A_para[num_tile,j],bd_sz,tile_sz);
    end
    A_out[(mat_sz-bd_sz+1):mat_sz,(mat_sz-bd_sz+1):mat_sz]=reshape(A_para[num_tile,num_tile],bd_sz,bd_sz);
else
    for j=1:num_tile
        A_out[(num_tile-1)*tile_sz+(1:tile_sz),(j-1)*tile_sz+(1:tile_sz)]=reshape(A_para[num_tile,j],tile_sz,tile_sz);
    end
end

A_out

end

#http://www.netlib.org/clapack/clapack-3.2.1-CMAKE/SRC/VARIANTS/cholesky/TOP/dpotrf.c
#http://www.netlib.org/clapack/cblas/dtrsm.c
#http://www.netlib.org/clapack/cblas/dsyrk.c
#http://www.netlib.org/clapack/cblas/dgemm.c
