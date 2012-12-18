function qr_blas(A,tile_sz)

# 1. Set up Data Structure
tile_sz2=tile_sz^2;
mat_sz=size(A,1);
bd_sz=tile_sz;
num_tile=int(ceil(mat_sz/tile_sz));
lower= repmat(linspace(1,tile_sz,tile_sz)',tile_sz,1).<repmat(linspace(1,tile_sz,tile_sz),1,tile_sz);
tile_lower=find(lower);
c_nb=tile_sz;
tile_sz3=c_nb*tile_sz;

A_para=cell(num_tile,num_tile);
Q_para=cell(num_tile);
# auxilary block
T_para=cell(num_tile,num_tile);

for i=1:num_tile
    for j=1:i
        A_para[i,j]=reshape(A[(i-1)*tile_sz+(1:tile_sz),(j-1)*tile_sz+(1:tile_sz)],1,tile_sz2);
        T_para[i,j]=zeros(1,tile_sz3);
    end
    for j=i+1:num_tile
        A_para[i,j]=reshape(A[(i-1)*tile_sz+(1:tile_sz),(j-1)*tile_sz+(1:tile_sz)],1,tile_sz2);
    end
end

c_sidel='L';
c_transt='T';
c_transn='N';
c_diag='N';
c_alphap=1.0;
c_alpham=-1.0;
c_info=0;
c_work=zeros(c_nb,tile_sz);
c_L=0;
c_H1=1;
c_H2=2;


# 2. Computation

# 2.1 DGEQRT
for k=1:num_tile

    (A_para[k,k],T_para[k,k],c_work) = dgeqrt(tile_sz,tile_sz,tile_sz,A_para[k,k],tile_sz,
	T_para[k,k],tile_sz,c_work,c_info)

#    ccall(dlsym(libBLAS, :dgeqrt_),Void,(Ptr{Int32}, Ptr{Int32},Ptr{Int32}, Ptr{Float64}, Ptr{Int32},Ptr{Float64}, Ptr{Int32},Ptr{Float64}, Ptr{Int32}),&tile_sz, &tile_sz, &tile_sz,A_para[k,k], &tile_sz,T_para[k,k],&tile_sz, c_work, &c_info)

    # 2.2 DTSQRT
    for m=k+1:num_tile

	(A_para[k,k],A_para[m,k],T_para[m,k],c_work) = dtpqrt(tile_sz,tile_sz,c_L,c_nb,
	    A_para[k,k],tile_sz,A_para[m,k],tile_sz,T_para[m,k],c_nb,c_work,c_info)

#       ccall(dlsym(libBLAS, :dtpqrt_),Void,(Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Float64},Ptr{Int32}, Ptr{Float64}, Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Int32}),&tile_sz, &tile_sz,&c_L,&c_nb,A_para[k,k], &tile_sz,A_para[m,k],&tile_sz,T_para[m,k],&c_nb,c_work, &c_info)
    end

    # 2.3 DORMQR
    for n=k+1:num_tile

	(A_para[k,n],) = dormqr(c_sidel,c_transn,tile_sz,tile_sz,tile_sz,A_para[k,k],tile_sz,
	    T_para[k,k],A_para[k,n],tile_sz,c_work,tile_sz,c_info)

#        ccall(dlsym(libBLAS, :dormqr_),Void,(Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Int32}),&c_sidel,&c_transn, &tile_sz,&tile_sz,&tile_sz,A_para[k,k],&tile_sz,T_para[k,k],A_para[k,n], &tile_sz,c_work,&tile_sz,&c_info)

        # 2.4 DSSMQR
        for m=k+1:num_tile

	    (A_para[k,n],A_para[m,n],) = dtpmqrt(c_sidel,c_transt,tile_sz,tile_sz,tile_sz,c_L,
		c_nb,A_para[m,k],tile_sz,T_para[m,k],tile_sz,A_para[k,n],tile_sz,A_para[m,n],
		tile_sz,c_work,c_info)
		
#        ccall(dlsym(libBLAS, :dtpmqrt_),Void,(Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Int32},Ptr{Float64},Ptr{Int32}),&c_sidel,&c_transt, &tile_sz,&tile_sz,&tile_sz,&c_L,&c_nb,A_para[m,k],&tile_sz,T_para[m,k],&tile_sz,A_para[k,n], &tile_sz,A_para[m,n],&tile_sz,c_work,&c_info)
        end
    end
    A_para[k,k][tile_lower]=0;
end



# two ways to recover Q:
# 1. step by step: Q=H1 H2...Hn (dorgqr)
# 2. Fill in matrix I-VTV' (may only be possible for panel but not tile )

# R_out is already overwritten in A_para
R_out=zeros(size(A))
for i=1:num_tile
    for j=1:i
        R_out[(j-1)*tile_sz+(1:tile_sz),(i-1)*tile_sz+(1:tile_sz)]=reshape(A_para[j,i],tile_sz,tile_sz);
    end	
end

# instead of Q=H1....Hn, we directly do Q=AR^{-1} by solving triangular linear system
# it can be naively paralleled in column panel
Q_out=(A/R_out);

# for i=1:mat_sz
#    A_col=A[:,i];
#    R_col[i]=R_out[:,i];
#        ccall(dlsym(libBLAS, :dtrsm_),Void,(Ptr{Uint8},Ptr{Uint8},Ptr{Uint8},Ptr{Uint8},Ptr{Int32},Ptr{Int32},Ptr{Float64}, Ptr{Float64}, Ptr{Int32},Ptr{Float64},Ptr{Int32}),&c_side, &c_uplo,&c_transt,&c_diag, &tile_sz,&tile_sz,&c_alphap,A_col,&tile_sz,R_col[i], &tile_sz)
# end

(Q_out, R_out)

end

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
