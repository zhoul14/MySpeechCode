/*#include "stdafx.h"*/
#include <cmath>

void MatrixMul(double *a,double *b,double *c,long DIM)
{
long	i,j,k,ii,jj;
double	Sum;

	for(i=0;i<DIM;i++)
	for(j=0;j<DIM;j++)
	{
		Sum = 0;
		ii=i*DIM;
		jj=j;
		for(k=0;k<DIM;k++)
		{
			Sum += a[ii]*b[jj];
			ii++;
			jj+=DIM;
		}
		c[i*DIM+j]=Sum;
	}
}

//------------------------------------------------------------------------------
// MatrixInv 是全选主元、同址矩阵求逆程序
// 输入：
//		a:	 指向矩阵的缓冲区地址，矩阵元素是按行顺序存储的
//		DIM: DIM X DIM 矩阵的维数
// 返回：
//		a:	指向逆矩阵的缓冲区地址
//		ret:函数的返回值是输入矩阵a的行列式值
//	算法：	 Gauss-Jordan矩阵求逆算法。此算法采用初等行变换的方法进行消元。
//			在消元的过程中采用全选主元的方法。全选主元后将交换主元到对角线上
//			可以证明这种交换与在对矩阵求逆之前先行交换后在求逆是等价的。即相当于
//			对(Q*A*R)求逆。这里Q是行变换矩阵，R是列变换矩阵。因此若INV(Q*A*R)=B
//			则INV(A)=R*B*Q，即全选主元时的行交换在恢复时就变成了列交换。
//------------------------------------------------------------------------------

double MatrixInv(double *a,long DIM)
{   
long	*is,*js,i,j,k,l,u,v,x,y;
double	d,fTmp,Det;

    is=new long[DIM];
    js=new long[DIM];

	Det = 1;

    for (k=0; k<DIM; k++)
    {
		//全选主元
		d=0.0;
		for( i=k,x=k*DIM; i<DIM; i++,x+=DIM )
		for( j=k,l=x+k; j<DIM; j++,l++ )
		{
			fTmp=( a[l] > 0 )?  a[l] : -a[l];
			if (fTmp>d)
			{ d=fTmp; is[k]=i; js[k]=j;}
		}

		if ( fabs(d) < 1e-50 )
		{	//主元为0，矩阵不可逆
			delete []is;
			delete []js;
            return(0);
		}
		//将主元交换到对角线上
		if ( is[k] != k )
		{	//交换行列式，行列式值变号
			Det=-Det;
			for ( j=0,u=k*DIM,v=is[k]*DIM ; j<DIM ; j++,u++,v++ )
			{	//第k行和第is[k]行交换，即a[k][j] <=> a[ is[k] ][j]
				fTmp=a[u];
				a[u]=a[v];
				a[v]=fTmp;
			}
		}
		if ( js[k] != k )
		{	//交换行列式，行列式值变号
			Det=-Det;
			for (i=0,u=k,v=js[k]; i<DIM; i++,u+=DIM,v+=DIM)
			{	//第k列和js[k]列交换,即a[i][k] <=> a[i][ js[k] ]
				fTmp=a[u]; 
				a[u]=a[v];
				a[v]=fTmp;
			}
		}

        l=k*DIM+k;
		Det = Det * a[l];		//计算行列式的值
        a[l]=1.0/a[l];
		for( j=0,u=k*DIM; j<DIM; j++,u++ )	//对第k行进行主元归一化
        if( j != k )
	    { 
			a[u] *= a[l]; 
		}
	    //按行对矩阵进行消元
		v=k*DIM;	//计算第k行的地址
		for( i=0,y=0; i<DIM; i++,y+=DIM )
		{
			if ( i != k )
			{	//对第i行消元
				fTmp=a[y+k];			//fTmp=a[i][k];
				a[y+k]=0;
				for( j=0,u=y,x=v; j<DIM; j++,u++,x++ )
				{
					a[u] -= fTmp*a[x];	//a[i][j]-=fTmp*a[k][j]
				}
			}
		}
    }
	//恢复原始的逆矩阵，即: R*B*Q,这里R是列变换矩阵，Q是行变换矩阵
	//矩阵消元时的行交换在恢复时是变成了列交换
	for( k=DIM-1; k>=0; k-- )
    {   
		if ( js[k] != k )
		{
			for( j=0,u=k*DIM,v=js[k]*DIM; j<DIM; j++,u++,v++ )
			{	//第k行和js[k]行交换
				//a[k][j] <=> a[ js[k] ][j];
				fTmp=a[u];	a[u]=a[v];	a[v]=fTmp;
			}
		}

		if ( is[k] != k )
		for( i=0,u=k,v=is[k]; i<DIM; i++,u+=DIM,v+=DIM )
		{	//第k列和is[k]列交换
			//a[i][k] <=> a[i][ is[k] ]
			fTmp=a[u];	a[u]=a[v];	a[v]=fTmp;
		}
    }
	delete []is;
	delete []js;
    return(Det);
}

float MatrixInv(float *a,long DIM)
{   
long	*is,*js,i,j,k,l,u,v,x,y;
float	d,fTmp,Det;

    is=new long[DIM];
    js=new long[DIM];

	Det = 1;

    for (k=0; k<DIM; k++)
    {
		//全选主元
		d=0.0;
		for( i=k,x=k*DIM; i<DIM; i++,x+=DIM )
		for( j=k,l=x+k; j<DIM; j++,l++ )
		{
			fTmp=( a[l] > 0 )?  a[l] : -a[l];
			if (fTmp>d)
			{ d=fTmp; is[k]=i; js[k]=j;}
		}

		if ( d == 0.0 )
		{	//主元为0，矩阵不可逆
			delete []is;
			delete []js;
            return(0);
		}
		//将主元交换到对角线上
		if ( is[k] != k )
		{	//交换行列式，行列式值变号
			Det=-Det;
			for ( j=0,u=k*DIM,v=is[k]*DIM ; j<DIM ; j++,u++,v++ )
			{	//第k行和第is[k]行交换，即a[k][j] <=> a[ is[k] ][j]
				fTmp=a[u];
				a[u]=a[v];
				a[v]=fTmp;
			}
		}
		if ( js[k] != k )
		{	//交换行列式，行列式值变号
			Det=-Det;
			for (i=0,u=k,v=js[k]; i<DIM; i++,u+=DIM,v+=DIM)
			{	//第k列和js[k]列交换,即a[i][k] <=> a[i][ js[k] ]
				fTmp=a[u]; 
				a[u]=a[v];
				a[v]=fTmp;
			}
		}

        l=k*DIM+k;
		Det = Det * a[l];		//计算行列式的值
        a[l]=1.0f/a[l];
		for( j=0,u=k*DIM; j<DIM; j++,u++ )	//对第k行进行主元归一化
        if( j != k )
	    { 
			a[u] *= a[l]; 
		}
	    //按行对矩阵进行消元
		v=k*DIM;	//计算第k行的地址
		for( i=0,y=0; i<DIM; i++,y+=DIM )
		{
			if ( i != k )
			{	//对第i行消元
				fTmp=a[y+k];			//fTmp=a[i][k];
				a[y+k]=0;
				for( j=0,u=y,x=v; j<DIM; j++,u++,x++ )
				{
					a[u] -= fTmp*a[x];	//a[i][j]-=fTmp*a[k][j]
				}
			}
		}
    }
	//恢复原始的逆矩阵，即: R*B*Q,这里R是列变换矩阵，Q是行变换矩阵
	//矩阵消元时的行交换在恢复时是变成了列交换
	for( k=DIM-1; k>=0; k-- )
    {   
		if ( js[k] != k )
		{
			for( j=0,u=k*DIM,v=js[k]*DIM; j<DIM; j++,u++,v++ )
			{	//第k行和js[k]行交换
				//a[k][j] <=> a[ js[k] ][j];
				fTmp=a[u];	a[u]=a[v];	a[v]=fTmp;
			}
		}

		if ( is[k] != k )
		for( i=0,u=k,v=is[k]; i<DIM; i++,u+=DIM,v+=DIM )
		{	//第k列和is[k]列交换
			//a[i][k] <=> a[i][ is[k] ]
			fTmp=a[u];	a[u]=a[v];	a[v]=fTmp;
		}
    }
	delete []is;
	delete []js;
    return(Det);
}
