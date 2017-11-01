/*************************************************************************/
/*                                                                       */
/*                     Carnegie Mellon University                        */
/*                         Copyright (c) 2009                            */
/*                        All Rights Reserved.                           */
/*                                                                       */
/*  Permission is hereby granted, free of charge, to use and distribute  */
/*  this software and its documentation without restriction, including   */
/*  without limitation the rights to use, copy, modify, merge, publish,  */
/*  distribute, sublicense, and/or sell copies of this work, and to      */
/*  permit persons to whom this work is furnished to do so, subject to   */
/*  the following conditions:                                            */
/*   1. The code must retain the above copyright notice, this list of    */
/*      conditions and the following disclaimer.                         */
/*   2. Any modifications must be clearly marked as such.                */
/*   3. Original authors' names are not deleted.                         */
/*   4. The authors' names are not used to endorse or promote products   */
/*      derived from this software without specific prior written        */
/*      permission.                                                      */
/*                                                                       */
/*  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    */
/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   */
/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          */
/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       */
/*  THIS SOFTWARE.                                                       */
/*                                                                       */
/*************************************************************************/
/*                                                                       */
/*            Authors: Gopala Krishna Anumanchipalli                     */
/*            Email:   gopalakr@cs.cmu.edu                               */
/*                                                                       */
/*************************************************************************/
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_CB 100000
#define MAX_CLUSTER 10000
#define STOP_VAL 10
#define PI 3.14159
#define INF 100000000
#define INIT_OCCUPANCY 50
#define TINY 1.0e-20

using namespace std;

struct Gaussian {
double *mn;
double *vr; // Diagonal for now
};

typedef struct Gaussian Gaussian;

struct Cluster {
int indx[MAX_CB];
int no_g;
int no_d;
Gaussian* gnormal;
};

typedef struct Cluster Cluster;

struct Tree{
        bool isleaf;
        //double gcomp[MAX_CB];
        Cluster gcomp;
        struct Tree* ltree;
        struct Tree* rtree;
        struct Tree* parent;
};

typedef struct Tree Tree;

void init_tree (Tree*& tr);
void make_basetree (Tree* , Gaussian *,int,int, int);
double KL_divergence (Gaussian g1, Gaussian g2, int n);
double SymKL_divergence (Gaussian g1, Gaussian g2, int n);
void init_cluster(Cluster*& comp,int d);
void init_gaussian (Gaussian* gau,int d);
void make_cluster(Cluster*&, Gaussian*,double*, int, int,double,bool*&);
void compute_globalg (Gaussian* cbook,Cluster*& comp,int n);
double kthsmallest (double* arr,int n,int k);
double** diag (double**& d, double* array,int n);
double** subtract(double**& diff, double** mat1,double** mat2, int r, int c);
double** inverse_diag (double**& inverse, double** matrix, int n);
double** transpose (double**& transpose, double** matrix, int nr, int nc);
double** multiply (double**& product, double** mat1, double** mat2, int nr1, int nc1, int nc2);
double det_diag (double** matrix, int n);
double trace (double** matrix, int n);
double CalcDeterminant( double **mat, int order);
int GetMinor(double **src, double **dest, int row, int col, int order);
void MatrixInversion(double **A, int order, double **Y);
void alloc2f(double**& arr, int r, int c);
void free2f(double**& arr, int r); 
void free_gaussian (Gaussian* comp);
double compute_log_prob(double *obs, double *mean, double *var, int Ndim);
int lubksb(double **a, int n, int *indx, double b[], double x[]);
int ludcmp(double **a,int n, int *indx,double *d);
int inverse_sphinx (double **ainv, double**a, int len);

int main (int argc,char *argv[])
{
        int i,j,k,l,m,n;
	int no_cbook, no_comps, no_dimc,no_feat,no_dimf,*truth,*counts;
	double dummy;
	ifstream fp_cb;
	ifstream fp_ft;
	ifstream fp_tt;
	ifstream fp_gt;
	Gaussian *cbook;
	Gaussian *comps;
	double **feat;

// Load codebooks and observations
	fp_cb.open(argv[1],ios::in);
	fp_ft.open(argv[2],ios::in);
	fp_tt.open(argv[3],ios::in);
	fp_gt.open(argv[4],ios::in);
	fp_cb >> no_cbook >> no_dimc;
	cbook=(Gaussian *)malloc (sizeof(Gaussian) * no_cbook);
	for (i=0;i<no_cbook;i++)
	{
		init_gaussian(&(cbook[i]),no_dimc);
		for (j=0;j<no_dimc/2;j++)
		{fp_cb >> cbook[i].mn[j] >> cbook[i].vr[j];
		cbook[i].vr[j]*=cbook[i].vr[j];} // NOTE: assuming std.dev
	}

	fp_gt >> no_comps >> dummy;
	comps= (Gaussian *)malloc (sizeof(Gaussian) * no_comps);
	for (i=0;i<no_comps;i++)
	{
		init_gaussian(&(comps[i]),no_dimc);
		for (j=0;j<no_dimc/2;j++)
		{fp_gt >> comps[i].mn[j] >> comps[i].vr[j];
		comps[i].vr[j]*=comps[i].vr[j];} // NOTE: assuming std.dev
	}
	fp_ft >> no_feat >> no_dimf;
	alloc2f(feat,no_feat,no_dimf);
	truth=(int*)malloc(no_feat*sizeof(int));
	for (i=0;i<no_feat;i++)
	{
		for (j=0;j<no_dimf;j++)
		{fp_ft >> feat[i][j];}
		fp_tt >> truth[i];
	}
/*	double **a;
	alloc2f (a, 3, 3);
	a[0][0]=1;
	a[0][1]=2;
	a[0][2]=3;
	a[1][0]=4;
	a[1][1]=5;
	a[1][2]=6;
	a[2][0]=7;
	a[2][1]=8;
	a[2][2]=8;

	double **i2;
	alloc2f (i2,3,3);
//	MatrixInversion(a,3,i2);
	inverse_sphinx(i2,a,3);
	for (i=0;i<3;i++){
 	 for (j=0;j<3;j++){cout << i2[i][j]<< " ";}cout << endl;}
	exit(0);*/

// Output the Log probabilities
//
	double **exmn,**exft, **p1,
		**p2,**p3,**t1,**i1, **Z, 
		**v, **D, ***G, **W, **Wv;

	alloc2f(exmn,no_dimf+1,1);
	alloc2f(exft,no_dimf,1);
	alloc2f(Z,no_dimf,no_dimf+1);
	alloc2f(W,no_dimf,no_dimf+1);
	alloc2f(Wv,no_dimf,no_dimf);
	counts=(int*)calloc(no_cbook,sizeof(int));
	G=(double***)malloc(no_dimf*sizeof(double**));
	for(l=0;l<no_dimf;l++){
		 alloc2f(G[l],no_dimf+1,no_dimf+1);}
	
	alloc2f(v,no_dimf,no_dimf);
	
	for (j=0;j<no_feat;j++)
	{
		i=truth[j];
		counts[i]++;
	}
	for (j=0;j<no_cbook;j++)
	{
		if(counts[j]>0){
		exmn[0][0]=1;
		for(k=0;k<no_dimf;k++){
			exmn[k+1][0]=cbook[j].mn[k];
		}
		
		transpose(t1,exmn,no_dimf+1,1);
		diag(p2,cbook[j].vr,no_dimf);
		alloc2f(i1,no_dimf,no_dimf);
		inverse_sphinx(i1,p2,no_dimf);
		for (m=0;m<no_dimf;m++)
		 for (n=0;n<no_dimf;n++)
		 {
			//v[m][n]= i1[m][n];
			v[m][n]= i1[m][n] * (double)counts[j];
		 }

		/*for(l=0;l<no_dimf;l++){
		 for(m=0;m<no_dimf;m++){cout << v[l][m] << " ";}cout << endl;}
		exit(0);	*/
		
		multiply(D,exmn,t1,no_dimf+1,1,no_dimf+1);// p1 == D, outer product of extended means
		for(l=0;l<no_dimf;l++){
		 for(m=0;m<no_dimf+1;m++)
		  for(n=0;n<no_dimf+1;n++)
			G[l][m][n] +=D[m][n] * v[l][l];
		}
		free2f(t1,1);
		free2f(p2,no_dimf);
		free2f(i1,no_dimf);
		free2f(D,no_dimf+1);
	}}
	for (j=0;j<no_feat;j++)
	{

		i=truth[j];
		exmn[0][0]=1;
		for(k=0;k<no_dimf;k++){
			exmn[k+1][0]=cbook[i].mn[k];
			exft[k][0]=feat[j][k];
		}
		/*transpose(t1,exmn,no_dimf+1,1);
		multiply(p1,exft,t1,no_dimf,1,no_dimf+1); // ft*mn
		diag(p2,cbook[i].vr,no_dimf);
		alloc2f(i1,no_dimf,no_dimf);
		inverse_sphinx(i1,p2,no_dimf);
		multiply(p3,i1,p1,no_dimf,no_dimf,no_dimf+1); // (var)^-1 * p1*/

		diag(p2,cbook[i].vr,no_dimf);
		alloc2f(i1,no_dimf,no_dimf);
		inverse_sphinx(i1,p2,no_dimf);
		multiply(p1,i1,exft,no_dimf,no_dimf,1);
		transpose(t1,exmn,no_dimf+1,1);
		multiply(p3,p1,t1,no_dimf,1,no_dimf+1);
		

		for(m=0;m<no_dimf;m++){
		 for(n=0;n<no_dimf+1;n++)
		 {
			Z[m][n] += p3[m][n];
		 }
		}


		free2f(p1,no_dimf);
		free2f(p2,no_dimf);
		free2f(p3,no_dimf);
		free2f(i1,no_dimf);
		free2f(t1,1);

	}
//	cout << "Build started" << endl;
	alloc2f(i1,no_dimf+1,no_dimf+1);
	for(l=0;l<no_dimf;l++)
	{
		inverse_sphinx(i1,G[l],no_dimf+1);
		for(m=0;m<no_dimf+1;m++){exmn[m][0]=Z[l][m];}
		multiply(p1,i1,exmn,no_dimf+1,no_dimf+1,1);
		for (i=0;i< no_dimf+1;i++){
	 	 W[l][i]=p1[i][0];if (i!= 0){Wv[l][i-1]=p1[i][0];}}//cout << p1[i][0]<< " ";}//cout << endl;
		free2f(p1,no_dimf+1);
	}
//--------ORIG
/*	for(l=0;l<no_dimf;l++){
	for(i=0;i<no_dimf+1;i++){
	cout << W[l][i] << " ";}cout << endl;}
	exit (0);*/
	/*alloc2f(p1,no_dimf+1,no_dimf);
	for(i=0;i<no_dimf+1;i++)
	 for(j=0;j<no_dimf;j++){
	  for(k=0;(k<no_dimf+1) && (k!=j);k++)
	  {
		sum+=w[i][j]
	  }
	p1[i][j]=Z[];}*/
	for(i=0;i<no_comps;i++){exmn[0][0]=1;
	 for(j=0;j<no_dimf;j++){exmn[j+1][0]=comps[i].mn[j];}
	multiply(p1,W,exmn,no_dimf,no_dimf+1,1);
	diag(p2,comps[i].vr,no_dimf);
	multiply(p3,Wv,p2,no_dimf,no_dimf,no_dimf);
	transpose(t1,Wv,no_dimf,no_dimf);
	free2f(p2,no_dimf);
	multiply(p2,p3,t1,no_dimf,no_dimf,no_dimf);
	//for(j=0;j<no_dimf;j++){cout<< p1[j][0] << " " << sqrt(comps[i].vr[j]) << " ";}
	for(j=0;j<no_dimf;j++){cout<< p1[j][0] << " " << sqrt(p2[j][j]) << " ";}
	// for(j=0;j<no_dimf;j++){cout<< comps[i].mn[j] << " " << sqrt(comps[i].vr[j]) << " ";}
	cout << endl;
	free2f(p1,no_dimf);
	free2f(p2,no_dimf);
	free2f(p3,no_dimf);
	free2f(t1,no_dimf);
	}

// Free Memory etc.,
	fp_cb.close();
	fp_ft.close();
	fp_tt.close();
	fp_gt.close();
	free2f(feat,no_feat);
	free2f(exmn,no_dimf+1);
	free2f(exft,no_dimf);
	for(i=0;i<no_dimf;i++)
	{
		free2f(G[i],no_dimf+1);
	}
	free(G);
	free2f(W,no_dimf);
	free2f(Wv,no_dimf);
	free2f(v,no_dimf);
	free2f(Z,no_dimf);
	for(i=0;i<no_comps;i++)
	{
		free_gaussian(&(comps[i]));
	}
	free(comps);
	for(i=0;i<no_cbook;i++)
	{
		free_gaussian(&(cbook[i]));
	}
	free(cbook);
}

int lubksb(double **a, int n, int *indx, double b[], double x[])
{
    int i,ii=0,ip,j,done=0;
    double sum;

    for (i = 0; i < n; i++) {
        ip = indx[i];
        sum = b[ip];
        b[ip] = b[i];
        if (done) {
            for (j = ii; j < i; j++)
                sum -= a[i][j] * b[j];
        }
        else if (sum) {
            ii=i;
            done=1;
        }
        b[i]=sum;
    }
    for (i=0;i<n;i++) {
        x[i] = b[i];
    }
   for (i = n-1; i >= 0; i--) {
        sum = x[i];

        for (j = n-1; j > i; j--)
            sum -= a[i][j]*x[j];

        x[i]=sum/a[i][i];
    }
   return 1;
}

int ludcmp(double **a,int n, int *indx,double *d)
{
    int i,imax=0,j,k;
    double big,dum,sum,t1;
    double  *vv;  /* vv stores the implicit scaling of each row */

    vv=(double*)calloc(n,sizeof(double));

    *d=1.0;
    for (i = 0; i < n; i++) { /*Loop over rows to get implicit scaling */
        big = 0.0;    /*information */
        for (j = 0; j < n; j++) {
            if((t1 = fabs(a[i][j])) > big)
                big=t1;
        }
        if (big == 0.0) {
            return 0;
        }
        vv[i] = 1.0/big; /* Save the scaling */
    }
    for (j = 0; j < n; j++) {
        for (i = 0; i < j; i++) {
            sum = a[i][j];
            for (k = 0; k < i; k++)
                sum -= a[i][k]*a[k][j];
            a[i][j] = sum;
        }
        big = 0.0;
        for (i = j; i < n;i++) {
            sum = a[i][j];
            for (k = 0; k < j; k++)
                sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
            if ( (dum = vv[i] * fabs(sum)) >= big) {
                big = dum;
                imax = i;
            }
        }
       if (j != imax) {
            for (k = 0; k < n; k++) {
                dum = a[imax][k];
                a[imax][k] = a[j][k];
                a[j][k] = dum;
            }
            *d = -(*d);
            vv[imax] = vv[j];
        }
        indx[j] = imax;
        if (a[j][j] == 0.0)
            a[j][j]=TINY;

        if (j != n-1) {
            dum = 1.0 / a[j][j];
            for (i = j+1; i < n; i++)
                a[i][j] *= dum;
        }
    }
    free(vv);
    return 1;
}


int inverse_sphinx (double **ainv, double**a, int len)
{
    int i, j;
    int *indx;
    double d;
    double *col;
    double **adcmp;

    indx=(int*)calloc(len, sizeof(int));
    col =(double*)calloc(len, sizeof(double));
    alloc2f(adcmp,len,len);

    for (i = 0; i < len; i++) {
        for (j = 0; j < len; j++) {
            adcmp[i][j] = a[i][j];
        }
    }

    ludcmp(adcmp, len, indx, &d);

    for (j = 0; j < len; j++) {
        for (i = 0; i < len; i++)
            col[i] = 0;
        col[j] = 1;
        lubksb(adcmp, len, indx, col, col);
        for (i = 0; i < len; i++) {
            ainv[i][j] = col[i];
        }
    }

    free(indx);
    free(col);
    free2f(adcmp,len);
    return 1;
}


double CalcDeterminant( double **mat, int order)
{
        int i;
        if( order == 1 )
                return mat[0][0];

        float det = 0;
	double** minor;
	alloc2f(minor,order-1,order-1);
        for(i = 0; i < order; i++ )
        {

                GetMinor( mat, minor, 0, i , order);
                det += pow( -1.0, i ) * mat[0][i] * CalcDeterminant( minor,order-1 );
        }
        free2f(minor,order-1);
        return det;
}

int GetMinor(double **src, double **dest, int row, int col, int order)
{
        int colCount=0,rowCount=0,i,j;
        for(i = 0; i < order; i++ )
        {
          if( i != row )
          {
             colCount = 0;
             for(j = 0; j < order; j++ )
             {
                 if( j != col )
                 {
                    dest[rowCount][colCount] = src[i][j];
                    colCount++;
                 }
             }
             rowCount++;
          }
        }
  	return 1;
}

void MatrixInversion(double **A, int order, double **Y)
{
        int i,j;
	for (i=0;i<order;i++){
 	 for (j=0;j<order;j++){cout << A[i][j]<< " ";}cout << endl;}
        double det = 1.0/CalcDeterminant(A,order);
	double **minor;
        alloc2f(minor,order-1,order-1);

        for(j=0;j<order;j++)
        {
                for(i=0;i<order;i++)
                {
                        GetMinor(A,minor,j,i,order);
                        Y[i][j] = det*CalcDeterminant(minor,order-1);
                        if( (i+j)%2 == 1)
                                Y[i][j] = -Y[i][j];
                }
        }
        free2f(minor,order-1);
}


double compute_log_prob(double *obs, double *mean, double *var, int Ndim)
{
	int i, j;
	double tmp,max,score;
	double loginvsqrtdet=0,*halfinvvar;
	halfinvvar=(double*)malloc(sizeof(double)*Ndim);
	// Assuming variances are preinverted and scaled by 2.
	for (i=0;i<Ndim;i++)
	{
		tmp=var[i];
		halfinvvar[i]=0.5/tmp;
		loginvsqrtdet-=0.5*log(tmp);
	}
	loginvsqrtdet-=(0.5*double(Ndim)) * log(2*PI);
	// Find max
	max = -1e+30;
	score = loginvsqrtdet;
	for (j=0;j<Ndim;j++){
		tmp = obs[j] - mean[j];
	//	cout << tmp << " ";
		score -= tmp*tmp*halfinvvar[j];
	}
	//cout << endl;
	free(halfinvvar);
	return(exp(score));
	//return(score);
}
	
void init_gaussian (Gaussian* gau,int d)
{
	gau->mn=(double*)calloc(d,sizeof(double));
	gau->vr=(double*)calloc(d,sizeof(double));
}

void free_gaussian (Gaussian* comp)
{
	free(comp->mn);
	free(comp->vr);
//	free(comp);
}
void init_cluster(Cluster*& comp,int d)
{
	comp->gnormal=(Gaussian*)malloc(sizeof(Gaussian));
	init_gaussian(comp->gnormal,d);
}

void free_cluster(Cluster* comp)
{
	free_gaussian(comp->gnormal);
	free(comp);
}

void init_tree(Tree*& tr)
{
	tr=(Tree*)malloc(sizeof(Tree));
        tr->isleaf=true;
        tr->parent=NULL;
        tr->ltree=NULL;
        tr->rtree=NULL;
}
void free_tree(Tree*& tr)
{
        if (tr->isleaf) free_cluster(&(tr->gcomp));
	else {free_tree(tr->ltree);free_tree(tr->rtree);}
        free(tr);
}

void make_basetree (Tree* tr, Gaussian* cbook, int n, int d, int k)
{
	//
	bool *unaligned;
	unaligned=(bool*)malloc(n*sizeof(bool));
	int i,j,clcnt=0;
	for (i=0;i<n;i++)unaligned[i]=true;
	double **d_kl,kth;
	Cluster** cls;
	cls=(Cluster**)malloc(MAX_CLUSTER*sizeof(Cluster*));
	alloc2f(d_kl,n,n);
	
	for (i=0;i < n; i++){
	if(unaligned[i]){
	for (j=i+1;j < n; j++){
	if(unaligned[j])
	{
		//d_kl[i][j]=KL_divergence(cbook[j],cbook[i],d);
		//using symmetric KL divergence
		d_kl[i][j]=SymKL_divergence(cbook[j],cbook[i],d);
		d_kl[j][i]=d_kl[i][j];
	//	cout<< i <<"-" << j << " " << d_kl[i][j]<< endl;
	}else{d_kl[i][j]=INF;}}
	// Pick the k_th smallest number
	kth = kthsmallest (d_kl[i],n,k);
	//cout << "Kth: "<<kth << endl;
	int pk=k;
//	cout << "KTH:" << kth  << " " << k<< endl;
	while (kth==INF && pk > 0){kth=kthsmallest(d_kl[i],n,pk--);}
	// make a new cluster of unaligned elements within radius kth
	cls[clcnt]=(Cluster*)malloc(sizeof(Cluster));
	init_cluster(cls[clcnt],d);
//	cout << "KTH:" << kth << endl;
	make_cluster(cls[clcnt++], cbook, d_kl[i], n, d, kth, unaligned);
	/*for (j=0;j<n;j++)
	if (d_kl[i][j] < kth)
	{ 	unaligned[j]=false;
		unaligned[i]=false;
		d_kl[i][j]=INF; 
		d_kl[j][i]=INF;
		//cout <<  j<<" ";
	}*/
			
	cout << i << " (" << cls[clcnt-1]->no_g << ") : ";
	for(j=0;j<cls[clcnt-1]->no_g;j++){
		cout << cls[clcnt-1]->indx[j] << " ";
	}
	cout << endl;
	}}
	for(i=0;i<clcnt;i++){
	cout << i << " (" << cls[i]->no_g << ") : ";
	for(j=0;j<cls[i]->no_g;j++){
		cout << cls[i]->indx[j] << " ";
	}
	cout << endl;
	}
	free2f(d_kl,n);
	//split_flag=split(tr,d_kl[i])
	//if (split_flag){make_tree(tr->ltree);make_tree(tr->rtree);}
}

void make_cluster(Cluster*& comp,Gaussian* cbook,double* d_kl,int n, int d, double kth, bool*& unaligned)
{
	int i,cnt=0;
	comp->no_d=d;
	for(i=0;i<n;i++)
	{	
		//cout << d_kl[i] << " "<< kth <<endl;
		if (d_kl[i] <= kth )
		if (unaligned[i]==true)
		{
			comp->indx[cnt++]=i;
			//unaligned[i]=false;
			//cout << "selected " << i << endl;
		}
	}
	comp->no_g=cnt;
	compute_globalg (cbook,comp,cnt);
}
	
void compute_globalg (Gaussian* cbook,Cluster*& comp,int n)
{
	int i,j,idx;
	init_gaussian(comp->gnormal,comp->no_d);
	for (i=0;i<n;i++){
		idx=comp->indx[i];
		for (j=0;j< comp->no_d;j++){
		comp->gnormal->mn[j]+=cbook[idx].mn[j];
		comp->gnormal->vr[j]+=cbook[idx].vr[j];
		}
	}

}

double kthsmallest (double* arr,int n,int k)
{
	//random pivot selection, can be improved by generalized sampling
	int i,pivot= rand()%n,j=1,l;
	double *temparr;
        double smallest;
	temparr=(double*)calloc(n,sizeof(double));
	temparr[0]=arr[pivot];
	int rank=0;
	//shatter the elements
	for (i=0;i<n ;i++)
	if (i!= pivot){
	if (arr[i] > arr[pivot] )
	{
		temparr[j++]=arr[i];
	} else{
		for (l=j;l>rank;l--){temparr[l] =temparr[l-1];}
		temparr[rank++]=arr[i];
		j++;
	}}
	//Checking for the conditions
	if(rank == k-1) smallest = temparr[rank];
	if(rank > k-1) smallest = kthsmallest(temparr,rank,k);
	if(rank < k-1) smallest = kthsmallest(temparr+rank,n-rank,k-rank);
	free (temparr); 
        return smallest;
}

double SymKL_divergence (Gaussian g1, Gaussian g2, int n)
{
// D_kl(g1||g2) = 0.5 * (log (det(v2)/det(v1)) + trace (inv(v2)*v1) + transpose(mu2 - mu1) * inv(var2)*(mu2 - mu1) - N )
	int i;
	double **m1,**m2,**v1, **v2, 
		**p1,**p2,**p3,**s1,
		**i2,**t1,**t2,**q1,
		**q2,**q3,**j2,**s2;
	double d1,d2,d_kl;
	diag(v1,g1.vr,n);
	d1=det_diag(v1,n);
	diag(v2,g2.vr,n);
	d2=det_diag(v2,n);
	p1=multiply(p1,inverse_diag(i2,v2,n),v1,n,n,n);
	q1=multiply(q1,inverse_diag(j2,v1,n),v2,n,n,n);
	alloc2f(m1,n,1);
	for(i=0;i<n;i++)m1[i][0]=g1.mn[i];
	alloc2f(m2,n,1);
	for(i=0;i<n;i++)m2[i][0]=g2.mn[i];
	s1=subtract(s1,m2,m1,n,1);
	s2=subtract(s2,m1,m2,n,1);
	p2=multiply(p2,transpose(t1,s1,n,1),i2,1,n,n);
	q2=multiply(q2,transpose(t2,s2,n,1),j2,1,n,n);
	p3=multiply(p3,p2,s1,1,n,1);
	q3=multiply(q3,q2,s2,1,n,1);
	d_kl= 0.5 * (log(d2/d1) + trace (p1,n) + p3[0][0] -n )/log(2);
	d_kl+= 0.5 * (log(d1/d2) + trace (q1,n) + q3[0][0] -n )/log(2);
	free2f(m1,n);
	free2f(m2,n);
	free2f(v1,n);
	free2f(v2,n);
	free2f(t1,1);
	free2f(i2,n);
	free2f(p1,n);
	free2f(p2,1);
	free2f(p3,1);
	free2f(s1,n);
	free2f(t2,1);
	free2f(j2,n);
	free2f(q1,n);
	free2f(q2,1);
	free2f(q3,1);
	free2f(s2,n);
	return d_kl;
}

double KL_divergence (Gaussian g1, Gaussian g2, int n)
{
// D_kl(g1||g2) = 0.5 * (log (det(v2)/det(v1)) + trace (inv(v2)*v1) + transpose(mu2 - mu1) * inv(var2)*(mu2 - mu1) - N )
	int i;
	double **m1,**m2,**v1, **v2, **p1,**p2,**p3,**s1,**i2,**t1;
	double d1,d2,d_kl;
	diag(v1,g1.vr,n);
	d1=det_diag(v1,n);
	diag(v2,g2.vr,n);
	d2=det_diag(v2,n);
	p1=multiply(p1,inverse_diag(i2,v2,n),v1,n,n,n);
	alloc2f(m1,n,1);
	for(i=0;i<n;i++)m1[i][0]=g1.mn[i];
	alloc2f(m2,n,1);
	for(i=0;i<n;i++)m2[i][0]=g2.mn[i];
	s1=subtract(s1,m2,m1,n,1);
	p2=multiply(p2,transpose(t1,s1,n,1),i2,1,n,n);
	p3=multiply(p3,p2,s1,1,n,1);
	d_kl= 0.5 * (log(d2/d1) + trace (p1,n) + p3[0][0] -n )/log(2);
	free2f(m1,n);
	free2f(m2,n);
	free2f(v1,n);
	free2f(v2,n);
	free2f(t1,1);
	free2f(i2,n);
	free2f(p1,n);
	free2f(p2,1);
	free2f(p3,1);
	free2f(s1,n);
	return d_kl;
}

double** subtract(double**& diff, double** mat1,double** mat2,int r, int c)
{
	int i,j;
	alloc2f (diff,r,c);
	for (i=0;i<r;i++)
	 for (j=0;j<c;j++) diff[i][j]=mat1[i][j] - mat2[i][j];
	return diff;
}

double** inverse_diag (double**& inverse, double** matrix, int n)
{
	int i;
	alloc2f(inverse,n,n);
	for (i=0;i<n;i++)inverse[i][i]=1/matrix[i][i];
	return inverse;
}

double** transpose (double**& transpose, double** matrix, int nr, int nc)
{
	int i,j;
	alloc2f(transpose,nc,nr);
	for (i=0;i<nr;i++)
	 for (j=0;j<nc;j++)transpose[j][i]=matrix[i][j];
	return transpose;
}

double** multiply (double**& product, double** mat1, double** mat2, int nr1, int nc1, int nc2)
{
	int i,j,k;
	alloc2f(product,nr1,nc2);
	
	/*for (i=0;i<nr1;i++)
	{ for (j=0;j<nc1;j++)
	 {cout << mat1[i][j] << " ";}
	cout  << endl;}*/
	for (i=0;i<nr1;i++)
	 for (j=0;j<nc2;j++)
	  for (k=0;k<nc1;k++) product[i][j]+=mat1[i][k] * mat2[k][j];
	
	return product;
}

double det_diag (double** matrix, int n)
{
	int i;
	double det=1;
	for (i=0;i<n;i++) det*=matrix[i][i];
	return det;
}

double trace (double** matrix, int n)
{
	int i;
	double trace=0;
	for (i=0;i<n;i++)trace+=matrix[i][i];
	return trace;
}

double** diag (double**& d, double* array,int n)
{
	int i;
	alloc2f(d,n,n);
	for (i=0;i<n;i++)d[i][i]=array[i];
	return d;
}

void alloc2f(double**& arr, int r, int c) {
   arr = new double*[r];
   for (int i = 0; i < r; i++) {
       arr[i] = new double[c];
       for (int j = 0; j < c; j++) {
          arr[i][j] = 0;
       }
   }
}

void free2f(double**& arr, int r) {
   for (int i = 0; i < r; i++) {
       delete [] arr[i];
   }
   delete [] arr;
}

