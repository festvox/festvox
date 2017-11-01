/*********************************************************************/
/*                                                                   */
/*            Nagoya Institute of Technology, Aichi, Japan,          */
/*       Nara Institute of Science and Technology, Nara, Japan       */
/*                                and                                */
/*             Carnegie Mellon University, Pittsburgh, PA            */
/*                      Copyright (c) 2003-2004                      */
/*                        All Rights Reserved.                       */
/*                                                                   */
/*  Permission is hereby granted, free of charge, to use and         */
/*  distribute this software and its documentation without           */
/*  restriction, including without limitation the rights to use,     */
/*  copy, modify, merge, publish, distribute, sublicense, and/or     */
/*  sell copies of this work, and to permit persons to whom this     */
/*  work is furnished to do so, subject to the following conditions: */
/*                                                                   */
/*    1. The code must retain the above copyright notice, this list  */
/*       of conditions and the following disclaimer.                 */
/*    2. Any modifications must be clearly marked as such.           */
/*    3. Original authors' names are not deleted.                    */
/*                                                                   */    
/*  NAGOYA INSTITUTE OF TECHNOLOGY, NARA INSTITUTE OF SCIENCE AND    */
/*  TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, AND THE CONTRIBUTORS TO  */
/*  THIS WORK DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,  */
/*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, */
/*  IN NO EVENT SHALL NAGOYA INSTITUTE OF TECHNOLOGY, NARA           */
/*  INSTITUTE OF SCIENCE AND TECHNOLOGY, CARNEGIE MELLON UNIVERSITY, */
/*  NOR THE CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR      */
/*  CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM   */
/*  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,  */
/*  NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN        */
/*  CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.         */
/*                                                                   */
/*********************************************************************/
/*                                                                   */
/*  VQ with LBG Algorithm                                            */
/*                                 Nov. 26, 1997 by S. Nakamura      */
/*                                                                   */
/*  Modified by Tomoki Toda (tomoki@ics.nitech.ac.jp)                */
/*  June 2004                                                        */
/*  Integrate as a Voice Conversion module                           */
/*                                                                   */
/*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fileio.h"
#include "../include/option.h"
#include "../include/voperate.h"

#include "./vq_sub.h"

int *ialloc(int cc)
{
  int *ptr;

  if ((ptr = (int *)malloc(cc * sizeof(int))) == NULL){
    fprintf(stderr,"Can't malloc\n");return(0);}
  return(ptr);
}

double *dalloc(int cc)
{
  double *ptr;

  if( (ptr=(double *)malloc(cc*sizeof(double))) == NULL){
    fprintf(stderr,"Can't malloc\n");
    exit(1);
  }
  return(ptr);
}

float *falloc(int cc)
{
  float *ptr;

  if( (ptr=(float *)malloc(cc*sizeof(float))) == NULL){
    fprintf(stderr,"Can't malloc\n");
    exit(1);
  }
  return(ptr);
}

void init_vqlabel(struct codebook *bookp, struct sample *smpl)
{
  int i;

  bookp->vlab = ialloc(smpl->nfrms);
  bookp->vemp = ialloc(bookp->vecsize);
  bookp->vpara= dalloc(bookp->vecsize * bookp->vecorder);

  for(i=0;i<smpl->nfrms;i++)
    bookp->vlab[i] = 0;
  for(i=0; i<bookp->vecsize; i++)
    bookp->vemp[i] = 0;

  return;
}

void free_vqlabel(struct codebook *bookp)
{
  free(bookp->vlab);
  free(bookp->vemp);
  free(bookp->vpara);

  return;
}

void lbg(struct analysis *condana, struct codebook *bookp, struct sample *smpl,
	 char *outflbl, XBOOL float_flag, XBOOL msg_flag)
{
  int k, j, tmp;
  float *tmpfdata = NULL;
  double dmm;
  char cb[MAX_MESSAGE] = "";
  FILE *fp;

  init_vqlabel(bookp,smpl);
  
  // LBG Algorithm
  if (msg_flag == XTRUE) fprintf(stderr,"Calculating Codebook\n");

  // initial centroid calculation
  bookp->vecno=0;
  centroid(condana,bookp,smpl);

  while(bookp->vecno+1 < bookp->vecsize){

      sprintf(cb, "%s%d.mat", outflbl, bookp->vecno+1);
      if ((fp = fopen(cb, "wb")) == NULL) {
	  fprintf(stderr, "Can't open file: %s\n", cb);
	  exit(1);
      }
      if (float_flag == XFALSE) {
	  fwrite(bookp->vpara, sizeof(double),
		 (bookp->vecno + 1) * bookp->vecorder, fp);
      } else {
	  tmpfdata = falloc((bookp->vecno + 1) * bookp->vecorder);
	  tmp = (bookp->vecno + 1) * bookp->vecorder;
	  for (j = 0; j < tmp; j++) tmpfdata[j] = (float)bookp->vpara[j];
	  fwrite(tmpfdata, sizeof(float),
		 (bookp->vecno + 1) * bookp->vecorder, fp);
	  free(tmpfdata);	tmpfdata = NULL;
      }
      fclose(fp);
      if (msg_flag == XTRUE) fprintf(stderr, "wrote %s\n", cb);
  
    splitting(condana,bookp,smpl);
    if (msg_flag == XTRUE)
	printf("***splitting (# of vectors=%d)***\n",bookp->vecno+1);

    for(k=0,dmm=99999.9;k<LBG_LOOP;k++){

      label(condana,bookp,smpl);
      if (msg_flag == XTRUE) 
	  printf("  #=%d , distortion=%10.6f, # of iteration=%d\n",
		 bookp->vecno+1,bookp->dist,k+1);
    
      if (dmm - bookp->dist < LBG_EPS && k>3) break;
      if ( k >= 20 && dmm - bookp->dist < 2*LBG_EPS ) break;
      if ( k >= 25 && dmm - bookp->dist < 3*LBG_EPS ) break;
      if ( k >= 30 && dmm - bookp->dist < 4*LBG_EPS ) break;
  
      // iterate for local minimum
      dmm = bookp->dist;
      centroid(condana,bookp,smpl);
    }

  }

  return;
}

void centroid(struct analysis *condana, struct codebook *bookp, struct sample *smpl)
{
  static int lll=0;
  int i,total,in,iii;
  struct avpara avpr;

  avpr.avp = dalloc(bookp->vecorder);

  total=0;
  lll ++;
  if (lll > smpl->nfrms)     lll=1;
  if (lll < 1)               lll=1;

  for(iii=0;iii < bookp->vecno+1;iii++){
    init_vparam(&avpr,bookp);

    for(i=0,in=0;i<smpl->nfrms;i++) { 
      if(bookp->vlab[i]==iii){	
	in+=1;
	sum_param(&avpr,smpl,i,bookp,condana);
      } 
    } 
    if(in==0) { 
      bookp->vemp[iii]=1;

#ifdef DEBUG 
      fprintf(stderr,"Code[%d] empty cell exists(cent).\n",iii); 
#endif

      set_vparam(bookp,smpl,condana,iii,lll); 
      lll++; 
      if (lll > smpl->nfrms)     lll=1;
      if (lll < 1)               lll=1;
      continue; 
    } else
      { total+=in; 
	avpr.tnum = in; 
	bookp->vemp[iii]=0;
	normalize_cb(bookp,iii,&avpr);
      }
  } 
  free(avpr.avp);

  return;
}

void init_vparam(struct avpara *avpr, struct codebook *bookp)
{
    int i;
    for(i=0;i<bookp->vecorder;i++) avpr->avp[i] = 0;

    return;
} 

void sum_param(struct avpara *avpr, struct sample *smpl, int no,
	       struct codebook *bookp, struct analysis *condana) 
{ 
  int i;

  for(i=0;i<bookp->vecorder;i++)
    avpr->avp[i]+=smpl->buff[no*condana->nts+bookp->dim_s+i];

  return;
}

void set_vparam(struct codebook *bookp, struct sample *smpl, struct analysis *condana, int no1, int no2)
{
  int i;

  for(i=0;i<bookp->vecorder;i++) {
    bookp->vpara[no1*bookp->vecorder+i]
      =smpl->buff[no2*condana->nts+bookp->dim_s+i];
  }

  return;
}

void normalize_cb(struct codebook *bookp, int no, struct avpara *avpr)
{
  int i;

  for(i=0;i<bookp->vecorder;i++)
    bookp->vpara[no*bookp->vecorder+i]=avpr->avp[i]/avpr->tnum;

  return;
}

void label(struct analysis *condana, struct codebook *bookp, struct sample *smpl)
{
  int i,ii,j;
  double d,dist;

  bookp->dist=0.0;
  for(i=0;i<smpl->nfrms;i++) {
    for(d=99999.9,ii=0,j=0;j<bookp->vecno+1;j++){
      if(bookp->vemp[j]==1) continue;
      dist=dist2(condana,bookp,smpl,j,i);
      if (dist < 0) dist = dist*(-1);
      if(d>dist){
	d=dist;
	ii=j;
	if (dist < 0){
#ifdef DEBUG
	  fprintf(stderr," (Label)Dist が負値です。 smpl[%3d], book[%3d] dist[%f] emp[%d] \n",i,j,dist,bookp->vemp[j]); 
	  for(k=0;k<bookp->vecorder;k++)
	    fprintf(stderr,"bookp->vpara[%3d:%2d] = %f\n",j,k,bookp->vpara[j*bookp->vecorder+k]);
	  for(k=0;k<bookp->vecorder;k++)
	    fprintf(stderr,"smpl->buff[%3d:%2d] = %f\n",i,k,smpl->buff[i*condana->nts+bookp->dim_s+k]);
#endif	  
	}
      }
//    skip:
      continue;
    }
    bookp->dist+=d;
    bookp->vlab[i]=ii;
  }
  bookp->dist/=smpl->nfrms;

}

double dist2(struct analysis *condana, struct codebook *bookp,
	     struct sample *smpl, int no1, int no2)
{
    int i;
    double dist;

    for(i=0,dist=0;i<bookp->vecorder;i++)
	dist += (bookp->vpara[no1*bookp->vecorder+i]-smpl->buff[no2*condana->nts+bookp->dim_s+i])*(bookp->vpara[no1*bookp->vecorder+i]-smpl->buff[no2*condana->nts+bookp->dim_s+i]);

    dist = (double)sqrt( (double)dist/(double)bookp->vecorder);

    return(dist);
}

void splitting(struct analysis *condana, struct codebook *bookp, struct sample *smpl)
{
  static int gomi=0;
  int i,kk,ii,j,i1,i2;
  double dm,dist,*alf;

  // temporary
  gomi++;
  if (gomi > smpl->nfrms)     gomi=1;
  if (gomi < 1)               gomi=1;

  alf = dalloc(bookp->vecorder);

  for(i=0;i<bookp->vecno+1;i++){
    kk=(bookp->vecno+i+1);

    if(bookp->vemp[i] == 1) { 
#ifdef DEBUG
      fprintf(stderr,"Code[%d:%d] empty cell is generated (split).\n",i,kk);
#endif

      set_vparam(bookp,smpl,condana,i,gomi);
      set_vparam(bookp,smpl,condana,kk,gomi);
  // temporary
      gomi++;
      if (gomi > smpl->nfrms)     gomi=1;
      if (gomi < 1)               gomi=1;

      bookp->vemp[kk]=1;

      continue;
    }
    for(dm=0.0,ii=0,j=0;j<smpl->nfrms;j++) {
      if(bookp->vlab[j]==i){
	dist=dist2(condana,bookp,smpl,i,j);
	if (dist < 0) dist = dist*(-1);
	if(dm<dist){
	  dm=dist;
	  ii=j;
	}
	if (dist < 0){
#ifdef DEBUG
	  fprintf(stderr," (splitting1)Dist が負値です。 smpl[%3d], book[%3d] dist[%f] emp[%d] \n",
		  j,i,dist,bookp->vemp[i]); 
#endif
	}
      }
    }
	  
    for(i1=ii,dm=0.0,ii=0,j=0;j<smpl->nfrms;j++) {
      if(bookp->vlab[j]==i) {
	dist=dist3(condana,bookp,smpl,i1,j);
	if (dist < 0) dist = dist*(-1);
	if(dm<dist){
	  dm=dist;
	  ii=j;
	}
	if (dist < 0){
#ifdef DEBUG
	  fprintf(stderr," (splitting2)Dist が負値です。 smpl[%3d], smpl[%3d] dist[%f] \n",i1,j,dist); 
#endif
	}
      }
    }
    i2=ii;

    splitcode(condana,bookp,smpl,i,kk,i1,i2);
  }

  bookp->vecno += bookp->vecno + 1;
  free(alf);

  return;
}

void splitcode(struct analysis *condana, struct codebook *bookp, struct sample *smpl, int k1, int k2, int i1, int i2)
{
  int i,dim;

  dim = bookp->vecorder;

  for(i=0;i< dim;i++){
    bookp->vpara[k2* dim+i]=SPLIT_COEF1*bookp->vpara[k1* dim+i]+
      SPLIT_COEF2*smpl->buff[i1*condana->nts+bookp->dim_s+i];
    bookp->vpara[k1* dim+i]=SPLIT_COEF1*bookp->vpara[k1* dim+i]+
      SPLIT_COEF2*smpl->buff[i2*condana->nts+bookp->dim_s+i];
  }

  return;
}

double dist3(struct analysis *condana, struct codebook *bookp,
	     struct sample *smpl, int no1, int no2)
{
    int i;
    double dist;
  
    for(i=0,dist=0;i<bookp->vecorder;i++)
	dist +=(smpl->buff[no1*condana->nts+bookp->dim_s+i]-smpl->buff[no2*condana->nts+bookp->dim_s+i])*(smpl->buff[no1*condana->nts+bookp->dim_s+i]-smpl->buff[no2*condana->nts+bookp->dim_s+i]);

    dist = (double)sqrt( (double)dist/(double)bookp->vecorder);

    return(dist);
}
