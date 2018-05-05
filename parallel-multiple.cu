#include<bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
__global__ void kernel_distance(double* dfeature_arr,int d_numfeatures,int d_querys, int d_querye, double* d_dist, double* d_label,int k,int num_blocks)
{
	int id = blockIdx.x*blockDim.x +  threadIdx.x;
	int tid = threadIdx.x;
	
	int i;
	int query_size = d_querye - d_querys;
	__shared__ double queryobj[20][15]; // 20-> querysize 15 -> num_features
	int start_q = (d_querys + tid -1)*d_numfeatures;
	if(tid<query_size)
	{
		for(i =0 ;i<d_numfeatures; i++)
		{
			queryobj[tid][i] = dfeature_arr[start_q];
			start_q++;
		}
	}
	__syncthreads();
	
	__shared__ double d_arr[258][20]; // 20 -> querysize
	__shared__ int check[258];
	double d1;
	check[tid] = 1;
	int rank,j;
	int start = id*d_numfeatures;
	int end = start + d_numfeatures-1;
	for(j = 0;j<query_size;j++)
	{
		double dis = 0;
		start_q = 0;
		for(i = start;i<end;i++)
		{
			d1 = dfeature_arr[i];
			dis += (d1 - queryobj[j][start_q])*(d1 - queryobj[j][start_q]);
			start_q++;
		}
		d_arr[tid][j] = dis;
	}
	//printf("\n Label = %lf",dfeature_arr[end]);
	__syncthreads();
	
	for(j=0;j<query_size;j++)
	{
		rank = 0;
		for(i=0;i<256;i++)
		{	
			//if(check[i]==1 && dis>d_arr[i]&& i!=tid)
			//	rank++;
			if(check[i]==1 &&i!=tid )
			{
				if(d_arr[tid][j]>d_arr[i][j])
					rank++;
				if(d_arr[tid][j]==d_arr[i][j]&&tid>i)
					rank++;
			}
		
		}
		if(rank<k)
		{
			d_dist[j*(k*num_blocks)+ blockIdx.x*k + rank] = d_arr[tid][j];
			d_label[j*(k*num_blocks)+ blockIdx.x*k + rank] = dfeature_arr[end];
			//printf("\nQuery_ object = %d BlockID = %d  Rank = %d  dist = %lf  label = %lf",j,blockIdx.x,rank,d_arr[tid][j],dfeature_arr[end]);
		}
	 }

}
__global__ void kernel_knn(double *dknn,int num_blocks, double* d_dist, double *d_label, int k)
{
    int tid = threadIdx.x;
    int start[100]; // 100 -> num_blocks
    double knn[20],mi; // 20 ->k
    int i,kid = 0;
    int j,ind;
    for(i=0;i<num_blocks;i++)
    {
		start[i] = tid*num_blocks*k +  k*i;
    }

    for(i=0;i<k;i++)
    {
		mi = 1000;
		for(j=0;j<num_blocks;j++)
		{
			if(mi>d_dist[start[j]])
			{
				mi = d_dist[start[j]];
				ind = j;
			}
		}
        //start[j]++;
	//printf("\n Distance = %lf label = %lf",mi,h_label[start[ind]]);
		knn[kid] = d_label[start[ind]];
		kid++;
		start[ind]++;
    }
    double sum=0.0;
    for(i=0;i<k;i++)
    {
		//printf("\n result = %lf",h_label[i]);
		sum+=knn[i];
    }
	sum = sum/(double)k;
	dknn[tid] = sum;
}
const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ";");
            tok && *tok;
            tok = strtok(NULL, ";\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}
main()
{
    FILE* stream = fopen("winequality-red.csv", "r");

    char line[1024];
    int cnt = 1,num_features,itr =-1,i,index = 0;
    double *feature_arr;
    feature_arr = (double *)malloc(sizeof(double)*250000);
    while (fgets(line, 1024, stream)!=NULL)
    {
        char* tmp = strdup(line);
        if(itr==-1)
        {
        	while(getfield(tmp,cnt)!=NULL)
        	{	
        		cnt++;
        		tmp = strdup(line);
        	}
        	num_features = cnt -1;
        	printf("\n Number of features = %d",num_features);
        	itr++;
        }
        else
        {
		for(i=1;i<=num_features;i++)
        	{
        		//printf("feaure cnt = %d",i);
			feature_arr[index] = atof(getfield(tmp,i));
			index++;
        		tmp = strdup(line);
        	}
        	itr++;
        }
        free(tmp);
    }
    fclose(stream);
    printf("\n Reading done");
    double *dfeature_arr, *d_dist,*d_label;

    cudaEvent_t st, stop;
    cudaEventCreate(&st);
    cudaEventCreate(&stop);
    cudaMalloc((void **)&dfeature_arr,itr*num_features*sizeof(double));
    
    cudaMemcpy(dfeature_arr,feature_arr,itr*num_features*sizeof(double),cudaMemcpyHostToDevice);
    
    int train = 0.8*(float)itr;
    int j;
    printf("\n %d",train);
    for(j=1;j<=15;j++)
    {
    int querys = train + 10;
    int querye = train + 10 + j;
    int query_size = querye -querys;
    int k =5;
    //printf("\nEnter value of k - ");
    //scanf("%d",&k);
    double h_dist[train][query_size];
    double h_label[train][query_size];
    cudaMalloc((void **)&d_dist,train*query_size*sizeof(double));
    cudaMalloc((void **)&d_label,train*query_size*sizeof(double));
        
    int num_threads =256;
    int num_blocks = ceil((float)train/num_threads);
    
    cudaEventRecord(st);
    kernel_distance<<<num_blocks,num_threads>>>(dfeature_arr,num_features,querys,querye,d_dist,d_label,k,num_blocks);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st, stop);


    cudaMemcpy(h_dist,d_dist,sizeof(double)*train*query_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_label,d_label,sizeof(double)*train*query_size,cudaMemcpyDeviceToHost);
    
   /* for(i=0;i<k*num_blocks;i++)
    {
	printf("\n Distance  = %lf  Label = %lf ",h_dist[i],h_label[i]);
    }*/

    double *knn;
    knn = (double *)malloc(sizeof(double)*query_size);
    
    double *dknn;
    cudaMalloc((void **)&dknn,query_size*sizeof(double));

    double *d_dist2,*d_label2;
    cudaMalloc((void **)&d_dist2,train*query_size*sizeof(double));
    cudaMalloc((void **)&d_label2,train*query_size*sizeof(double));

    cudaMemcpy(d_dist2,h_dist,sizeof(double)*train*query_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_label2,h_label,sizeof(double)*train*query_size,cudaMemcpyHostToDevice);

    cudaEvent_t st1, stop1;
    cudaEventCreate(&st1);
    cudaEventCreate(&stop1);

    cudaEventRecord(st1);

    kernel_knn<<<1,query_size>>>(dknn, num_blocks, d_dist2, d_label2, k);
    cudaEventRecord(stop1);


    cudaEventSynchronize(stop1);
    float millisecond = 0;
    cudaEventElapsedTime(&millisecond, st1, stop1);


    cudaMemcpy(knn,dknn,sizeof(double)*query_size,cudaMemcpyDeviceToHost);
    
   
    //for(i =0; i<query_size; i++)
    //printf("\nLabel for query %d = %lf",i,knn[i]);
    
    printf("\n%d\t%lf",j,(double)(milliseconds+ millisecond));
    }
}
