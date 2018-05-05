#include<bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
__global__ void kernel_distance(double* dfeature_arr,int d_numfeatures,int d_query,double* d_dist, double* d_label,int k)
{
	int id = blockIdx.x*blockDim.x +  threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double d_arr[258];
	__shared__ int check[258];
	__shared__ double query_obj[50]; // 50 -> num_features
	double d1, d2;
	check[tid] = 1;
	int rank,j;
	int start = id*d_numfeatures;
	int end = start + d_numfeatures-1;
	int i;
	double dis = 0;
	int start_q = (d_query-1)*d_numfeatures;

	if(tid<d_numfeatures)
	query_obj[tid] = dfeature_arr[start_q + tid];
	__syncthreads();

	for(i = start, j= 0;i<end;i++, j++)
	{
		d1 = dfeature_arr[i];
		d2 = query_obj[j];
		dis += (d1 - d2)*(d1 - d2);
		//start_q++;
	}
	//d_dist[id] = dis;
	d_arr[tid] = dis;
	//printf("\n Label = %lf",dfeature_arr[end]);
	__syncthreads();
	rank = 0;
	for(i=0;i<256;i++)
	{
		//if(check[i]==1 && dis>d_arr[i]&& i!=tid)
		//	rank++;
		if(check[i]==1 &&i!=tid )
		{
			if(dis>d_arr[i])
				rank++;
			if(dis==d_arr[i]&&tid>i)
				rank++;
		}
		
	}

	if(rank<k)
	{
		d_dist[blockIdx.x*k+rank] = dis;
		d_label[blockIdx.x*k + rank] = dfeature_arr[end];
		//printf("\n BlockID = %d  Rank = %d  dist = %lf  label = %lf",blockIdx.x,rank,dis,d_label[blockIdx.x*k + rank]);
	}
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
    printf("\n %d",train);
    int query = train + 10;
    int k ;
    //printf("\nEnter value of k - ");
    //scanf("%d",&k);
    //for(k=3;k<=20;k++){
    double h_dist[train];
    double h_label[train];
    cudaMalloc((void **)&d_dist,train*sizeof(double));
    cudaMalloc((void **)&d_label,train*sizeof(double));
        
    int num_threads =256;
    int num_blocks = ceil((float)train/num_threads);
     for(k=3;k<=20;k++){
    cudaEventRecord(st);
    kernel_distance<<<num_blocks,num_threads>>>(dfeature_arr,num_features,query,d_dist,d_label,k);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st, stop);


    cudaMemcpy(h_dist,d_dist,sizeof(double)*train,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_label,d_label,sizeof(double)*train,cudaMemcpyDeviceToHost);
    
    int start[num_blocks];
    double knn[k],mi;
    int kid = 0;
    int j,ind;
    /*for(i=0;i<k*num_blocks;i++)
    {
	printf("\n %lf   %lf",h_label[i],h_dist[i]);
    }*/
    clock_t s, e;
    s = clock();
    for(i=0;i<num_blocks;i++)
    {
	start[i] = k*i;
	//printf("\n i = %d, start = %d",i,start[i]);
    }

    for(i=0;i<k;i++)
    {
	mi = 1000;
	for(j=0;j<num_blocks;j++)
	{
		if(mi>h_dist[start[j]])
		{
			mi = h_dist[start[j]];
			ind = j;
		}
	}
        //start[j]++;
	//printf("\n Distance = %lf label = %lf",mi,h_label[start[ind]]);
	knn[kid] = h_label[start[ind]];
	kid++;
	start[ind]++;
    }

    double sum=0.0;
    for(i=0;i<k;i++)
    {
		//printf("\n result = %lf",h_label[i]);
		sum+=knn[i];
    }
    //printf("\n Sum = %lf",sum);
    sum = sum/(double)k;
    e = clock();
    //printf("\nLabel = %lf",sum);
    
    printf("\n %d\t%lf",k,((double) (e - s))* 1000.0 / CLOCKS_PER_SEC +  (double)milliseconds);}
}
