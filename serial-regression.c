#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int partition (double dist[],double res[], int low, int high)
{
    int pivot = dist[high];    
    int i = (low - 1);  
 	double temp;
 	int j;
    for ( j = low; j <= high- 1; j++)
    {
        if (dist[j] <= pivot)
        {
            i++; 
            //swap(&arr[i], &arr[j]);
            temp = dist[i];
            dist[i] = dist[j];
            dist[j] = temp;
            
            temp = res[i];
            res[i] = res[j];
            res[j] = temp;
        }
    }
    //swap(&arr[i + 1], &arr[high]);
    temp = dist[i+1];
	dist[i+1] = dist[high];
	dist[high] = temp;
            
	temp = res[i+1];
	res[i+1] = res[high];
	res[high] = temp;
    return (i + 1);
}

void quickSort(double dist[], double res[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(dist,res, low, high);
         quickSort(dist,res, low, pi - 1);
        quickSort(dist,res, pi + 1, high);
    }
}


void insertion_sort(double dist[], double res[], int n)
{
   int i, j;
   double key, val;
   for (i = 1; i < n; i++)
   {
       key = dist[i];
       val = res[i];
       j = i-1;
       while (j >= 0 && dist[j] > key)
       {
           dist[j+1] = dist[j];
           res[j+1] = res[j];
           j = j-1;
       }
       dist[j+1] = key;
       res[j+1] = val;
   }
}
double KNN(double feature_arr[][25],int num_features,int train,int query,int k)
{
	double dist[train+1],result[train+1];
	int i,j;
	for(i=0;i<train;i++)
	{
		//Euclidean distance
		float dis = 0;
		for(j=0;j<num_features-1;j++)
		dis += (feature_arr[i][j] - feature_arr[query][j])*(feature_arr[i][j] - feature_arr[query][j]);
		
		dist[i] = (double)sqrt(dis);
		
		result[i] = feature_arr[i][num_features-1];
	}
	//insertion_sort(dist,result,train);
	quickSort(dist,result,0,train -1);
	double sum=0.0;
	for(i=0;i<k;i++)
	{
		//printf("\n result = %lf",result[i]);
		sum+=result[i];
	}
	return sum/(double)k;
	
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
int main()
{
    FILE* stream = fopen("winequality-red.csv", "r");

    char line[1024];
    int cnt = 1,num_features,itr =-1,i;
    double feature_arr[10000][25];
    while (fgets(line, 1024, stream)!=NULL)
    {
        char* tmp = strdup(line);
        if(itr==-1)
        {
        	while(getfield(tmp,cnt)!=NULL)
        	{	
        		cnt++;
        		//printf("\n Count = %d",cnt);
        		tmp = strdup(line);
        	}
        	num_features = cnt -1;
        	//printf("\n Number of features = %d",num_features);
        	itr++;
        }
        else
        {
        	//printf("\n itr = %d",itr);
        	for(i=1;i<=num_features;i++)
        	{
        		feature_arr[itr][i-1] = atof(getfield(tmp,i));
        		tmp = strdup(line);
        	}
        	itr++;
        }
        free(tmp);

    }
    int train = 0.8*(float)itr;
    printf("\n %d",train);
    int query = train + 10;
    int k ;
    //printf("\nEnter value of k - ");
    //scanf("%d",&k);
    for(k=3;k<=20;k++)
    {
    clock_t start, end;
	start = clock();
    double ans = KNN(feature_arr,num_features,train,query,k);
    end = clock();
    //printf("\nLabel = %lf",ans);
    printf("\n %d\t%lf",k,((double) (end - start))*1000.0 / CLOCKS_PER_SEC);
    }
    
    fclose(stream);
}
