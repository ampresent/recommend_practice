#include <cstdio>
#include <cstring>

const int inf = 0x3f3f3f3f;
const int MAXN = 1000;

int deg[MAXN];
int u[MAXN];
int eu[MAXN];
bool vis[MAXN];
int top = 0;
int S = 0;

void node_permutation(/*input*/int *A, /*output*/int *p, int n){
	S = 0;
	memset(u, 0, sizeof u);
	memset(eu, 0, sizeof eu);
	memset(vis, 0, sizeof vis);
	top = 0;
	for (int i=0;i<n;i++){
		for (int j=0;j<n;j++){
			if (A[i*n +j] > 0){
				eu[i] ++;
				deg[i] ++;
			}
		}
	}
	for (int i=0;i<n;i++){
		int mineu = inf;
		top = 0;
		for (int j=0;j<n;j++){
			if (vis[j]) continue;
			if (mineu > eu[j]){
				mineu = eu[j];
				top = 0;
				u[top++] = j;
			}
			else if (mineu == eu[j]){
				u[top++] = j;
			}
		}

		int mindeg = inf;
		int minv = -1;
		for (int j=0;j<top;j++){
			int v = u[j];
			if (mindeg > deg[v]){
				mindeg = deg[v];
				minv = v;
			}
		}
		S = S*2 + mindeg;
		p[i] = minv;
		vis[minv] = true;
		for (int j=0;j<n;j++){
			if (A[j*n +minv] > 0)
				eu[j] --;
		}
	}
}

int A[5][5] = {{0,1,1,0,1},{1,0,0,1,0},{1,0,0,0,1},{0,1,0,0,1},{1,0,1,1,0}};
int p[10];

int main(){
	//printf("sf");
	node_permutation((int*)A, p, 5);
	for (int i=0;i<5;i++)
		printf("%d,",p[i]);
	printf("\nSum density = %d\n",S);
}
