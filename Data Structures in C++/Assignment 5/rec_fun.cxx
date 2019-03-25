#include "rec_fun.h"
#include <iostream>
#include <cassert>
using namespace std;

void binary_print(ostream& outs, unsigned int n)
{
	if (n<2)
	{
		outs<<n;
		return;
	}
	binary_print(outs,n/2);
	outs<<n%2;
	return;
}

void triangle(ostream& outs, unsigned int m, unsigned int n)
// Precondition: m<=n
// Postcondition; The function has printed a pattern of 2*(n-m+1) lines
// to the output stream outs. The first line contains m asterisks, the next
// line contains m+1 asterisks, and so on up to a line with n asterisks.
// Then the pattern is repeated backwards, going n back down to m.

{
	assert(m<=n);
	if(m==n)
	{
		for(int i = 0;i<n;i++)
		{
			outs<<"*";	
		}
	outs<<"\n";
	return;
	}
	for(int i = 0;i <m;i++)
	{
		outs<<"*";
	}
	outs<<"\n";	

	triangle(outs,m+1,n);

	for(int i = 0;i < m;i++)
	{
		outs<<"*";
	}
	outs<<"\n";
	return;		
}

double pow(double x, int n)
{	
	if(x==0)
	{
		assert(n>0);
	}
	if (n == 1) return x;
	else if (n>0)
	{
		return x*pow(x,n-1);
	}
	else 
	{
		return 1/(pow(x,-n));
	}
}
void indented_sentences(size_t m, size_t n)
// Precondition: m<=n
// Postcondition: print out the above pattern by 
// calling number from m to n 
{
	assert(m<=n);
	if(m==n)
	{
	for (int i=1;i<m;i++) cout<<"  ";
	cout<<"This was written by calling number "<<m<<".\n";
	for (int i=1;i<m;i++) cout<<"  ";
	cout<<"This was ALSO written by calling number "<<m<<".\n";
	return;	
	}
 
	for (int i=1;i<m;i++) cout<<"  ";
	cout<<"This was written by calling number "<<m<<".\n";
	indented_sentences(m+1,n);
	for (int i=1;i<m;i++) cout<<"  ";
	cout<<"This was ALSO written by calling number "<<m<<".\n";
	return;	
}
