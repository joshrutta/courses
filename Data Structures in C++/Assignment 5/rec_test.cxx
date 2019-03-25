#include "rec_fun.h"
#include <iostream>
using namespace std;

int main()
{
	//Testing binary_print
	cout<<"\n"<<"Testing binary_print \n\n";
	int n1 = 1;
	int n2 = 4;	
	int n3 = 31;
	cout<<"n = "<<n1<<" Output: ";
	binary_print(cout,n1);
	cout<<"\n"<<"n = "<<n2<<" Output: ";
	binary_print(cout,n2);
	cout<<"\n"<<"n = "<<n3<<" Output: ";
	binary_print(cout,n3);
	cout<<"\n\n";

	//Testing triangle
	cout<<"Testing triangle \n\n";
	cout<<"Calling triangle(3,5):\n\n";
	triangle(cout,3,5);
	cout<<"\n\n";
	//Testing pow function
	cout<<"Testing pow function \n\n";
	cout<<"pow(-2,4) = "<<pow(-2,4)<<"\n";
	cout<<"pow(0,10) = "<<pow(0,10)<<"\n";
	cout<<"pow(5,3) = "<<pow(5,3)<<"\n\n";
	//Testing indented_sentences
	cout<<"Testing indented_sentences \n\n";
	cout<<"Calling indented_sentences(1,6):\n\n";
	indented_sentences(1,6);
	cout<<"\n";
	cout<<"Calling indented_sentences(3,5): \n\n";
	indented_sentences(3,5);
	cout<<"\n";
	return 0;
	
}
