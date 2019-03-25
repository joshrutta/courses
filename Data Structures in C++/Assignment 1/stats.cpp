#include "stats.h"
using namespace std;
namespace main_savitch_2C
{
	statistician::statistician()
	{	
		count = 0;
		total = 0;
		tinyest = 0;
		largest = 0;
	}

	void statistician::next(double r)
	{
		if (count==0)
		{	
			tinyest = r;
			largest = r;
		}
		count++;
		total+=r;
		if (r < tinyest)
			tinyest = r;
		else if (r > largest)
			largest = r;
	}

	void statistician::reset()
	{		
		count = 0;
		total = 0;
		tinyest = 0;
		largest = 0;
	}

	statistician operator + 
		(const statistician& s1, const statistician& s2)
	{
		statistician s3 = statistician();
		s3.count = s1.count + s2.count;
		s3.total = s1.total + s2.total;

		if ((s2.count==0)||(s1.count==0))
		{
			if (s1.count==0)
			{
				s3.tinyest=s2.tinyest;
				s3.largest=s2.largest;
			}
			if (s2.count==0)
			{
				s3.tinyest=s1.tinyest;
				s3.largest=s1.largest;
			}
		}

		else

		{ 
			if (s2.tinyest < s1.tinyest) 
				s3.tinyest = s2.tinyest;
			else s3.tinyest = s1.tinyest;	
		
			if (s2.largest  > s1.largest) 
				s3.largest = s2.largest;
			else s3.largest = s1.largest;	
		}	
		
		return s3;
	}

	statistician operator *
		(double scale, const statistician& s)
	{	
		statistician s2 = statistician();
		s2.count = s.count;
		s2.total = scale*s.total;

		if (scale >=0)
		{
			s2.tinyest = scale*s.tinyest;
			s2.largest = scale*s.largest; 
		}
		else 
		{
			s2.tinyest = scale*s.largest;
			s2.largest = scale*s.tinyest;
		}
		return s2;
	}

	bool operator ==(const statistician& s1, const statistician& s2)
	{
		if ((s1.length()==0)&&(s2.length()==0))	return true;
		else 
		{
			return ((s1.length()==s2.length())&&
				(s1.mean()==s2.mean())&&
				(s1.minimum()==s2.minimum())&&
				(s1.maximum()==s2.maximum())&&
				(s1.sum()==s2.sum()));

		}
	}
}

	
	
	
