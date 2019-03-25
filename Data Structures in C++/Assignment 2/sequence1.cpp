#include "sequence1.h"
#include <cassert>
using namespace std;
namespace main_savitch_3
{
	sequence::sequence()
	{
		current_index = 0;
		used = 0;
	}

	void sequence::start()
	{
		current_index = 0;
	}
	
	void sequence::advance()
	{
		assert(is_item());
		current_index++;
	}

	void sequence::insert(const value_type& entry)
	{
		assert( size() < CAPACITY);
		if(is_item())
		{
			for(int i = used; i > current_index; i--)
			{
			 data[i]=data[i-1];		
			}
			data[current_index] = entry;
		}
		else if (current_index == 0){data[current_index] = entry;}
		else 
		{
			current_index=0;
			for(int i = used ; i > current_index; i--)
			{
			 data[i]=data[i-1];		
			}
			data[current_index] = entry;				
		}
			used++;
	}	

	void sequence::attach(const value_type& entry)
	{
		assert( size() < CAPACITY);	
		if(is_item())	
		{
			for(int i = used; i > current_index; i--)
			{
			 data[i]=data[i-1];		
			}
			data[current_index+1] = entry;
			current_index++;	
		}

		else if (current_index == 0){data[current_index] = entry;} 
		else 
		{
			current_index=used;
			data[current_index] = entry;				
		}
		used++;
	}

	void sequence::remove_current()
	{
		assert(is_item());
		for (int i = current_index; i < used - 1; i++)
		{
			data[i] = data[i+1];		
		}			
		used--;			
	}	

}
