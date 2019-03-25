//FILE: sequence2.cpp
//CLASS IMPLEMENTED: sequence (see sequence2.h for details)
//INVARIANT for the sequence class:
//	1. For an empty sequence, we don’t care what is stored in any of data; for a non-
//	empty sequence, the items are stored in data[0] through data[used-1], where data 
//	is a member variable that points to a dynamic array, and we don’t care about what 
//	is stored in the rest of data.
//	2. The number of items in the sequence is in the member variable used
//	3. The current item in the sequence object is in the member variable current_index 
//	4. The capacity of the sequence object is in the member variable capacity


#include "sequence2.h"
#include <cassert>
#include <algorithm>
using namespace std;
namespace main_savitch_4
{
	sequence::sequence(size_type initial_capacity)
	{
		data = new value_type[initial_capacity];
		used = 0;
		capacity = initial_capacity;
		current_index = 0;
	}

	sequence::sequence(const sequence& source)
	{
		data = new value_type[source.capacity];
		capacity = source.capacity;
		used = source.used;
		current_index = source.current_index;
		copy(source.data, source.data + used, data);
	}

	sequence::~sequence()
	{
		delete [] data;
	}

	void sequence::resize(size_type new_capacity)
	{
		value_type *new_data;
		
		if (new_capacity == capacity) return;
		
		if (new_capacity < used) new_capacity = used;

		new_data = new value_type[new_capacity];
		copy(data,data+used,new_data);
		delete [] data;
		data = new_data;
		capacity = new_capacity;
	}

	void sequence::operator=(const sequence& source)
	{
		value_type *new_data;
		if (this == &source) return;
		if (capacity != source.capacity)
		{
			new_data = new value_type[source.capacity];
			delete [] data;
			data = new_data;
			capacity = source.capacity;
		}
	
		used = source.used;
		current_index = source.current_index;
		copy(source.data,source.data+used,data);
	}	

	void sequence::operator+=(const sequence& addend)
	{	
		if(addend.used != 0)
		{
			value_type *new_data;
			new_data = new value_type[capacity + addend.capacity];
			//copying data from addend to new_data
			copy(addend.data,addend.data+addend.used,new_data+used+1);
			//copying old data to new_data
			copy(data,data+used,new_data);
			data = new_data;
			used = used + addend.used;
			capacity = capacity + addend.capacity;
		}
		return;	
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
		if(used == capacity)
		{
			resize(int(capacity*(1.1)));
		}
		if (is_item())
		{
			for(int i = used; i > current_index; i--)
			{
				data[i] = data[i-1];	
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
		if (used == capacity)
		{
			resize(int(capacity*(1.1)));
		}
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

	sequence::value_type sequence::operator[](size_type index) const 
	{
		assert(index < used);
		return data[index];
		
	}

	sequence operator+(const sequence& s1, const sequence& s2)
	{
		sequence answer(s1.size() + s2.size());
		answer+=s1;
		answer+=s2;
		return answer;		
	}	
}






