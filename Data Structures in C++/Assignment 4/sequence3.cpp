// File: sequence3.cpp
// Class implemented: sequence (See sequence3.h for documentation.)
// INVARIANT for sequence class:
//	1. The items in the bag are stored in a linked list.
//	2. The head pointer of the list is stored in head_ptr.
//	3. The tail pointer of the list is stored in tail_ptr.
//	4. The current node pointer of the list is stored in cursor.
//	5. The member variable precursor stores the node pointer pointing
// 		to the node before the current node.
//	6. The amount of nodes in the list is stored in many_nodes.

#include <cassert>
#include <cstdlib>
#include <iostream>
#include "node1.h"
#include "sequence3.h"
using namespace std;

namespace main_savitch_5
{
	sequence::sequence()
	{	//O(1)
		head_ptr=NULL;
		tail_ptr=NULL;
		cursor = NULL;
		precursor = NULL;
		many_nodes = 0;	
	}

	sequence::sequence(const sequence& source)
	{	//O(n)
        list_copy(source.head_ptr,head_ptr,tail_ptr);
        if (source.is_item())
        {
            cursor = list_search(head_ptr, source.current());
        }
        else
        {
            cursor = NULL;
        }
        many_nodes = source.many_nodes;
	}
	
	sequence::~sequence()
	{	//O(n)
		list_clear(head_ptr);
		many_nodes = 0;
	}

	void sequence::start()
	{	//O(1)
		cursor = head_ptr;
		precursor = NULL;
	}

	void sequence::advance()
	{	//O(1)
		assert(is_item());
		precursor = cursor;
		cursor = cursor->link();	
	}

	void sequence::insert(const value_type& entry)
	{	
		//O(1)	
		if(head_ptr == NULL)
		{
			list_head_insert(head_ptr,entry);
			cursor = head_ptr;
			tail_ptr = head_ptr;
			precursor = NULL;
		}
		else if (precursor == NULL || (!is_item()) )
		{
			list_head_insert(head_ptr,entry);
			cursor = head_ptr;
		}
		else
		{
			list_insert(precursor,entry);
			cursor = precursor->link();
		}
		many_nodes++;	
	}

	void sequence::attach(const value_type& entry)
	{	//O(1)
		if (head_ptr == NULL) 
		{
			list_head_insert(head_ptr,entry);
			cursor = head_ptr;
			tail_ptr = head_ptr;
			precursor = NULL;
		}
		else if (cursor == tail_ptr||(!is_item()))
		{
			list_insert(tail_ptr, entry);
			precursor = tail_ptr;
			tail_ptr = tail_ptr->link();
			cursor = tail_ptr;
		}
		else 
		{
			list_insert(cursor,entry);
			precursor = cursor;
			cursor = cursor->link();
		}
		many_nodes++;		
	}
	
	void sequence::remove_current()
	{	
		//O(1)
		assert(is_item());
		if(cursor == head_ptr)
		{
			cursor = cursor->link();
			list_head_remove(head_ptr);
			head_ptr = cursor;
		}
		else 
		{		
			cursor = cursor->link();
			list_remove(precursor);		
		}
		many_nodes--;		
	}

	void sequence::operator=(const sequence& source)
	{   //O(n)
		if (this == &source) return;
		list_clear(head_ptr);
		list_copy(source.head_ptr, head_ptr, tail_ptr);
        if (source.is_item())
        {
            cursor = list_search(head_ptr, source.current());
        }
        else
        {
            cursor = NULL;
        }
		many_nodes = source.many_nodes;
	}
}	
	


