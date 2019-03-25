//
//  main.cpp
//  Assignment 6
//
//  Created by Josh Rutta on 11/16/17.
//  Copyright © 2017 Josh Rutta. All rights reserved.
//

#include <iostream>
#include "bag6.h"
#include "bintree.h"
using namespace std;
using namespace main_savitch_10;

int main(int argc, const char * argv[]) {
    // insert code here...
    bag<int> *example = new bag<int>;
    example->insert(2);
    example->insert(1);
    example->insert(3);
    example->insert(3);
    example->insert(3);
    cout<<"example.count(3) = "<<example->count(3)<<"\n";
    cout<<"Calling example->erase(3), items removed: "<<example->erase(3)<<"\n";
    cout<<"example.count(3) = "<<example->count(3)<<"\n";
    
    return 0;
}
