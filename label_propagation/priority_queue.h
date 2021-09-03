#include <cstring>
#include <utility>

#include <unordered_map>

using namespace std;

typedef unsigned long int uli;
typedef pair<uli, float> pif;

// define mutable min priority queue of pairs
class PairPQ {
public:
    PairPQ(uli *keys, float *values, uli size);
    PairPQ(uli size);
    ~PairPQ();
    void push(pif p);
    pif pop();
    bool empty();
    uli size();
    void increase_key(uli index, float key);
    void decrease_key(uli index, float key);
    uli get_index(uli key);

private:
    uli size_;
    uli capacity_;
    pif *pq_;
    unordered_map<uli, uli> index_;
    void swap(uli i, uli j);
    void heapify(uli i);
};





