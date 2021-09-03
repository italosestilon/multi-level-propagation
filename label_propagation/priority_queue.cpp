#include <label_propagation/priority_queue.h>

PairPQ::PairPQ(uli *keys, float *values, uli size) {
    size_ = size;
    capacity_ = size;
    pq_ = new pif[size];
    for (uli i = 0; i < size; i++) {
        pq_[i] = make_pair(keys[i], values[i]);
        push(pq_[i]);
    }
}

PairPQ::PairPQ(uli size) {
    size_ = 0;
    capacity_ = size;
    pq_ = new pif[size];
}

PairPQ::~PairPQ() {
    delete[] pq_;
}

pif PairPQ::pop() {
    pif p = pq_[0];
    swap(0, size_ - 1);
    size_--;
    heapify(0);
    return p;
}

void PairPQ::push(pif p) {
    if (size_ == capacity_) {
        capacity_ *= 2;
        pif *new_pq = new pif[capacity_];
        uli *new_index = new uli[capacity_];

        memcpy(pq_, new_pq, sizeof(pif) * size_);
        
        delete[] pq_;

        pq_ = new_pq;
    }
    size_++;
    pq_[size_ - 1] = p;
    index_[p.first] = size_ - 1;
    heapify(size_ - 1);
}

void PairPQ::swap(uli i, uli j) {
    pif temp = pq_[i];
    pq_[i] = pq_[j];
    pq_[j] = temp;
    uli temp_index = index_[i];
    index_[temp.first] = j;
    index_[pq_[i].second] = i;
}

void PairPQ::heapify(uli i) {
    uli l = 2 * i + 1;
    uli r = 2 * i + 2;
    uli smallest = i;
    if (l < size_ && pq_[l].second < pq_[i].second) {
        smallest = l;
    }
    if (r < size_ && pq_[r].second < pq_[smallest].second) {
        smallest = r;
    }
    if (smallest != i) {
        swap(i, smallest);
        heapify(smallest);
    }
}

void PairPQ::decrease_key(uli index, float key) {
    pq_[index].first = key;
    uli i = index;
    while (i > 0 && pq_[(i - 1) / 2].second > pq_[i].second) {
        swap(i, (i - 1) / 2);
        i = (i - 1) / 2;
    }
}

uli PairPQ::get_index(uli key) {
    return index_[key];
}