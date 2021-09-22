#include <label_propagation/priority_queue.h>
#include <gtest/gtest.h>

// test push and pop in priority queue
TEST(PriorityQueue, PushAndPop) {
    PairPQ pq = PairPQ(10);
    pq.push(make_pair(3, 3.0));
    pq.push(make_pair(1, 1.0));
    pq.push(make_pair(2, 2.0));

    pif top = pq.pop();

    EXPECT_EQ(top.first, 1);
    EXPECT_EQ(top.second, 1.0);

    pq.push(make_pair(4, 0.1));
    pq.push(make_pair(5, 0.2));

    EXPECT_EQ(pq.pop().first, 4);
    EXPECT_EQ(pq.pop().first, 5);
    EXPECT_EQ(pq.pop().first, 2);
    EXPECT_EQ(pq.pop().first, 3);
}

// test indexes in priority queue
TEST(PriorityQueue, Indexes) {
    PairPQ pq = PairPQ(10);
    pq.push(make_pair(3, 1.5));
    pq.push(make_pair(1, 1.0));
    pq.push(make_pair(2, 2.0));

    EXPECT_EQ(pq.get_index(1), 0);
    EXPECT_EQ(pq.get_index(3), 1);
    EXPECT_EQ(pq.get_index(2), 2);


    pq.push(make_pair(4, 0.1));
    EXPECT_EQ(pq.get_index(4), 0);
    EXPECT_EQ(pq.get_index(1), 1);
    EXPECT_EQ(pq.get_index(3), 3);
    EXPECT_EQ(pq.get_index(2), 2);

}

// test decrease key
TEST(PriorityQueue, DecreaseKey) {
    PairPQ pq = PairPQ(10);
    pq.push(make_pair(1, 1.0));
    pq.push(make_pair(3, 1.5));
    pq.push(make_pair(2, 2.0));

    uli idx = pq.get_index(3);
    pq.decrease_key(idx, 0.5);

    pif top = pq.pop();

    EXPECT_EQ(top.first, 3);
    EXPECT_EQ(top.second, 0.5);
    EXPECT_EQ(pq.get_index(3), 0);
}

// test increase key
TEST(PriorityQueue, IncreaseKey) {
    PairPQ pq = PairPQ(10);
    pq.push(make_pair(1, 1.0));
    pq.push(make_pair(3, 1.5));
    pq.push(make_pair(2, 2.0));
    pq.push(make_pair(4, 2.1));
    pq.push(make_pair(5, 2.2));

    uli idx = pq.get_index(3);
    pq.increase_key(idx, 3.0);

    EXPECT_EQ(pq.get_index(3), 3);

}