#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float random_float()
{
    float scale = rand() / (float) RAND_MAX; // 0 <= scale <= 1, [0, 1]
    return -1 + scale * 2; // first, multiply by 2 -> 0*2 <= 2*scale <= 1*2, then subtract by 1 ->  0 - 1 <= 2*scale - 1, [-1, 1]
}

int main(void)
{
    int number_in_circle = 0, number_of_tosses = 1 << 24;
    srand(time(NULL));

    for (int toss = 0; toss < number_of_tosses; toss++)
    {
        float x = random_float();
        float y = random_float();
        float distance_squared = x*x + y*y;
        if (distance_squared <= 1) number_in_circle++;
    }
    float estimated_pi = 4.0f * number_in_circle / ((double) number_of_tosses);
    printf("Estimated Pi = %f\n", estimated_pi);

    return 0;

}