#include <stdio.h>
#include "network.h"
#include "calculate.h"

int main()
{
    BPnetwork N;
    printf("Initializing network...\n");
    InitNetwork(&N);
    printf("done\n");
    printf("training the network...\n");
    TrainNetwork(&N);
    printf("done");
    return 0;
}