#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ALIGN 64
typedef struct {
    double x, y;
}Point;


typedef struct{
    Point p1, p2, p3, p4, P;
    void *NW, *NE, *SW, *SE;
}Node;
void printNode(Node *node){
    printf("Node: \n");
    printf("\tP1(%7.3lf, %7.3lf) P2(%7.3lf, %7.3lf) P3(%7.3lf, %7.3lf) P4(%7.3lf, %7.3lf)\n", node->p1.x, node->p1.y, node->p2.x, node->p2.y, node->p3.x, node->p3.y, node->p4.x, node->p4.y);
    printf("\tNW[%p] NE[%p] SW[%p] SE[%p]\n", node->NW, node->NE, node->SW, node->SE);
    printf("\n\n");
}

int AABB(Node *node, Point p){
    double minX = node->p1.x,
           minY = node->p1.y,
           maxX = node->p3.x,
           maxY = node->p3.y;

    if (p.x < minX) return 0;
    if (p.x > maxX) return 0;
    if (p.y < minY) return 0;
    if (p.y > maxY) return 0;

    return 1;
}

void insert(Node *node, Point p){
    if (!AABB(node, p)) return; //critério de parada
        printf("Tem elemento para inserir \n");
    Node *NW, *NE, *SW, *SE;
    NW = (Node*) aligned_alloc(ALIGN, sizeof(Node));bzero(NW, sizeof(Node));
    NE = (Node*) aligned_alloc(ALIGN, sizeof(Node));bzero(NE, sizeof(Node));
    SW = (Node*) aligned_alloc(ALIGN, sizeof(Node));bzero(SW, sizeof(Node));
    SE = (Node*) aligned_alloc(ALIGN, sizeof(Node));bzero(SE, sizeof(Node));

}


int main(int ac, char **av){
    Node *root = NULL;

    Point vet[7];
    vet[0].x = 1.25f; vet[0].y = 1.25;
    vet[1].x = 1.25f; vet[1].y = 1.625;
    vet[2].x = 1.625f; vet[2].y = 1.25;
    vet[3].x = 1.625f; vet[3].y = 1.625;
    vet[4].x = 8.35f; vet[4].y = 8.34;
    vet[5].x = 13.5f; vet[5].y = 13.5;
    vet[6].x = 15.0f; vet[6].y = 15.0;


    root = (Node*) aligned_alloc(ALIGN, sizeof(Node));
    bzero(root, sizeof(Node));

    root->p1.x = 0.0f;root->p1.y = 0.0f;
    root->p2.x = 0.0f;root->p2.y = 20.0f;
    root->p3.x = 20.0f;root->p3.y = 20.0f;
    root->p4.x = 20.0f;root->p4.y = 0.0f;
    printf("\nExemplo de Quadtree\n");
    printNode(root);
    for (unsigned int i = 0; i < 7; i++)
        printf("Testando(%7.3lf, %7.3lf) = [%d] \n", vet[i].x, vet[i].y, AABB(root, vet[i]));


    insert(root, vet[0]);
    free(root);
    return EXIT_SUCCESS;

}
