#include <iostream>
#include <cstdio>
#include <cstring>

#define CELL_WIDTH 3
#define CELL_HEIGHT 3
#define WIDTH 3
#define HEIGHT 3
#define BOARD_WIDTH CELL_WIDTH * WIDTH
#define BOARD_HEIGHT CELL_HEIGHT * HEIGHT
#define MAX CELL_WIDTH * CELL_HEIGHT

using namespace std;

int board[BOARD_HEIGHT][BOARD_WIDTH];
bool given_val[BOARD_HEIGHT][BOARD_WIDTH];

bool col[BOARD_WIDTH][MAX];
bool row[BOARD_HEIGHT][MAX];
bool cell[HEIGHT][WIDTH][MAX];

int i, j;

void solve(int y, int x);
void print_sol();
FILE *fout = NULL;

int main()
{
    FILE *fp = NULL;
    //const char *file = "097";
    //fp = fopen("test097.inp", "r");
    //fout = fopen("test097.out", "w");
    fp = fopen("../sample_puzzle.dat", "r");
    fout = fopen("../sample_puzzle.sol", "w");
    if(fp == NULL)
    {
        printf("Cannot open file!\n");
        return -1;
    }

    memset(row, 0, sizeof(row));
    memset(col, 0, sizeof(col));
    memset(cell, 0, sizeof(cell));

    for(i = 0; i < BOARD_HEIGHT; i++)
        for(j = 0; j < BOARD_WIDTH; j++)
        {
            fscanf(fp, "%d", &board[i][j]);
            //if(board[i][j] != 0) given_val[i][j] = 1;//given_val[i][j] = board[i][j];
            if(board[i][j] != 0)
            {
                given_val[i][j] = 1;
                col[j][board[i][j] - 1] = 1;
                row[i][board[i][j] - 1] = 1;
                cell[i / CELL_HEIGHT][j / CELL_WIDTH][board[i][j] - 1] = 1;
            }
        }

    solve(0,0);

    fclose(fp);
    fclose(fout);
    return 0;
}

void solve(int y, int x)
{
    if(given_val[y][x])
    {
        //return; // wrong!
        if(x == BOARD_WIDTH - 1)
            solve(y + 1, 0);
        else
            solve(y, x + 1);
    }

//    if(y == BOARD_HEIGHT - 1 && x == BOARD_WIDTH - 1)
//    {
//        print_sol();
//        return;
//    }
    int i; // what will happen if we use global variable instead!?
    for(i = 1; i <= MAX; i++)
    {
        if(1
           && !given_val[y][x] // try to comment this line and see what happen to given value
           //&& board[y][x] == 0
           && col[x][i - 1] == 0
           && row[y][i - 1] == 0
           && cell[y / CELL_HEIGHT][x / CELL_WIDTH][i - 1] == 0
           && 1
           )
        {
            col[x][i - 1] = 1;
            row[y][i - 1] = 1;
            cell[y / CELL_HEIGHT][x / CELL_WIDTH][i - 1] = 1;

            board[y][x] = i;
            if(y == BOARD_HEIGHT - 1 && x == BOARD_WIDTH - 1)
            {
                print_sol();
                return;
            }
            if(x == BOARD_WIDTH - 1)
                solve(y + 1, 0);
            else
                solve(y, x + 1);

            board[y][x] = 0;
            col[x][i - 1] = 0;
            row[y][i - 1] = 0;
            cell[y / CELL_HEIGHT][x / CELL_WIDTH][i - 1] = 0;
        }
    }
}

void print_sol()
{
    int i, j;
    for(i = 0; i < BOARD_HEIGHT; i++)
    {
        for(j = 0; j < BOARD_WIDTH; j++)
        {
            fprintf(fout, "%2d", board[i][j]);
        }
        fprintf(fout, "\n");
    }
    //fprintf(fout, "\n"); //For multiple result only
}
