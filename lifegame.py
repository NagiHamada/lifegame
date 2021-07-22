import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import curses

from curses import wrapper

from pycuda.compiler import SourceModule


BLOCKSIZE = 32


cell_value = lambda world, height, width, y, x: world[y % height, x % width]

row2str = lambda row: ''.join(['O' if c != 0 else '-' for c in row])

def print_world(stdscr, gen, world):

    #盤面をターミナルに出力

    stdscr.clear()
    stdscr.nodelay(True)
    scr_height, scr_width = stdscr.getmaxyx()
    height, width = world.shape
    height = min(height, scr_height)
    width = min(width, scr_width - 1)
    for y in range(height):
        row = world[y][:width]
        stdscr.addstr(y, 0, row2str(row))
    stdscr.refresh()

def calc_next_cell_state(world, next_world, height, width, y, x):
    cell = cell_value(world, height, width, y, x)
    next_cell = cell
    num = 0
    num += cell_value(world, height, width, y - 1, x - 1)
    num += cell_value(world, height, width, y - 1, x    )
    num += cell_value(world, height, width, y - 1, x + 1)
    num += cell_value(world, height, width, y    , x - 1)
    num += cell_value(world, height, width, y    , x + 1)
    num += cell_value(world, height, width, y + 1, x - 1)
    num += cell_value(world, height, width, y + 1, x    )
    num += cell_value(world, height, width, y + 1, x + 1)

    if cell == 0 and num == 3:
        next_cell = 1
    elif cell == 1 and num in (2, 3):
        next_cell = 1
    else:
        next_cell = 0
    next_world[y,x] = next_cell

def calc_next_world(world, next_world):

    #現行世代の盤面の状況を元に次世代の盤面を計算する

    height, width = world.shape
    for y in range(height):
        for x in range(width):
            calc_next_cell_state(world, next_world, height, width, y, x)

#def calc_next_cell_state_gpu(world, next_world, height, width, y, x):

mod = SourceModule("""
    
    __global__ void calc_next_cell_state_gpu(const int *world, int *next_world, const int height, const int width) {

    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int index = y * width + x;

    int cell, next_cell;
    int num = 0;

    if (y >= height) {
        return;
    }
    if (x >= width) {
        return;
    }

    cell = world[index];

    num += world[((y - 1) % height) * width + ((x - 1) % width)];
    num += world[((y - 1) % height) * width + ((x + 1) % width)];
    num += world[((y - 1) % height) * width + ( x      % width)];
    num += world[( y      % height) * width + ((x - 1) % width)];
    num += world[( y      % height) * width + ((x + 1) % width)];
    num += world[((y + 1) % height) * width + ((x - 1) % width)];
    num += world[((y + 1) % height) * width + ((x + 1) % width)];
    num += world[((y + 1) % height) * width + ( x      % width)];

    if (cell == 0 && num == 3){
        next_cell = 1;
        }
    else if (cell == 1 && (num == 2 || num == 3)){
        next_cell = 1;
        }
    else{
        next_cell = 0;
        }

    next_world[index] = next_cell;

    }


    """)

calc_next_cell_state_gpu = mod.get_function("calc_next_cell_state_gpu")


def calc_next_world_gpu(world,next_world):

    height, width = world.shape
    block = (BLOCKSIZE, BLOCKSIZE, 1)
    grid = ((width + block[0] - 1) // block[0], (height +  block[1] - 1) // block[1])

    calc_next_cell_state_gpu(cuda.In(world), cuda.Out(next_world), numpy.int32(height), numpy.int32(width), block = block, grid = grid)


def gol(stdscr, height, width):
    #状態を持つ二次元配列を生成し0or1の乱数で初期化する
    world = numpy.random.randint(2, size=(height, width), dtype = numpy.int32)

    gen = 0
    while True:
        print_world(stdscr, gen, world)

        next_world = numpy.empty((height, width), dtype = numpy.int32)
        calc_next_world_gpu(world, next_world)
        world = next_world.copy()

        gen += 1

def main(stdscr):

    gol(stdscr, 1000, 1000)

if __name__ == '__main__':
    curses.wrapper(main)


