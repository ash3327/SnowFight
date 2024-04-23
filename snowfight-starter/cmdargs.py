import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, help="The render mode",
                    choices=['ai', 'human', 'human_rand', 'rgb_array'],
                    default='rgb_array')
parser.add_argument('-n', '--size', type=int, help='The board size',
                    choices=range(10,100), metavar='[5-99]',
                    default=30)
parser.add_argument('-ne', '--enemies', type=int, help='Number of enemies on the board.',
                    choices=range(1,5), metavar='[1-5]',
                    default=5)
parser.add_argument('-s', "--seed", type=int, 
                    help="The seed for random number generator", 
                    default=None)
parser.add_argument('-e', "--episodes", type=int, 
                    help="The number of episodes.", 
                    default=1000)
parser.add_argument('-ms', "--max_steps", type=int, 
                    help="The maximum number of steps in an episode", 
                    default=1000)
parser.add_argument('-fps', "--fps", type=int, 
                    help="The rendering speed in frames per second",
                    default=None)
parser.add_argument('-f', "--file", type=str, 
                    help="The file name of the Q-table file",
                    default=None)
parser.add_argument('-o', "--output_file", type=str,
                    help="The file name of the output evaluation files",
                    default=None)
parser.add_argument('-t', "--n_threads", type=int,
                    help="Number of threads during training. Default is 6 and if you have a cpu with hexacore, it is advised to use 10.",
                    choices=range(1,40), metavar='[1-39]',
                    default=6)
parser.add_argument('-ws', "--window_size", type=int,
                    help="Setting the relative size of the window (16 is the largest and default)",
                    choices=range(1, 17), metavar='[1-16]', default=16)
args = parser.parse_args()
print(args)
