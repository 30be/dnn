import torch
import math


def draw_detector():
    a = torch.zeros((200, 200))


def draw_detector2():
    a = torch.zeros((200, 200))
    x_grid, y_grid = torch.linspace
    for i in range(10):
        a[36 * (i + 1) < math.atan2(y_grid, x_grid) < 36 * i] = i
