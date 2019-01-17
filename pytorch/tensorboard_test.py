import math as m
from tensorboardX import SummaryWriter


writer = SummaryWriter()

funcs = {"sin":m.sin, "cos":m.cos, "tan":m.tan}

for angle in range(0,360):
    angle_rad = angle * m.pi/180
    for name, fun in funcs.items():
        val = fun(angle_rad)
        writer.add_scalar(name, val, angle)
    writer.close()

# TODO: fix bug when calling add_scalar...