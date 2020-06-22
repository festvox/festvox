from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/tbgraph')
writer.add_graph('net', 'images')
writer.close()

