from models.jessie.first import train_eval_loop as jessie
from models.matt.first import main as matt

jessie('dataset/lips', 'models/jessie/lip.pt')
# matt('dataset/lips', 'models/matt/lip.pt')

# TODO!!
# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/2
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight