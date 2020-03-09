import torch
from PIL import Image
from models.jessie.first import main, build, test_transforms
# from models.matt.first import main, build, test_transforms

dir = 'dataset/lips'
model_name = 'models/jessie/lip.pt'

"""TRAINING"""
# main(dir, model_name)

"""TESTING"""
def load_image(path):
    image = Image.open(path)
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = test_transforms(image)[:3, :, :].unsqueeze(0)
    return image

def predict(path, model):
    image = load_image(path)
    model = model.cpu()
    model.eval()
    return torch.argmax(model(image))

model, device, criterion, optimizer = build()
model.load_state_dict(torch.load(model_name))


from glob import glob

class_names = ['no', 'yes']
img_dir = 'image_jessie/dataset_tongue'
notkawa = np.array(glob(img_dir + "/no/*"))
kawa = np.array(glob(img_dir + "/yes/*"))

# color green means the model predicted correctly
# range can be chosen arbitrarily as long as it stays in bound
for i in range(15, 105, 30):
    img_path = notkawa[i]
    img = Image.open(img_path)
    if predict_kawasaki(model, class_names, img_path) == 'no':
        print(colored('Not Kawasaki', 'green'))
    else:
        print(colored('Kawasaki', 'red'))
    plt.imshow(img)
    plt.show()

print("------------------------------------------------------")

for i in range(13, 103, 30):
    img_path = kawa[i]
    img = Image.open(img_path)
    if predict_kawasaki(model, class_names, img_path) == 'yes':
        print(colored('Kawasaki', 'green'))
    else:
        print(colored('Not Kawasaki', 'red'))
    plt.imshow(img)
    plt.show()



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