import foolbox
import numpy as np
import torchvision.models as models
import torch.nn as nn

# instantiate model (supports PyTorch, Keras, TensorFlow (Graph and Eager), MXNet and many more)
model = models.resnet18(pretrained=True).eval()
# model.fc=nn.Linear(model.fc.in_features,10)

preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

# get a batch of images and labels and print the accuracy
images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=16, data_format='channels_first', bounds=(0, 1))
print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))
# -> 0.9375

# apply the attack
attack = foolbox.attacks.FGSM(fmodel)
adversarials = attack(images, labels)
# if the i'th image is misclassfied without a perturbation, then adversarials[i] will be the same as images[i]
# if the attack fails to find an adversarial for the i'th image, then adversarials[i] will all be np.nan

# Foolbox guarantees that all returned adversarials are in fact in adversarials
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
print(adversarials[0])
print(type(adversarials[0]))
print(adversarials[0].shape)
print(type(images[0]))
print(images[0].shape)

# # -> 0.0
#
# # ---
#
# # In rare cases, it can happen that attacks return adversarials that are so close to the decision boundary,
# # that they actually might end up on the other (correct) side if you pass them through the model again like
# # above to get the adversarial class. This is because models are not numerically deterministic (on GPU, some
# # operations such as `sum` are non-deterministic by default) and indepedent between samples (an input might
# # be classified differently depending on the other inputs in the same batch).
#
# # You can always get the actual adversarial class that was observed for that sample by Foolbox by
# # passing `unpack=False` to get the actual `Adversarial` objects:
# attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.Linf)
# adversarials = attack(images, labels, unpack=False)
#
# adversarial_classes = np.asarray([a.adversarial_class for a in adversarials])
# print(labels)
# print(adversarial_classes)
# print(np.mean(adversarial_classes == labels))  # will always be 0.0
# # The `Adversarial` objects also provide a `distance` attribute. Note that the distances
# # can be 0 (misclassified without perturbation) and inf (attack failed).
# distances = np.asarray([a.distance.value for a in adversarials])
# print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))
# print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))
# print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))