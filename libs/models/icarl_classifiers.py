from typing import Iterator

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.nn import Parameter
from torch.utils.data import Dataset

import libs.utils as utils
from libs.modified_resnet import resnet32
from libs.cifar100 import Cifar100
from libs.utils import get_one_hot
from libs.modified_resnet import BiasLayer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

import copy


class ExemplarSet(Dataset):

    def __init__(self, images, labels, transforms):
        assert len(images) == len(labels)

        self.images = list(images)
        self.labels = list(labels)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]

        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class iCaRLModel(nn.Module):

    def __init__(self, train_dataset: Cifar100, num_classes=100, memory=2000, batch_size=128, classifier='fc',
                 device='cuda'):
        super(iCaRLModel, self).__init__()
        self.num_classes = num_classes
        self.memory = memory
        self.known_classes = 0
        self.old_net = None
        self.batch_size = batch_size
        self.device = device

        if classifier == 'cosine':
            self.net = resnet32(num_classes=num_classes, classifier=classifier)
        else:
            self.net = resnet32(num_classes=num_classes)

        self.dataset = train_dataset

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.exemplar_sets = []

        self.compute_means = True
        self.exemplar_means = []

        self.clf = None  # store classifiers object (SVM, KNN...) to test them
        # multiple times without fitting it at each test
        # (if no training in the meanwhile)
        self.params_clf = []

        if classifier == 'bias':
            self.criterion_bias = nn.BCEWithLogitsLoss(reduction='mean')
            self.bias_layer = BiasLayer().to(device)

    def forward(self, x):
        return self.net(x)

    def _extract_features(self, images, normalize=True):
        features = self.net(images, features=True)
        if normalize:
            features = features / features.norm(dim=1).unsqueeze(1)
        return features

    def increment_known_classes(self, n_new_classes=10):
        self.known_classes += n_new_classes

    def combine_trainset_exemplars(self, train_dataset: Cifar100):
        exemplar_indexes = np.hstack(self.exemplar_sets)
        images, labels = self.dataset.get_items_of(exemplar_indexes)
        exemplar_dataset = ExemplarSet(images, labels, utils.get_train_eval_transforms()[0])
        return utils.create_augmented_dataset(train_dataset, exemplar_dataset)

    def update_representation(self, train_dataset: Cifar100, optimizer, scheduler, num_epochs, fit_clf=None):
        self.compute_means = True
        self.net = self.net.to(self.device)

        if len(self.exemplar_sets) > 0:
            self.old_net = copy.deepcopy(self.net)
            self.old_net = self.old_net.to(self.device)
            train_dataset = self.combine_trainset_exemplars(train_dataset)

        loader = utils.get_train_loader(train_dataset, self.batch_size, drop_last=False)

        train_losses = []
        train_accuracies = []

        for epoch in range(num_epochs):
            print(f"\tSTARTING EPOCH {epoch + 1} - LR={scheduler.get_last_lr()}...")
            cumulative_loss = .0
            running_corrects = 0
            self.net.train()
            for i, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.net(images)

                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == labels.data).data.item()

                loss = self.compute_distillation_loss(images, labels, outputs)
                loss_value = loss.item()
                cumulative_loss += loss_value

                loss.backward()
                optimizer.step()

                if i != 0 and i % 20 == 0:
                    print(f"\t\tEpoch {epoch + 1}: Train_loss = {loss_value}")

            curr_train_loss = cumulative_loss / float(len(train_dataset))
            curr_train_accuracy = running_corrects / float(len(train_dataset))
            train_losses.append(curr_train_loss)
            train_accuracies.append(curr_train_accuracy)
            scheduler.step()

            print(f"\t\tRESULT EPOCH {epoch + 1}:")
            print(f"\t\t\tTrain Loss: {curr_train_loss} - Train Accuracy: {curr_train_accuracy}\n")

        if fit_clf in ['other_classifiers', 'svm']:
            self._fit_clf(fit_clf, loader)
        elif fit_clf == 'cosine':
            self.params_clf.append(self.net.get_sigma())

        return np.mean(train_losses), np.mean(train_accuracies)

    def _fit_clf(self, clf_type, dataloader):
        if clf_type == 'other_classifiers':
            clf = KNeighborsClassifier(weights="distance")
            param_grid = {"n_neighbors": [9, 11, 13, 15]}
        elif clf_type == 'svm':
            clf = Pipeline(steps=[("scaler", StandardScaler()), ("clf", SVC())])
            param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
        else:
            return

        print(f"Training {clf_type}")

        X, y = [], []

        self.net.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                y.append(labels)
                features = self._extract_features(images, normalize=False)
                X.append(features)

            X = torch.cat(X).cpu().numpy()
            y = torch.cat(y).cpu().numpy()

        rus = RandomUnderSampler()
        X, y = rus.fit_resample(X, y)
        grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, scoring='accuracy', cv=4)
        grid_search.fit(X, y)
        self.clf = grid_search.best_estimator_
        self.params_clf.append(grid_search.best_params_)

    # https://github.com/sairin1202/BIC/blob/master/trainer.py

    def _bias_training(self, bias_optimizer, scheduler_bias, eval_dataloader):

        criterion = self.criterion_bias

        if self.known_classes == 0:
            self.net.eval()
            current_step = 0
            epochs = 20

            for epoch in range(epochs):
                print(f"\tSTARTING Bias Training EPOCH {epoch + 1} - LR={scheduler_bias.get_last_lr()}...")

                # Iterate over the dataset
                for i, (images, labels) in enumerate(eval_dataloader):
                    # Bring data over the device of choice
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    bias_optimizer.zero_grad()  # Zero-ing the gradients

                    # Forward pass to the network and to the bias layer
                    outputs = self.net(images)
                    outputs = self.bias_forward(outputs, self.known_classes)

                    # One hot encoding labels for binary cross-entropy loss
                    labels_one_hot = utils.get_one_hot(labels, self.num_classes, self.device)

                    loss = criterion(outputs, labels_one_hot)

                    if i != 0 and i % 10 == 0:
                        print(f"\t\tEpoch {epoch + 1}: Train_loss = {loss.item()}")

                    loss.backward()  # backward pass: computes gradients
                    bias_optimizer.step()  # update weights based on accumulated gradients
                    current_step += 1
            scheduler_bias.step()

    def compute_distillation_loss(self, images, labels, new_outputs):
        if self.known_classes == 0:
            return self.bce_loss(new_outputs, get_one_hot(labels, self.num_classes, self.device))

        sigmoid = nn.Sigmoid()
        n_old_classes = self.known_classes
        old_outputs = self.old_net(images)

        targets = get_one_hot(labels, self.num_classes, self.device)
        targets[:, :n_old_classes] = sigmoid(old_outputs)[:, :n_old_classes]
        tot_loss = self.bce_loss(new_outputs, targets)

        return tot_loss

    def classify(self, images, method='nearest-mean'):
        self.net = self.net.to(self.device)
        self.net.eval()
        if method == 'nme':
            return self._nme(images)
        elif method == 'cosine':
            return self._cosine_similarity(images)
        elif method == 'other_classifiers' or method == 'svm':
            return self._knn_svm(images)
        elif method == 'bias':
            return self._bias_correction(images)
        elif method == 'fc':
            outputs = self.net(images)
            _, preds = torch.max(outputs.data, 1)
            return preds

    def bias_forward(self, inp, n):
        #forward as detailed in the paper:
        #apply the bias correction only to the new classes

        out_old = inp[:, :n]
        out_new = inp[:, n:]
        out_new = self.bias_layer(out_new)
        return torch.cat([out_old, out_new], dim=1)

    def _bias_correction(self, images):
        self.bias_layer.eval()
        with torch.no_grad():
            preds = self.net(images)
            preds = self.bias_layer(preds).argmax(dim=-1)
            return preds

    def _compute_means(self):
        exemplar_means = []
        for exemplar_class_idx in self.exemplar_sets:
            imgs, labs = self.dataset.get_items_of(exemplar_class_idx)
            exemplars = ExemplarSet(imgs, labs, utils.get_train_eval_transforms()[1])
            loader = utils.get_eval_loader(exemplars, self.batch_size)

            flatten_features = []
            with torch.no_grad():
                for imgs, _ in loader:
                    imgs = imgs.to(self.device)
                    features = self._extract_features(imgs)
                    flatten_features.append(features)

                flatten_features = torch.cat(flatten_features).to(self.device)
                class_mean = flatten_features.mean(0)
                class_mean = class_mean / class_mean.norm()
                exemplar_means.append(class_mean)

        self.compute_means = False
        self.exemplar_means = exemplar_means

    def _cosine_similarity(self, images):
        if self.compute_means:
            self._compute_means()

        with torch.no_grad():
            similarity = torch.nn.CosineSimilarity(dim=1)

            exemplar_means = self.exemplar_means
            means = torch.stack(exemplar_means)  # (n_classes, feature_size)
            means = torch.stack([means] * len(images))  # (batch_size, n_classes, feature_size)
            means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

            with torch.no_grad():
                feature = self._extract_features(images, normalize=True)
                feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

                dists = similarity(feature, means).squeeze()  # (batch_size, n_classes)
                _, preds = dists.max(1)

        return preds

    def _nme(self, images):
        if self.compute_means:
            self._compute_means()

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * len(images))  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        with torch.no_grad():
            feature = self._extract_features(images)
            feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
            feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

            dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
            _, preds = dists.min(1)

        return preds

    def _knn_svm(self, images):
        assert self.clf is not None

        with torch.no_grad():
            features = self._extract_features(images, normalize=False).cpu()
            preds = np.array(self.clf.predict(features))

        return torch.from_numpy(preds).to(self.device)

    def reduce_exemplar_set(self, m, label):
        # for i, exemplar_set in enumerate(self.exemplar_sets):
        self.exemplar_sets[label] = self.exemplar_sets[label][:m]

    def construct_exemplar_set(self, indexes, images, label, m, herding=True):
        if herding:
            self.herding_construct_exemplar_set(indexes, images, label, m)
        else:
            self.random_construct_exemplar_set(indexes, label, m)

    def herding_construct_exemplar_set(self, indexes, images, label, m):
        exemplar_set = ExemplarSet(images, [label] * len(images), utils.get_train_eval_transforms()[1])
        loader = utils.get_eval_loader(exemplar_set, self.batch_size)

        self.net.eval()
        flatten_features = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                features = self._extract_features(images)
                flatten_features.append(features)

            flatten_features = torch.cat(flatten_features).cpu().numpy()
            class_mean = np.mean(flatten_features, axis=0)
            class_mean = class_mean / np.linalg.norm(class_mean)
            # class_mean = torch.from_numpy(class_mean).to(self.device)
            flatten_features = torch.from_numpy(flatten_features).to(self.device)

        exemplars = set()  # lista di exemplars selezionati per la classe corrente
        exemplar_feature = []  # lista di features per ogni exemplars già selezionato
        for k in range(m):
            S = 0 if k == 0 else torch.stack(exemplar_feature).sum(0)
            phi = flatten_features
            mu = class_mean
            mu_p = ((phi + S) / (k + 1)).cpu().numpy()
            mu_p = mu_p / np.linalg.norm(mu_p)
            distances = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
            # Evito che si creino duplicati
            sorted_indexes = np.argsort(distances)
            for i in sorted_indexes:
                if indexes[i] not in exemplars:
                    exemplars.add(indexes[i])
                    exemplar_feature.append(flatten_features[i])
                    break

        assert len(exemplars) == m
        self.exemplar_sets.append(list(exemplars))

    def random_construct_exemplar_set(self, indexes, label, m):
        choices = np.arange(len(indexes))
        exemplars = np.random.choice(choices, m, replace=False)

        assert len(self.exemplar_sets) == label  # si assicura che l'inserimento avvenga nell'ordine corretto
        self.exemplar_sets.append(exemplars)

    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]:
        return self.net.parameters()

