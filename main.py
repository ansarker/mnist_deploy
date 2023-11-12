import argparse
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from model import TestNet
from dataloader import MnistFashionDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def imshow(img, labels):
    img = img/2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(labels)
    plt.show()

def visualize_results(images, labels, predictions):
	num_samples = len(images)
	
	fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
	
	for i in range(num_samples):
		axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
		axes[i].set_title(f'GT: {labels[i]}, Pred: {predictions[i]}')
		axes[i].axis('off')
	plt.savefig(os.path.join('./runs', 'sample_results.png'))
	plt.show()

def train(model, train_loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	
	for i, (images, labels) in enumerate(train_loader, 0):
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
	
	return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
	model.eval()
	correct = 0
	total = 0
	running_loss = 0.0

	with torch.no_grad():
		for i, (images, labels) in enumerate(val_loader):
			images = images.to(device)
			labels = labels.to(device)
			
			outputs = model(images)
			loss = criterion(outputs, labels)
			running_loss += loss.item()

			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	
	accuracy = correct / total
	average_loss = running_loss / len(val_loader)

	return accuracy, average_loss

def test(model, test_loader, device, classes):
	model.eval()
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for i, (images, labels) in enumerate(test_loader):
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			_, preds = torch.max(outputs, 1)

			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())
	
	cm = confusion_matrix(all_labels, all_preds)
	
	plt.figure(figsize=(12, 10))
	sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
	plt.xlabel("Prediction")
	plt.ylabel("Ground truth")
	plt.title("Confusion matrix")
	# plt.show()
	plt.savefig("./runs/confusion_matrix.png")

	# Calculate ROC curves for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	
	all_labels_array = np.array(all_labels)
	all_preds_array = np.array(all_preds)

	for i in range(len(classes)):
		fpr[i], tpr[i], _ = roc_curve((all_labels_array == i).astype(int), all_preds_array)
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Plot ROC curves
	plt.figure(figsize=(8, 6))
	for i in range(len(classes)):
		plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {classes[i]}, AUC = {roc_auc[i]:.2f})')

	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curves for Each Class')
	plt.legend(fontsize=7)
	# plt.show()
	plt.savefig('./runs/ROC_curve.png')

	accuracy = accuracy_score(all_labels, all_preds)

	precision = precision_score(all_labels, all_preds, average=None)
	recall = recall_score(all_labels, all_preds, average=None)
	f1 = f1_score(all_labels, all_preds, average=None)
	
	with open('./runs/scores.txt', 'a') as score_file:
		print(f'Accuracy: {accuracy:.4f}')
		score_file.write(f'Accuracy: {accuracy:.4f}\n')

		for i, class_name in enumerate(classes):
			text = f'Class: {class_name}\nPrecision: {precision[i]:.4f}\nRecall: {recall[i]:.4f}\nF1-score: {f1[i]:.4f}\n---\n'
			print(text)
			score_file.write(text)

def main(opt):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	print(f'Running on {device}')
	print("---" * 10)
	print(opt)
	print("---" * 10)

	if opt.train:
		train_data = MnistFashionDataset('fashion-mnist_train.csv')
		dataset_size = len(train_data)
		train_size = int(0.7 * dataset_size)
		val_size = dataset_size - train_size
		train_data, val_data = random_split(train_data, [train_size, val_size])

		train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		
		model = TestNet().to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
		print(model)

		train_losses = []
		val_losses = []
		val_accuracies = []

		os.makedirs('./runs', exist_ok=True)
		with open('./runs/loss_log.txt', 'a') as log_file:
			for epoch in range(opt.num_epochs):
				train_loss = train(model, train_loader, criterion, optimizer, device)
				val_accuracy, val_loss = validate(model, val_loader, criterion, device)

				train_losses.append(train_loss)
				val_losses.append(val_loss)
				val_accuracies.append(val_accuracy)

				print(f'Epoch {epoch+1}/{opt.num_epochs}, Train loss: {train_loss:.3f}, Test accuracy: {val_accuracy:.3f}, Test loss: {val_loss:.3f}')
				log_file.write(f'Epoch {epoch+1}/{opt.num_epochs}, Train loss: {train_loss:.3f}, Test accuracy: {val_accuracy:.3f}, Test loss: {val_loss:.3f}\n')
				
				os.makedirs(opt.checkpoints_dir, exist_ok=True)
				if (epoch+1) % 20 == 0:
					torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir, f'mnist-fashion-{epoch + 1}.pth'))
			
			sty = 'seaborn-v0_8'
			mpl.style.use(sty)
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.set_title(f'Learning curve')
			ax.plot(train_losses, 'C1', label='train loss')
			ax.plot(val_losses, 'C2', label='val loss')
			ax.plot(val_accuracies, 'C3', label='val accuracy')
			ax.legend()

			# plt.show()
			plt.savefig('./runs/training_loss_plot.png')
		print('Training finished!')
	else:
		test_data = MnistFashionDataset('fashion-mnist_test.csv')
		test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

		classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

		model = TestNet().to(device)
		model.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.checkpoints_name)))
		print(model)

		test(model, test_loader, device, classes)
		print("Test done!")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
	parser.add_argument('-j', '--num_workers', type=int, default=2, help='Number of workers')
	parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_false')
	parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='Checkpoints path')
	parser.add_argument('--checkpoints_name', type=str, default='mnist-fashion-200.pth', help='Checkpoints path')
	opt = parser.parse_args()

	main(opt)