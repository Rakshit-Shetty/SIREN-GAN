import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.fashion_mnist import MNIST, FashionMNIST
from utils.dataset import PreProcessDataset

def get_data_loader(args):

	if args.dataset == 'mnist':
		trans = transforms.Compose([
			transforms.Resize(32),
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,)),
		])
		train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
		test_dataset = MNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

	elif args.dataset == 'fashion-mnist':
		trans = transforms.Compose([
			transforms.Resize(32),
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,)),
		])
		train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
		test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

	elif args.dataset == 'cifar':
		trans = transforms.Compose([
			transforms.Resize(32),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])

		train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
		test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)

	elif args.dataset == 'stl10':
		trans = transforms.Compose([
			transforms.Resize(32),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		train_dataset = dset.STL10(root=args.dataroot, train=True, download=args.download, transform=trans)
		test_dataset = dset.STL10(root=args.dataroot, train=False, download=args.download, transform=trans)
	elif args.dataset == 'lsun':
		trans = transforms.Compose([
			transforms.Resize(32),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		train_dataset = dset.LSUN(root = args.dataroot, classes=['bedroom_train'], transform=trans)
		test_dataset = dset.LSUN(root = args.dataroot, classes=['bedroom_val'], transform=trans)
	elif args.dataset == 'imagenet':
		trans = transforms.Compose([
			transforms.Resize(32),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		train_dataset = dset.ImageFolder(root=args.dataroot,transfrom = trans)
		test_dataset = dset.ImageFolder(root=args.dataroot, transform = trans)
	elif args.dataset == 'custom':
		trans = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		train_dataset = PreProcessDataset(args.dataroot, train=True, transform=trans)
		test_dataset = PreProcessDataset(args.dataroot, train=False, transform=trans)
	# Check if everything is ok with loading datasets
	assert train_dataset
	assert test_dataset

	train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	test_dataloader  = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

	return train_dataloader, test_dataloader
