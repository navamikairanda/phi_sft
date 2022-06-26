import torch
def get_device():
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		torch.cuda.set_device(device)
	else:
		device = torch.device("cpu")
	return device	

