from .models import *
from .training_testing import *
from .regularization import *
from .data_engine import *
from .utility import *
from .grad_cam import *
from .torch_lr_finder import *

class NetworkPipeline():
    def __init__(self, data_path, inp_size, seed, means, stdevs, need_albumentation, batch_size, labels_list, model_name, criterion, optimizer_name, scheduler_name=None, dropout=0, num_classes=10, train_loader=None, test_loader=None, is_custom=False, test_data_path=None):
      self.available_models = {"Net": Net, "Quiz9_DNN_Net": Quiz9_DNN_Net, "ResNet18": ResNet18, "ResNet34": ResNet34, "ResNet50": ResNet50, "ResNet101": ResNet101, "ResNet152": ResNet152, "CustomResNet": CustomResNet, "CustomResidualNet": CustomResidualNet}
      self.available_optimizers = {"SGD": optim.SGD, "Adam": optim.Adam}
      self.available_schedulers = {"OneCycleLR": torch.optim.lr_scheduler.OneCycleLR}
      self.seed = seed
      self.device = set_device()
      self.batch_size = batch_size
      self.labels_list = labels_list
      self.num_classes = num_classes
      print("\n Generating train and test loaders.....")
      if train_loader is None and test_loader is None:
        self.train_loader, self.test_loader = generate_train_test_loader(data_path, self.seed, means, stdevs, need_albumentation, batch_size=self.batch_size, is_custom=is_custom, test_data_path=test_data_path)
      else:
        if train_loader is not None:
          self.train_loader = train_loader
        if test_loader is not None:
          self.test_loader = test_loader
      self.model_name = model_name
      self.dropout = dropout
      if self.model_name not in ("Net", "Quiz9_DNN_Net", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"):
        self.model = self.available_models[self.model_name]().to(self.device)
      elif self.model_name in ("ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"):
        self.model = self.available_models[self.model_name](self.num_classes).to(self.device)
      else:
        self.model = self.available_models[self.model_name](self.dropout).to(self.device)
      self.criterion = criterion
      self.optimizer_name = self.available_optimizers[optimizer_name]
      if scheduler_name is not None and scheduler_name in self.available_schedulers.keys():
        self.scheduler_name = self.available_schedulers[scheduler_name]
      else:
        self.scheduler_name = None
      self.optimizer = None
      self.scheduler = None
      self.train_losses = []
      self.test_losses = []
      self.train_acc = []
      self.test_acc = []
      self.num_of_batches = len(self.train_loader)
      self.inp_size = inp_size
      #show_sample_images(self.train_loader, self.labels_list)

    def find_network_lr(self, init_lr, init_weight_decay, end_lr=1, num_epochs=100):
      self.model = self.model.to(self.device)
      if self.optimizer_name == optim.SGD:
        print(f"Finding max LR for One Cycle Policy using LR Test Range over {num_epochs} epochs...")
        lr_range_test_optimizer = optim.SGD(self.model.parameters(), lr=init_lr, weight_decay=init_weight_decay)
        lr_finder = LRFinder(self.model, lr_range_test_optimizer, self.criterion, device=self.device)
        lr_finder.range_test_over_epochs(self.train_loader, end_lr=end_lr, num_epochs=num_epochs)
        max_val_index = lr_finder.history['loss'].index(lr_finder.best_acc)
        best_lr = lr_finder.history['lr'][max_val_index]
        print(f"LR (max accuracy {lr_finder.best_acc}) to be used: {best_lr}")
        
        lr_finder.plot(show_lr=best_lr, yaxis_label="Training Accuracy") # to inspect the accuracy-learning rate graph
        lr_finder.reset() # to reset the self.model and optimizer to their initial state
        
        return best_lr
      else:
        raise Exception("Defined only for SGD Optimizer")

    def build_network(self, optim_params, scheduler_params=None):
        print("Creating model...")
        print("\n Model Summary:")
        self.model = self.model.to(self.device)
        summary(self.model, input_size=self.inp_size)
		
        self.optimizer = self.optimizer_name(self.model.parameters(), **optim_params)##optim.SGD(self.model.parameters(), lr=LRMIN, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        if self.scheduler_name is not None:
          self.scheduler = self.scheduler_name(self.optimizer, **scheduler_params)
		#torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LRMAX, steps_per_epoch=NUM_OF_BATCHES, epochs=EPOCHS, pct_start=PCT_START, anneal_strategy="linear")#, div_factor=DIV_FACTOR, final_div_factor=FINAL_DIV_FACTOR)
    
    def train_network(self, epochs, is_ocp=True):
        print("\n Training the model...")
        self.model = self.model.to(self.device)
        for epoch in range(epochs):
            print("EPOCH:", epoch+1)
            #print("Learning Rate:", get_lr(self.optimizer))
            train(self.model, self.device, self.train_loader, self.optimizer, self.criterion, epoch, self.train_losses, self.train_acc, is_ocp=is_ocp, scheduler=self.scheduler)
            test(self.model, self.device, self.criterion, self.test_loader, self.test_losses, self.test_acc)
        print("\n Model training completed...")

    def save_network(self, path, model_file_name):
        print("\n Saving trained model and parameters...")
        torch.save(self.model.state_dict(), path+"/"+model_file_name+".pth")
        # save train and test losses and accuracies
        train_test_data = {"Training Loss": self.train_losses, "Test Loss": self.test_losses, "Training Accuracy": self.train_acc, "Test Accuracy": self.test_acc}
        torch.save(train_test_data, path+"/"+model_file_name+"_train_test_params.pt")

    def load_network(self, path, model_file_name):
        print("\n Loading trained model...")
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(path+"/"+model_file_name+".pth"))