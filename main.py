from dataset import *
from model import *

if __name__ == "__main__":
    train_model("data", epochs=30, batch_size=16, lr_list=[1e-4, 5e-5], hidden_dims=[(512, 256), (256, 128)])
