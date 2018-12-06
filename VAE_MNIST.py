import argparse
from src import arch, dataset
import torch

parser = argparse.ArgumentParser(description="Utility for training a custom \
VAE based on input arguments")

parser.add_argument("--deter", action="store_true",
                    help="Defines wheither the encoder is determinist or not")
parser.add_argument("--zdim", type=int, default=32,
                    help="Dimension of the latent space")
parser.add_argument("--epoch", type=int, default=100,
                    help="Number of epochs.")
parser.add_argument("--dataset", type=str, default="MNIST",
                    help="Name of the dataset to be used")
parser.add_argument("--batch", type=int, default=128,
                    help="Size of the minibatch")
args = parser.parse_args()

print("\033[01mCreating a VAE with the following specs:\033[0m")
print("   - Latent space has %d dimensions" % args.zdim)
print("   - Encoder is %s." % ("determinist" if args.deter else "stochastic"))
print("   - Training is made on %d epochs." % args.epoch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = arch.VariationnalAutoEncoder(determinist_encoder=args.deter,
                                    z_dim=args.zdim)

model = model.to(device)

print("   - Device used is %s." % str(device))

print("\033[01mLoading specified dataset\033[0m")
print("     Loading %s..." % args.dataset)
train_loader, test_loader = dataset.load_MNIST(args.dataset, args.batch)
print("     Done!")
