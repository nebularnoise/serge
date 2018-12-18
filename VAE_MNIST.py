import argparse
from src import arch, dataset
from src.train.base_VAE import train_model
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
parser.add_argument("--cuda", type=int, default=-1, help="Define wich GPU to use")
parser.add_argument("--rec-loss", type=str, default="BCE", help="Define what\
                    reconstruction loss estimation to use")
parser.add_argument("--mode", type=str, default="V", help="Define the Latent\
                    Space regularization method (V or W)")
parser.add_argument("--lr", type=int, default=3, help="Define how slow the\
                    learning is 10^(1-5)")
parser.add_argument("--name", type=str, default="untitled", help="Name of the\
                    output file")
args = parser.parse_args()

print("\033[01mCreating a VAE with the following specs:\033[0m")
print("   - Latent space has %d dimensions" % args.zdim)
print("   - Encoder is %s." % ("determinist" if args.deter else "stochastic"))
print("   - Training is made on %d epochs." % args.epoch)

device = torch.device("cuda:%d" % args.cuda if args.cuda >= 0 else "cpu")

model = arch.VariationnalAutoEncoder(determinist_encoder=args.deter,
                                    z_dim=args.zdim)

model = model.to(device)

#INITIALIZING DATA

for param in model.parameters():
    torch.init.xavier_normal_(param.data)

print("   - Device used is %s." % str(device))

print("\033[01mLoading specified dataset\033[0m")
print("     Loading %s..." % args.dataset)
train_loader, test_loader = dataset.load_MNIST(args.dataset, args.batch)
print("     Done!")


print("\033[01mTraining initialization")

if args.rec_loss == "BCE":
    rec_loss_f = torch.nn.functional.binary_cross_entropy

print("Training...\033[0m")

statut = train_model(model, device, train_loader, test_loader,
                    args.epoch, rec_loss_f, args.name,
                    mode=args.mode, lr=10**(-args.lr))


if statut:
    print("\033[01mEverything went okay, exiting...\033[0m")
else:
    print("\033[01mSomething went wrong, exiting...\033[0m")
