import copy
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from Problems.CNOBenchmarks import Darcy, Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, \
    WaveEquation, ShearLayer, ApeNS, KS, Bruss


training_properties = {
    "learning_rate": 0.001,
    "weight_decay": 1e-6,
    "scheduler_step": 10,
    "scheduler_gamma": 0.98,
    "epochs": 500,
    "batch_size": 16,
    "exp": 1,                # Do we use L1 or L2 errors? Default: L1
    "training_samples": 16  # How many training samples?
}

model_architecture_ = {

    #Parameters to be chosen with model selection:
    "N_layers": 3,            # Number of (D) & (U) blocks
    "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
    "N_res": 4,               # Number of (R) blocks in the middle networs.
    "N_res_neck" : 6,         # Number of (R) blocks in the BN

    #Other parameters:
    "in_size": 128,            # Resolution of the computational grid
    "retrain": 2,             # Random seed
    "kernel_size": 3,         # Kernel size.
    "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
    "activation": 'cno_lrelu_torch',# cno_lrelu or cno_lrelu_torch or lrelu or

    #Filter properties:
    "cutoff_den": 2.0001,     # Cutoff parameter.
    "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
    "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
    "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
    "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
}

#   "which_example" can be

#   poisson             : Poisson equation
#   wave_0_5            : Wave equation
#   cont_tran           : Smooth Transport
#   disc_tran           : Discontinuous Transport
#   allen               : Allen-Cahn equation
#   shear_layer         : Navier-Stokes equations
#   airfoil             : Compressible Euler equations
#   darcy               : Darcy Flow
#   apens               : Nabier-Stokes equations from APEBench
#   ks                  : Kuramoto-Sivashinsky Equation
#   bruss               : Brusselator Diffusion-Reaction System

which_example = "apens"

# Save the models here:
folder = "TestModels1/"+"CNO_transfer_"+which_example

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')


if which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, training_samples, size=64)
    example_nlt = ShearLayer(model_architecture_, device, batch_size, training_samples, size=128, nlt=False, in_dist=False)
elif which_example == "poisson":
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
    example_nlt = SinFrequency(model_architecture_, device, batch_size, training_samples, nlt=True)
elif which_example == "wave_0_5":
    example = WaveEquation(model_architecture_, device, batch_size, training_samples)
    example_nlt = WaveEquation(model_architecture_, device, batch_size, training_samples, in_dist=False, nlt=False)
elif which_example == "allen":
    example = AllenCahn(model_architecture_, device, batch_size, training_samples)
    example_nlt = AllenCahn(model_architecture_, device, batch_size, training_samples, nlt=True)
elif which_example == "cont_tran":
    example = ContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "disc_tran":
    example = DiscContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "airfoil":
    model_architecture_["in_size"] = 128
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
elif which_example == "darcy":
    example = Darcy(model_architecture_, device, batch_size, training_samples)
elif which_example == 'apens':
    example = ApeNS(model_architecture_, device, batch_size, training_samples)
    # in_dist: if True, use in-distribution data; else use out-of-distribution
    # nlt: if True, use transfer learning
    example_nlt = ApeNS(model_architecture_, device, batch_size, training_samples, in_dist=False, nlt=False)
elif which_example == 'ks':
    example = KS(model_architecture_, device, batch_size, 256)
    example_nlt = KS(model_architecture_, device, batch_size, training_samples, in_dist=False, nlt=False)
elif which_example == 'bruss':
    example = KS(model_architecture_, device, batch_size, 256)
    example_nlt = Bruss(model_architecture_, device, batch_size, training_samples, in_dist=False, nlt=False)
else:
    raise ValueError()

#-----------------------------------Train--------------------------------------

#-----------------------target model-------------------------------------------
tar_model = example_nlt.model
sou_model = torch.load(f'TrainedModels5/CNO_apens_5_512/model.pkl')
tar_model.load_state_dict(sou_model.state_dict())
# tar_model.load_state_dict(torch.load(f"TrainedModels5/CNO_shear_layer_512/model.pt"), strict=False)

tar_train_loader = example_nlt.train_loader
tar_val_loader = example_nlt.val_loader
tar_test_loader = example_nlt.test_loader


# freeze all parameter
for name, param in tar_model.named_parameters():
    param.requires_grad = False
#
# fine-tune the specific layers
for k, v in tar_model.named_parameters():
    if 'decoder.2' in k or 'decoder_inv.3' in k:
        v.requires_grad = True
# print layers need update
for k, v in tar_model.named_parameters():
    if v.requires_grad:
        print(k)

tar_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, tar_model.parameters()), lr=learning_rate, weight_decay=weight_decay)
tar_scheduler = torch.optim.lr_scheduler.StepLR(tar_optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tar_model = tar_model.cuda()

freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()

best_model_testing_error = 1000  # Save the model once it has less than 1000% relative L1 error
patience = int(0.2 * epochs)  # Early stopping parameter
counter = 0

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")

for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:

        # sou_model.train()
        tar_model.train()

        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        target_mse = 0.0
        train_mse_t = 0.0
        running_relative_train_mse = 0.0

        #---------------------------------finetune----------------------------------------------
        for i, (input_t, output_t) in enumerate(tar_train_loader):
            input_t = input_t.to(device)
            output_t = output_t.to(device)
            pred_t = tar_model(input_t)

            mse_t = loss(pred_t, output_t) / loss(torch.zeros_like(output_t).to(device), output_t)

            tar_optimizer.zero_grad()
            mse_t.backward()
            tar_optimizer.step()

            train_mse_t = train_mse_t * i / (i + 1) + mse_t.item() / (i + 1)
            #         tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})
            tepoch.set_postfix({'Batch': i + 1, 'Finetune loss (in progress)': train_mse_t})
        writer.add_scalar("train_loss/train_loss", train_mse_t, epoch)
        # writer.add_scalar("target_loss/target_loss", target_mse, epoch)


        with torch.no_grad():
            tar_model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0

            for step, (input_batch, output_batch) in enumerate(tar_val_loader):

                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = tar_model(input_batch)

                if which_example == "airfoil":  # Mask the airfoil shape
                    output_pred_batch[input_batch == 1] = 1
                    output_batch[input_batch == 1] = 1

                loss_val = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_val.item()
            test_relative_l2 /= len(tar_val_loader)

            for step, (input_batch, output_batch) in enumerate(tar_train_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = tar_model(input_batch)

                if which_example == "airfoil":  # Mask the airfoil shape
                    output_pred_batch[input_batch == 1] = 1
                    output_batch[input_batch == 1] = 1

                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(tar_train_loader)

            writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
            writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(tar_model)
                # torch.save(best_model, folder + "/model.pkl")
                torch.save(best_model.state_dict(), folder + "/model.pt")
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter += 1

        tepoch.set_postfix(
            {'Train loss': train_mse_t, 'Target loss': target_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Target Error: " + str(target_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
        # file.write("Params: " + str(n_params) + "\n")
    tar_scheduler.step()

    if counter > patience:
        print("Early Stopping")
        break

