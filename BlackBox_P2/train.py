import torch.optim as optim
import torch
from approximate_gradients import *
from resnet import *
import torch.nn.functional as FN
import random
from gan import *
import time
from movie_test import *
import os


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(student, generator, nz=256, device='cpu', epoch=0, epoch_itrs=501, g_iter=1, d_iter=5,
          batch_size=16):  # d_iter=5
    """Main Loop for one epoch of Training Generator and Student"""
    # global file
    # teacher.eval()
    student.train()

    optimizer_S = optim.SGD(student.parameters(), lr=1e-2, weight_decay=5e-2, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, weight_decay=5e-2)  # , betas = (0.5,0.999))
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, patience=5)
    scheduler_S = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_S, patience=5)
    gradients = []

    for i in range(epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(g_iter):
            # Sample Random Noise
            #
            # z = torch.randn((batch_size, 3, 1, 64, 64)).to(device)  # 64, 64 => W,H
            z = torch.randn((batch_size, nz)).to(device)
            optimizer_G.zero_grad()
            # generator.train()
            # Get fake image from generator
            fake_inputs = generator(z)  # .permute(0, 2, 1, 3, 4)
            # Fake should be (batch_sz,F,C,W,H)
            # print(f'fake ip shape {fake_inputs.shape}')

            # APPROX GRADIENT
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(student, x=fake_inputs.contiguous(), epsilon=1e-3,
                                                                    device=device)  # Anup changed eps from 1e-3 to 1e-7

            # permuted_grad = approx_grad_wrt_x.permute(0, 2, 1, 3, 4)
            # exit(0)
            # print(f"Approx Grad{approx_grad_wrt_x.shape}, FakeInput{fake_inputs.shape}")
            # exit(0)
            fake_inputs.backward(approx_grad_wrt_x)
            optimizer_G.step()

            # if i == 0 and args.rec_grad_norm:  ##(Only for logging)
            #   x_true_grad = measure_true_grad_norm(args, fake)

        for _ in range(d_iter):
            # z = torch.randn((batch_size, 3, 1, 64, 64)).to(device)
            z = torch.randn((batch_size, nz)).to(device)
            fake_inputs = generator(z).detach().permute(0, 2, 1, 3, 4)
            optimizer_S.zero_grad()

            with torch.no_grad():
                teacher_label = get_probabs(fake_inputs.detach().cpu(), device=device)

            s_logit = student(fake_inputs.permute((0, 2, 1, 3, 4)))

            criterion = FN.cross_entropy
            # criterion = torch.nn.CrossEntropyLoss()
            loss_S = criterion(s_logit, teacher_label, reduction="mean")
            loss_S.backward()
            # print(loss_S.item())
            optimizer_S.step()

        print(f"Iteration {i} ---- G_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f}")
        scheduler_G.step(loss_G)
        scheduler_S.step(loss_S)
        # print(f"SwinT preds = {teacher_labels}")

        if i % 50 == 0:
            torch.save(student, "checkpoint/BlackBox_P2_" + str(i) + "_" + str(epoch) + ".pt")
            torch.save(generator, "checkpoint/generator_P2_" + str(i) + "_" + str(epoch) + ".pt")


device = "cuda"
nz = 256
student = get_vid_resnet(num_classes=600).to(device=device)
generator = GeneratorA(nz=nz, activation=torch.tanh).to(device=device)  #
# generator = Generator().to(device=device)

if not os.path.exists('checkpoint/'):
    os.mkdir('checkpoint/')

for epoch in range(1):
    print(f"------ EPOCH {epoch} -------")
    train(student, generator, 256, device=device, epoch=epoch)
