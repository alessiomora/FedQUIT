import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from can_bad_teaching.dataset import UnLearningData
import numpy as np
from can_bad_teaching.utils import *


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_weights = None

    def early_stop(self, validation_loss, state_dict):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_weights = state_dict

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def get_best_weights(self):
        return self.best_weights


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits,
                  KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(model, unlearning_teacher, full_trained_teacher,
                    unlearn_data_loader, optimizer,
                    device, KL_temperature):
    losses = []
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(output=output, labels=y,
                             full_teacher_logits=full_teacher_logits,
                             unlearn_teacher_logits=unlearn_teacher_logits,
                             KL_temperature=KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def unlearning_step_incorrect(model, unlearning_teacher,
                              unlearn_data_loader, optimizer,
                              device, KL_temperature):
    losses = []
    for batch in unlearn_data_loader:
        x, y, _ = batch
        x = x.to(device)
        with torch.no_grad():
            # full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

        student_out = F.log_softmax(output / KL_temperature, dim=1)
        loss = F.kl_div(student_out, u_teacher_out)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def unlearning_step_logit(model, global_model,
                          unlearn_data_loader, optimizer,
                          device, KL_temperature):
    losses = []
    for batch in unlearn_data_loader:
        x, y, z = batch
        x, y, z = x.to(device), y.to(device), z.to(device)
        with torch.no_grad():
            global_logits = global_model(x)
            # unlearn_teacher_logits = unlearning_teacher(x)

        output = model(x)
        optimizer.zero_grad()

        # u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

        index = torch.unsqueeze(z, 1)
        # print(index)
        src = torch.zeros_like(index, dtype=global_logits.dtype)
        # print(src)
        global_logits = global_logits.scatter_(1, index, src)
        global_out = F.softmax(global_logits / KL_temperature, dim=1)
        student_out = F.log_softmax(output / KL_temperature, dim=1)
        loss = F.kl_div(student_out, global_out)

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def unlearning_step_softmax(model, global_model,
                          unlearn_data_loader, optimizer,
                          device, KL_temperature, num_classes=20):
    losses = []
    for batch in unlearn_data_loader:
        x, y, z = batch
        x, y, z = x.to(device), y.to(device), z.to(device)
        with torch.no_grad():
            global_logits = global_model(x)
            # unlearn_teacher_logits = unlearning_teacher(x)

        output = model(x)
        optimizer.zero_grad()

        global_out = F.softmax(global_logits / KL_temperature, dim=1)
        student_out = F.log_softmax(output / KL_temperature, dim=1)
        index = torch.unsqueeze(z, 1)
        # predicted_true_probability = torch.gather(1, index, global_out)
        # a = 1 / num_classes
        # delta = (predicted_true_probability - a) / (num_classes - 1)
        # global_out = global_out + delta
        # a_filled = torch.full(index.shape(), a)
        # global_out = global_out.scatter_(1, index, a_filled)

        predicted_true_probability = torch.gather(global_out, 1, index)
        a = 1 / num_classes
        delta = (predicted_true_probability - a) / (num_classes - 1)
        global_out = global_out + delta
        a_filled = torch.full(index.shape, a)
        global_out = global_out.scatter_(1, index.to(device), a_filled.to(device))

        loss = F.kl_div(student_out, global_out)

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def fit_one_unlearning_cycle(epochs, model, train_loader, val_loader, lr, device):
    history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            loss.backward()
            train_losses.append(loss.detach().cpu())

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history


def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data,
                        forget_data, epochs=10,
                        optimizer='adam', lr=0.01, batch_size=256, num_workers=32,
                        device='cuda', KL_temperature=1):
    # creating the unlearning dataset.
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    optimizer = optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # if optimizer is not a valid string, then assuming it as a function to return optimizer
        optimizer = optimizer  # (model.parameters())

    for epoch in range(epochs):
        loss = unlearning_step(model=model, unlearning_teacher=unlearning_teacher,
                               full_trained_teacher=full_trained_teacher,
                               unlearn_data_loader=unlearning_loader,
                               optimizer=optimizer, device=device,
                               KL_temperature=KL_temperature)
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))


def logit_unlearner(model, global_model,
                    forget_data, epochs=10,
                    lr=0.01, batch_size=256, num_workers=32,
                    device='cuda', KL_temperature=1, variable_lr=False):
    # creating the unlearning dataset.
    forget_train_dl = DataLoader(forget_data, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=True,
                                 pin_memory=False)

    global_model.eval()

    if variable_lr:
        try:
            param_groups = [
                {'params': model.base.parameters(), 'lr': 0.0001},
                {'params': model.final.parameters(), 'lr': 0.001}
            ]
            optimizer = torch.optim.Adam(param_groups)
        except:
            print("---- Error ----")
            optimizer = torch.optim.Adam(model.parameters(), 0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopper(patience=3, min_delta=0)
    for epoch in range(epochs):
        loss = unlearning_step_logit(model=model, global_model=global_model,
                                     unlearn_data_loader=forget_train_dl,
                                     optimizer=optimizer, device=device,
                                     KL_temperature=KL_temperature)
        if early_stopper.early_stop(loss, model.state_dict()):
            break

        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))

    model.load_state_dict(early_stopper.get_best_weights())


def softmax_unlearner(model, global_model,
                    forget_data, epochs=10,
                    lr=0.00, batch_size=256, num_workers=32,
                    device='cuda', KL_temperature=1, variable_lr=False):
    # creating the unlearning dataset.
    forget_train_dl = DataLoader(forget_data, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=True,
                                 pin_memory=False)

    global_model.eval()

    if variable_lr:
        try:
            param_groups = [
                {'params': model.base.parameters(), 'lr': 0.0001},
                {'params': model.final.parameters(), 'lr': 0.001}
            ]
            optimizer = torch.optim.Adam(param_groups)
        except:
            print("---- Error ----")
            optimizer = torch.optim.Adam(model.parameters(), 0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    early_stopper = EarlyStopper(patience=3, min_delta=0)
    for epoch in range(epochs):
        loss = unlearning_step_softmax(model=model, global_model=global_model,
                                     unlearn_data_loader=forget_train_dl,
                                     optimizer=optimizer, device=device,
                                     KL_temperature=KL_temperature)
        if early_stopper.early_stop(loss, model.state_dict()):
            break

        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))


def incorrect_unlearner(model, unlearning_teacher,
                        forget_data, epochs=10,
                        optimizer='adam', lr=0.01, batch_size=256, num_workers=32,
                        device='cuda', KL_temperature=1, variable_lr=False):
    # creating the unlearning dataset.
    # unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    forget_train_dl = DataLoader(forget_data, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=True,
                                 pin_memory=False)

    unlearning_teacher.eval()

    if variable_lr:
        try:
            param_groups = [
                {'params': model.base.parameters(), 'lr': 0.0001},
                {'params': model.final.parameters(), 'lr': 0.001}
            ]
            optimizer = torch.optim.Adam(param_groups)
        except:
            print("---- Error ----")
            optimizer = torch.optim.Adam(model.parameters(), 0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopper(patience=3, min_delta=0)

    for epoch in range(epochs):
        loss = unlearning_step_incorrect(model=model,
                                         unlearning_teacher=unlearning_teacher,
                                         unlearn_data_loader=forget_train_dl,
                                         optimizer=optimizer, device=device,
                                         KL_temperature=KL_temperature)
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))

        if early_stopper.early_stop(loss, model.state_dict()):
            break


class UNSIR_noise(torch.nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


def UNSIR_noise_train(noise, model, forget_class_label, num_epochs, noise_batch_size,
                      device='cuda'):
    opt = torch.optim.Adam(noise.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        total_loss = []
        inputs = noise()
        labels = torch.zeros(noise_batch_size).to(device) + forget_class_label
        outputs = model(inputs)
        loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(
            torch.sum(inputs ** 2, [1, 2, 3]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss.append(loss.cpu().detach().numpy())
        if epoch % 5 == 0:
            print("Loss: {}".format(np.mean(total_loss)))

    return noise


def UNSIR_create_noisy_loader(noise, forget_class_label, retain_samples, batch_size,
                              num_noise_batches=80, device='cuda'):
    noisy_data = []
    for i in range(num_noise_batches):
        batch = noise()
        for i in range(batch[0].size(0)):
            noisy_data.append(
                (batch[i].detach().cpu(), torch.tensor(forget_class_label), \
                 torch.tensor(forget_class_label)))

    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][2]), \
             torch.tensor(retain_samples[i][2])))
    noisy_data += other_samples
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=batch_size,
                                               shuffle=True)

    return noisy_loader
