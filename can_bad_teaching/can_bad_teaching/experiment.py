import torch
import hydra
from can_bad_teaching.dataset import *
from can_bad_teaching.mia_svc import SVC_MIA
from can_bad_teaching.model import ResNet18, ViT
from can_bad_teaching.unlearn import *
from can_bad_teaching.metrics import UnLearningScore, get_membership_attack_prob
from can_bad_teaching.utils import *
from torch.utils.data import DataLoader
import random
import pandas as pd
import os
from omegaconf import DictConfig, OmegaConf


def get_model(model_architecture="resnet", device=None):
    if model_architecture not in ["vit"]:
        model = ResNet18(num_classes=20, pretrained=True).to(device)
    else:
        model = ViT(num_classes=20).to(device)
    return model


@hydra.main(config_path="conf", config_name="unlearning", version_base=None)
def main(cfg: DictConfig) -> None:
    experiments = [cfg.experiment]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_arch = "resnet"
    forget_classes_dict = {"rocket": [69],
                           "mushroom": [51],
                           "baby": [2],
                           "lamp": [40],
                           "sea": [71]}
    forget_classes_superclass_dict = {"rocket": [41, 81, 85, 89],
                                      "mushroom": [0, 53, 57, 83],
                                      "baby": [11, 35, 46, 98],
                                      "lamp": [39, 22, 87, 86],
                                      "sea": [49, 33, 23, 60]}
    # forget_valid = []
    # forget_classes = [69]
    # forget_superclasses = [41, 81, 85, 89]

    configs = [
        # {"variable_lr": True, "lr": 0.00, "epochs": 100},
        # {"variable_lr": False, "lr": 0.001, "epochs": 100},
        # {"variable_lr": False, "lr": 0.0001, "epochs": 100},
        {"variable_lr": False, "lr": 0.0005, "epochs": 100},
        # {"variable_lr": False, "lr": 0.01, "epochs": 100},
        # {"variable_lr": True, "lr": 0.00, "epochs": 10},
        # {"variable_lr": False, "lr": 0.001, "epochs": 10},
        # {"variable_lr": False, "lr": 0.0001, "epochs": 10},
        {"variable_lr": False, "lr": 0.0005, "epochs": 10},
        # {"variable_lr": False, "lr": 0.01, "epochs": 10}
    ]

    for experiment in experiments:
        print(f"[Experiment: {experiment}]")
        accuracy = []
        original = {}
        retrain = {}

        train_ds = CustomCIFAR100(root='.', train=True, download=True,
                                  transform=transform_train)
        valid_ds = CustomCIFAR100(root='.', train=False, download=True,
                                  transform=transform_train)

        batch_size = 256
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0,
                              pin_memory=False)
        valid_dl = DataLoader(valid_ds, batch_size, num_workers=0, pin_memory=False)

        num_classes = 100
        classwise_train = {}
        for i in range(num_classes):
            classwise_train[i] = []

        for img, label, clabel in train_ds:
            classwise_train[label].append((img, label, clabel))

        classwise_test = {}
        for i in range(num_classes):
            classwise_test[i] = []

        for img, label, clabel in valid_ds:
            classwise_test[label].append((img, label, clabel))

        device = device

        model = get_model(model_arch, device)
        epochs = 5
        history, last_lr = fit_one_cycle(epochs, model, train_dl, valid_dl,
                                         device=device)
        print("Learning rates: ", last_lr)
        torch.save(model.state_dict(),
                   "ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt")

        device = device
        # model = ResNet18(num_classes=20, pretrained=True).to(device)
        model = get_model(model_arch, device)
        model.load_state_dict(
            torch.load("ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
                       map_location=device))

        # Getting the forget and retain validation data
        forget_valid = []
        forget_classes = forget_classes_dict[experiment]
        forget_superclasses = forget_classes_superclass_dict[experiment]
        for cls in range(num_classes):
            if cls in forget_classes:
                for img, label, clabel in classwise_test[cls]:
                    forget_valid.append((img, label, clabel))

        retain_valid = []
        test_data = []
        for cls in range(num_classes):
            if cls not in forget_classes:
                for img, label, clabel in classwise_test[cls]:
                    retain_valid.append((img, label, clabel))
                    test_data.append((img, label, clabel))
            else:
                for img, label, clabel in classwise_test[cls]:
                    test_data.append((img, label, clabel))

        # Added
        retain_class_valid = []
        for cls in range(num_classes):
            if cls in forget_superclasses:
                for img, label, clabel in classwise_test[cls]:
                    retain_class_valid.append((img, label, clabel))

        forget_train = []
        for cls in range(num_classes):
            if cls in forget_classes:
                for img, label, clabel in classwise_train[cls]:
                    forget_train.append((img, label, clabel))

        retain_train = []
        for cls in range(num_classes):
            if cls not in forget_classes:
                for img, label, clabel in classwise_train[cls]:
                    retain_train.append((img, label, clabel))

        forget_valid_dl = DataLoader(forget_valid, batch_size, num_workers=0,
                                     pin_memory=False)
        retain_valid_dl = DataLoader(retain_valid, batch_size, num_workers=0,
                                     pin_memory=False)
        retain_class_valid_dl = DataLoader(retain_class_valid, batch_size,
                                           num_workers=0, pin_memory=False)

        forget_train_dl = DataLoader(forget_train, batch_size, num_workers=0,
                                     pin_memory=False)
        retain_train_dl = DataLoader(retain_train, batch_size, num_workers=0,
                                     pin_memory=False,
                                     shuffle=True)

        retain_train_subset = random.sample(retain_train, int(0.3 * len(retain_train)))
        # retain_train_subset_dl = DataLoader(retain_train_subset, batch_size, num_workers=0,
        #                                     pin_memory=False, shuffle=True)

        # Performance of Fully trained model on retain set
        results = evaluate(model, retain_valid_dl, device)
        original["dr"] = results["Acc"]

        # Performance of Fully trained model on retain class set
        results = evaluate(model, retain_class_valid_dl, device)
        original["dr_class"] = results["Acc"]

        # Performance of Fully trained model on forget set
        results = evaluate(model, forget_valid_dl, device)
        original["df"] = results["Acc"]

        original["name"] = "original"

        # MIA
        # balanced datasets
        retain_train_subset_mia = random.sample(retain_train, len(test_data))
        retain_train_mia_dl = DataLoader(retain_train_subset_mia, batch_size,
                                         num_workers=0,
                                         pin_memory=False)
        test_train_mia_dl = DataLoader(test_data, batch_size, num_workers=0,
                                       pin_memory=False)

        # results_mia = get_membership_attack_prob(
        #     retain_loader=retain_train_mia_dl,
        #     forget_loader=test_train_mia_dl,
        #     test_loader=forget_train_dl,
        #     model=model)
        # print("Results MIA")
        # print(results_mia)
        # original["mia_entropy"] = results_mia
        results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                              shadow_test=test_train_mia_dl,
                              target_train=forget_train_dl,
                              target_test=None,
                              model=model)
        print("Results MIA confidence")
        print(results_mia)
        original["mia_confidence"] = results_mia

        for seed in range(0, 1):
            # Retrain the model from Scratch
            print("Retrain model from scratch")
            retrain = {}
            device = device
            # gold_model = ResNet18(num_classes=20, pretrained=True).to(device)
            gold_model = get_model(model_arch, device)
            history = fit_one_cycle(epochs, gold_model, retain_train_dl,
                                    retain_valid_dl,
                                    device=device)

            torch.save(gold_model.state_dict(),
                       "ResNET18_CIFAR100Super20_Pretrained_Gold_Class69_5_Epochs.pt")
            device = device
            # gold_model = ResNet18(num_classes=20, pretrained=True).to(device)
            gold_model = get_model(model_arch, device)
            gold_model.load_state_dict(
                torch.load(
                    "ResNET18_CIFAR100Super20_Pretrained_Gold_Class69_5_Epochs.pt",
                    map_location=device))

            # evaluate gold model on retain set
            results = evaluate(gold_model, retain_valid_dl, device)
            retrain["dr"] = results["Acc"]

            # Performance of Fully trained model on retain class set
            results = evaluate(gold_model, retain_class_valid_dl, device)
            # loss.append(["Loss"]
            retrain["dr_class"] = results["Acc"]

            # evaluate gold model on forget set
            results = evaluate(gold_model, forget_valid_dl, device)
            retrain["df"] = results["Acc"]
            retrain["name"] = "retrain"

            # results_mia = get_membership_attack_prob(
            #     retain_loader=retain_train_mia_dl,
            #     forget_loader=test_train_mia_dl,
            #     test_loader=forget_train_dl,
            #     model=gold_model)
            # print("Results MIA")
            # print(results_mia)
            # retrain["mia_entropy"] = results_mia

            results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                  shadow_test=test_train_mia_dl,
                                  target_train=forget_train_dl,
                                  target_test=None,
                                  model=gold_model)
            print("Results MIA confidence")
            print(results_mia)
            retrain["mia_confidence"] = results_mia

        # UnLearning via proposed method
        for seed in range(0, 3):
            print("[Paper] Unlearning..")
            unlearned = {}
            device = device
            # unlearning_teacher = ResNet18(num_classes=20, pretrained=False).to(device).eval()
            unlearning_teacher = get_model(model_arch, device).eval()
            # student_model = ResNet18(num_classes=20, pretrained=False).to(device)
            student_model = get_model(model_arch, device)

            student_model.load_state_dict(
                torch.load(
                    "ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
                    map_location=device))
            model = model.eval()

            KL_temperature = 1

            optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)

            blindspot_unlearner(model=student_model,
                                unlearning_teacher=unlearning_teacher,
                                full_trained_teacher=model,
                                retain_data=retain_train_subset,
                                forget_data=forget_train, epochs=1,
                                optimizer=optimizer, lr=0.0001,
                                batch_size=256, num_workers=0, device=device,
                                KL_temperature=KL_temperature)

            # performance of unlearned model on retain set
            results = evaluate(student_model, retain_valid_dl, device)
            unlearned["dr"] = results["Acc"]
            # Performance of Fully trained model on retain class set
            results = evaluate(student_model, retain_class_valid_dl, device)
            # loss.append(["Loss"]
            unlearned["dr_class"] = results["Acc"]
            # performance of unlearned model on forget set
            results = evaluate(student_model, forget_valid_dl, device)
            unlearned["df"] = results["Acc"]
            unlearned["name"] = "unlearned"

            original_score = UnLearningScore(model, unlearning_teacher, forget_valid_dl,
                                             256, device)
            print("Initial Score: {}".format(original_score))
            retrain_score = UnLearningScore(gold_model, unlearning_teacher,
                                            forget_valid_dl, 256,
                                            device)
            print("Gold Score: {}".format(retrain_score))

            unlearned_score = UnLearningScore(student_model, unlearning_teacher,
                                              forget_valid_dl, 256,
                                              device)
            print("IC Score: {}".format(unlearned_score))
            js_div_unlearned = 1 - UnLearningScore(gold_model, student_model,
                                                   forget_valid_dl, 256,
                                                   device)
            print("JS Div: {}".format(js_div_unlearned))

            unlearned["zrf"] = unlearned_score.item()
            unlearned["js_div"] = js_div_unlearned.item()

            original["zrf"] = original_score.item()

            retrain["zrf"] = retrain_score.item()

            # results_mia = get_membership_attack_prob(
            #     retain_loader=retain_train_mia_dl,
            #     forget_loader=test_train_mia_dl,
            #     test_loader=forget_train_dl,
            #     model=student_model)
            # print("Results MIA")
            # print(results_mia)
            # unlearned["mia_entropy"] = results_mia

            results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                  shadow_test=test_train_mia_dl,
                                  target_train=forget_train_dl,
                                  target_test=None,
                                  model=student_model)
            print("Results MIA confidence")
            print(results_mia)
            unlearned["mia_confidence"] = results_mia

            accuracy.append(original)
            accuracy.append(retrain)
            accuracy.append(unlearned)

            # After 1 epoch
            for ep in range(1, 4):
                unl_resumed = {}
                # history = fit_one_cycle(1, student_model, retain_train_subset_dl, valid_dl,
                #                         device=device)
                history = fit_one_cycle(1, student_model, retain_train_dl, retain_valid_dl,
                                        device=device)
                results = evaluate(student_model, retain_valid_dl, device)
                unl_resumed["dr"] = results["Acc"]

                # Performance of Fully trained model on retain class set
                results = evaluate(student_model, retain_class_valid_dl, device)
                # loss.append(["Loss"]
                unl_resumed["dr_class"] = results["Acc"]

                # performance of unlearned model on forget set
                results = evaluate(student_model, forget_valid_dl, device)
                unl_resumed["df"] = results["Acc"]
                unl_resumed["name"] = f"unl_resumed_{ep}"

                logit_score = UnLearningScore(student_model, unlearning_teacher,
                                              forget_valid_dl, 256,
                                              device)
                print("IC Score: {}".format(logit_score))
                js_div_logit = 1 - UnLearningScore(gold_model, student_model,
                                                   forget_valid_dl, 256,
                                                   device)
                print("JS Div: {}".format(js_div_logit))

                unl_resumed["zrf"] = logit_score.item()
                unl_resumed["js_div"] = js_div_logit.item()

                # results_mia = get_membership_attack_prob(
                #     retain_loader=retain_train_mia_dl,
                #     forget_loader=test_train_mia_dl,
                #     test_loader=forget_train_dl,
                #     model=student_model)
                # print("Results MIA")
                # print(results_mia)
                # unl_resumed["mia_entropy"] = results_mia

                results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                      shadow_test=test_train_mia_dl,
                                      target_train=forget_train_dl,
                                      target_test=None,
                                      model=student_model)
                print("Results MIA confidence")
                print(results_mia)
                unl_resumed["mia_confidence"] = results_mia

                accuracy.append(unl_resumed)

        for config in configs:
            variable_lr = config["variable_lr"]
            lr = config["lr"]
            epochs_unlearning = config["epochs"]
            config_string = f"{variable_lr}_{lr}_{epochs_unlearning}"
            for seed in range(0, 3):
                # UnLearning via incorrect teacher
                print("[Incorrect] Unlearning..")
                incorrect = {}
                device = device
                # unlearning_teacher = ResNet18(num_classes=20, pretrained=False).to(device).eval()
                unlearning_teacher = get_model(model_arch, device).eval()
                # student_model = ResNet18(num_classes=20, pretrained=False).to(device)
                student_model = get_model(model_arch, device)
                student_model.load_state_dict(
                    torch.load(
                        "ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
                        map_location=device))
                model = model.eval()

                KL_temperature = 1

                incorrect_unlearner(model=student_model,
                                    unlearning_teacher=unlearning_teacher,
                                    forget_data=forget_train, epochs=epochs_unlearning,
                                    lr=lr, batch_size=256, num_workers=0, device=device,
                                    KL_temperature=KL_temperature,
                                    variable_lr=variable_lr)

                # performance of unlearned model on retain set
                results = evaluate(student_model, retain_valid_dl, device)
                incorrect["dr"] = results["Acc"]

                # Performance of Fully trained model on retain class set
                results = evaluate(student_model, retain_class_valid_dl, device)
                # loss.append(["Loss"]
                incorrect["dr_class"] = results["Acc"]

                # performance of unlearned model on forget set
                results = evaluate(student_model, forget_valid_dl, device)
                incorrect["df"] = results["Acc"]
                incorrect["name"] = f"incorrect_{config_string}"

                incorrect_score = UnLearningScore(student_model, unlearning_teacher,
                                                  forget_valid_dl, 256,
                                                  device)
                print("IC Score: {}".format(incorrect_score))
                js_div_incorrect = 1 - UnLearningScore(gold_model, student_model,
                                                       forget_valid_dl, 256,
                                                       device)
                print("JS Div: {}".format(js_div_incorrect))

                incorrect["zrf"] = incorrect_score.item()
                incorrect["js_div"] = js_div_incorrect.item()

                # results_mia = get_membership_attack_prob(
                #     retain_loader=retain_train_mia_dl,
                #     forget_loader=test_train_mia_dl,
                #     test_loader=forget_train_dl,
                #     model=student_model)
                # print("Results MIA")
                # print(results_mia)
                # incorrect["mia_entropy"] = results_mia

                results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                      shadow_test=test_train_mia_dl,
                                      target_train=forget_train_dl,
                                      target_test=None,
                                      model=student_model)
                print("Results MIA confidence")
                print(results_mia)
                incorrect["mia_confidence"] = results_mia

                accuracy.append(incorrect)

                for ep in range(1, 4):
                    # After 1 epoch
                    inc_resumed = {}
                    # history = fit_one_cycle(1, student_model, retain_train_subset_dl, valid_dl, device=device)
                    history = fit_one_cycle(1, student_model, retain_train_dl,
                                            retain_valid_dl,
                                            device=device)
                    results = evaluate(student_model, retain_valid_dl, device)
                    inc_resumed["dr"] = results["Acc"]

                    # Performance of Fully trained model on retain class set
                    results = evaluate(student_model, retain_class_valid_dl, device)
                    # loss.append(["Loss"]
                    inc_resumed["dr_class"] = results["Acc"]

                    # performance of unlearned model on forget set
                    results = evaluate(student_model, forget_valid_dl, device)
                    inc_resumed["df"] = results["Acc"]
                    inc_resumed["name"] = f"inc_resumed_{config_string}_ep_{ep}"

                    logit_score = UnLearningScore(student_model, unlearning_teacher,
                                                  forget_valid_dl, 256,
                                                  device)
                    print("IC Score: {}".format(logit_score))
                    js_div_logit = 1 - UnLearningScore(gold_model, student_model,
                                                       forget_valid_dl, 256,
                                                       device)
                    print("JS Div: {}".format(js_div_logit))

                    inc_resumed["zrf"] = logit_score.item()
                    inc_resumed["js_div"] = js_div_logit.item()

                    # results_mia = get_membership_attack_prob(
                    #     retain_loader=retain_train_mia_dl,
                    #     forget_loader=test_train_mia_dl,
                    #     test_loader=forget_train_dl,
                    #     model=student_model)
                    # print("Results MIA")
                    # print(results_mia)
                    # inc_resumed["mia_entropy"] = results_mia

                    results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                          shadow_test=test_train_mia_dl,
                                          target_train=forget_train_dl,
                                          target_test=None,
                                          model=student_model)
                    print("Results MIA confidence")
                    print(results_mia)
                    inc_resumed["mia_confidence"] = results_mia

                    accuracy.append(inc_resumed)

        for config in configs:
            variable_lr = config["variable_lr"]
            lr = config["lr"]
            epochs_unlearning = config["epochs"]
            config_string = f"{variable_lr}_{lr}_{epochs_unlearning}"
            for seed in range(0, 3):
                # UnLearning via modified teacher's logit
                print("[Logit] Unlearning..")
                logit = {}
                device = device
                global_model = get_model(model_arch, device).eval()
                student_model = get_model(model_arch, device)
                student_model.load_state_dict(
                    torch.load(
                        "ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
                        map_location=device))
                global_model.load_state_dict(
                    torch.load(
                        "ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
                        map_location=device))
                # model = model.eval()
                KL_temperature = 1

                logit_unlearner(model=student_model, global_model=global_model,
                                forget_data=forget_train, epochs=epochs_unlearning,
                                lr=lr,
                                batch_size=batch_size, num_workers=0, device=device,
                                KL_temperature=KL_temperature, variable_lr=variable_lr)

                # performance of unlearned model on retain set
                results = evaluate(student_model, retain_valid_dl, device)
                logit["dr"] = results["Acc"]

                # Performance of Fully trained model on retain class set
                results = evaluate(student_model, retain_class_valid_dl, device)
                # loss.append(["Loss"]
                logit["dr_class"] = results["Acc"]

                # performance of unlearned model on forget set
                results = evaluate(student_model, forget_valid_dl, device)
                logit["df"] = results["Acc"]
                logit["name"] = f"logit_{config_string}"

                logit_score = UnLearningScore(student_model, unlearning_teacher,
                                              forget_valid_dl, 256,
                                              device)
                print("IC Score: {}".format(logit_score))
                js_div_logit = 1 - UnLearningScore(gold_model, student_model,
                                                   forget_valid_dl, 256,
                                                   device)
                print("JS Div: {}".format(js_div_logit))

                logit["zrf"] = logit_score.item()
                logit["js_div"] = js_div_logit.item()

                # results_mia = get_membership_attack_prob(
                #     retain_loader=retain_train_mia_dl,
                #     forget_loader=test_train_mia_dl,
                #     test_loader=forget_train_dl,
                #     model=student_model)
                # print("Results MIA")
                # print(results_mia)
                # logit["mia_entropy"] = results_mia

                results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                      shadow_test=test_train_mia_dl,
                                      target_train=forget_train_dl,
                                      target_test=None,
                                      model=student_model)
                print("Results MIA confidence")
                print(results_mia)
                logit["mia_confidence"] = results_mia

                accuracy.append(logit)

                for ep in range(1, 4):
                    # After 1 epoch
                    logit_resumed = {}
                    # history = fit_one_cycle(1, student_model, retain_train_subset_dl, valid_dl, device=device)
                    history = fit_one_cycle(1, student_model, retain_train_dl,
                                            retain_valid_dl,
                                            device=device)
                    results = evaluate(student_model, retain_valid_dl, device)
                    logit_resumed["dr"] = results["Acc"]

                    # Performance of Fully trained model on retain class set
                    results = evaluate(student_model, retain_class_valid_dl, device)
                    # loss.append(["Loss"]
                    logit_resumed["dr_class"] = results["Acc"]

                    # performance of unlearned model on forget set
                    results = evaluate(student_model, forget_valid_dl, device)
                    logit_resumed["df"] = results["Acc"]
                    logit_resumed["name"] = f"logit_resumed_{config_string}_ep_{ep}"

                    logit_score = UnLearningScore(student_model, unlearning_teacher,
                                                  forget_valid_dl, 256,
                                                  device)
                    print("IC Score: {}".format(logit_score))
                    js_div_logit = 1 - UnLearningScore(gold_model, student_model,
                                                       forget_valid_dl, 256,
                                                       device)
                    print("JS Div: {}".format(js_div_logit))

                    logit_resumed["zrf"] = logit_score.item()
                    logit_resumed["js_div"] = js_div_logit.item()

                    # results_mia = get_membership_attack_prob(
                    #     retain_loader=retain_train_mia_dl,
                    #     forget_loader=test_train_mia_dl,
                    #     test_loader=forget_train_dl,
                    #     model=student_model)
                    # print("Results MIA")
                    # print(results_mia)
                    # logit_resumed["mia_entropy"] = results_mia

                    results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                          shadow_test=test_train_mia_dl,
                                          target_train=forget_train_dl,
                                          target_test=None,
                                          model=student_model)
                    print("Results MIA confidence")
                    print(results_mia)
                    logit_resumed["mia_confidence"] = results_mia

                    accuracy.append(logit_resumed)

        for config in configs:
            variable_lr = config["variable_lr"]
            lr = config["lr"]
            epochs_unlearning = config["epochs"]
            config_string = f"{variable_lr}_{lr}_{epochs_unlearning}"
            for seed in range(0, 3):
                # UnLearning via modified teacher's logit
                print("[Softmax] Unlearning..")
                softmax = {}
                device = device
                global_model = get_model(model_arch, device).eval()
                student_model = get_model(model_arch, device)
                student_model.load_state_dict(
                    torch.load(
                        "ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
                        map_location=device))
                global_model.load_state_dict(
                    torch.load(
                        "ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
                        map_location=device))
                # model = model.eval()
                KL_temperature = 1

                softmax_unlearner(model=student_model, global_model=global_model,
                                  forget_data=forget_train, epochs=epochs_unlearning,
                                  lr=lr,
                                  batch_size=256, num_workers=0, device=device,
                                  KL_temperature=KL_temperature,
                                  variable_lr=variable_lr)

                # performance of unlearned model on retain set
                results = evaluate(student_model, retain_valid_dl, device)
                softmax["dr"] = results["Acc"]

                # Performance of Fully trained model on retain class set
                results = evaluate(student_model, retain_class_valid_dl, device)
                # loss.append(["Loss"]
                softmax["dr_class"] = results["Acc"]

                # performance of unlearned model on forget set
                results = evaluate(student_model, forget_valid_dl, device)
                softmax["df"] = results["Acc"]
                softmax["name"] = f"softmax_{config_string}"

                softmax_score = UnLearningScore(student_model, unlearning_teacher,
                                                forget_valid_dl, 256,
                                                device)
                print("IC Score: {}".format(softmax_score))
                js_div_softmax = 1 - UnLearningScore(gold_model, student_model,
                                                     forget_valid_dl, 256,
                                                     device)
                print("JS Div: {}".format(js_div_softmax))

                softmax["zrf"] = softmax_score.item()
                softmax["js_div"] = js_div_softmax.item()

                results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                      shadow_test=test_train_mia_dl,
                                      target_train=forget_train_dl,
                                      target_test=None,
                                      model=student_model)
                print("Results MIA confidence")
                print(results_mia)
                softmax["mia_confidence"] = results_mia

                accuracy.append(softmax)

                for ep in range(1, 4):
                    # After 1 epoch
                    softmax_resumed = {}
                    # history = fit_one_cycle(1, student_model, retain_train_subset_dl, valid_dl, device=device)
                    history = fit_one_cycle(1, student_model, retain_train_dl,
                                            retain_valid_dl,
                                            device=device)
                    results = evaluate(student_model, retain_valid_dl, device)
                    softmax_resumed["dr"] = results["Acc"]

                    # Performance of Fully trained model on retain class set
                    results = evaluate(student_model, retain_class_valid_dl, device)
                    # loss.append(["Loss"]
                    softmax_resumed["dr_class"] = results["Acc"]

                    # performance of unlearned model on forget set
                    results = evaluate(student_model, forget_valid_dl, device)
                    softmax_resumed["df"] = results["Acc"]
                    softmax_resumed["name"] = f"softmax_resumed_{config_string}_ep_{ep}"

                    softmax_score = UnLearningScore(student_model, unlearning_teacher,
                                                    forget_valid_dl, 256,
                                                    device)
                    print("IC Score: {}".format(softmax_score))
                    js_div_softmax = 1 - UnLearningScore(gold_model, student_model,
                                                         forget_valid_dl, 256,
                                                         device)
                    print("JS Div: {}".format(js_div_softmax))

                    softmax_resumed["zrf"] = softmax_score.item()
                    softmax_resumed["js_div"] = js_div_softmax.item()

                    results_mia = SVC_MIA(shadow_train=retain_train_mia_dl,
                                          shadow_test=test_train_mia_dl,
                                          target_train=forget_train_dl,
                                          target_test=None,
                                          model=student_model)
                    print("Results MIA confidence")
                    print(results_mia)
                    softmax_resumed["mia_confidence"] = results_mia

                    accuracy.append(softmax_resumed)
        #
        # for seed in range(0, 10):
        #     # UnLearning via modified teacher's logit
        #     print("Unlearning..")
        #     logit_head = {}
        #     device = device
        #     # global_model = ResNet18(num_classes=20, pretrained=False).to(device).eval()
        #     global_model = get_model(model_arch, device).eval()
        #     # student_model = ResNet18(num_classes=20, pretrained=False).to(device)
        #     student_model = get_model(model_arch, device)
        #     student_model.load_state_dict(
        #         torch.load("ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
        #                    map_location=device))
        #     global_model.load_state_dict(
        #         torch.load("ResNET18_CIFAR100Super20_Pretrained_ALL_CLASSES_5_Epochs.pt",
        #                    map_location=device))
        #     # model = model.eval()
        #     KL_temperature = 1
        #     for p in student_model.base.parameters():
        #         p.requires_grad = False
        #
        #     optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
        #
        #     logit_unlearner(model=student_model, global_model=global_model,
        #                     forget_data=forget_train, epochs=40,
        #                     optimizer=optimizer, lr=0.001,
        #                     batch_size=128, num_workers=0, device=device,
        #                     KL_temperature=KL_temperature)
        #
        #     # performance of unlearned model on retain set
        #     results = evaluate(student_model, retain_valid_dl, device)
        #     logit_head["dr"] = results["Acc"]
        #
        #     # Performance of Fully trained model on retain class set
        #     results = evaluate(student_model, retain_class_valid_dl, device)
        #     # loss.append(["Loss"]
        #     logit_head["dr_class"] = results["Acc"]
        #
        #     # performance of unlearned model on forget set
        #     results = evaluate(student_model, forget_valid_dl, device)
        #     logit_head["df"] = results["Acc"]
        #     logit_head["name"] = "logit_head"
        #
        #     logit_score = UnLearningScore(student_model, unlearning_teacher,
        #                                       forget_valid_dl, 256,
        #                                       device)
        #     print("IC Score: {}".format(logit_score))
        #     js_div_logit = 1 - UnLearningScore(gold_model, student_model,
        #                                            forget_valid_dl, 256,
        #                                            device)
        #     print("JS Div: {}".format(js_div_logit))
        #
        #     logit_head["zrf"] = logit_score.item()
        #     logit_head["js_div"] = js_div_logit.item()
        #
        #     accuracy.append(logit_head)
        #
        #     # After 1 epoch
        #     logit_head_resumed = {}
        #     # history = fit_one_cycle(1, student_model, retain_train_subset_dl, valid_dl, device=device)
        #
        #     for p in student_model.base.parameters():
        #         p.requires_grad = True
        #
        #     history = fit_one_cycle(1, student_model, retain_train_dl, retain_valid_dl,
        #                             device=device)
        #     results = evaluate(student_model, retain_valid_dl, device)
        #     logit_head_resumed["dr"] = results["Acc"]
        #
        #     # Performance of Fully trained model on retain class set
        #     results = evaluate(student_model, retain_class_valid_dl, device)
        #     # loss.append(["Loss"]
        #     logit_head_resumed["dr_class"] = results["Acc"]
        #
        #     # performance of unlearned model on forget set
        #     results = evaluate(student_model, forget_valid_dl, device)
        #     logit_head_resumed["df"] = results["Acc"]
        #     logit_head_resumed["name"] = "logit_head_resumed"
        #
        #     logit_score = UnLearningScore(student_model, unlearning_teacher,
        #                                   forget_valid_dl, 256,
        #                                   device)
        #     print("IC Score: {}".format(logit_score))
        #     js_div_logit = 1 - UnLearningScore(gold_model, student_model,
        #                                        forget_valid_dl, 256,
        #                                        device)
        #     print("JS Div: {}".format(js_div_logit))
        #
        #     logit_head_resumed["zrf"] = logit_score.item()
        #     logit_head_resumed["js_div"] = js_div_logit.item()
        #
        #     accuracy.append(logit_head_resumed)

        print(accuracy)

        df = pd.DataFrame(accuracy)
        # print(df)
        # df = df.groupby(['name'])[['dr', 'dr_class', 'df', 'zrf', 'js_div']].agg(['mean', 'std'])
        print(df)
        filename = f'results_{experiment}.csv'
        path = os.path.join("can_bad_teaching", "results_csv", model_arch, filename)
        df.to_csv(path, mode='a', header=not os.path.exists(path))


if __name__ == "__main__":
    main()

# 5 epochs
# batch size: 32
#                   dr             dr_class              df
#                 mean       std       mean       std  mean       std
# name
# incorrect  36.483373  5.933172  10.531684  9.437036   2.1  2.024846
# logit      69.143279  0.436375  40.483941  2.927931   3.7  1.337494
# original   74.098328       NaN  68.142365       NaN  69.0       NaN
# retrain    76.015816       NaN  80.794273       NaN   1.0       NaN
# unlearned  76.852501       NaN  72.135414       NaN   0.0       NaN

# batch size: 64
#                   dr             dr_class              df
#                 mean       std       mean       std  mean       std
# name
# incorrect  66.252073  1.422636  59.038629  4.702461  14.2  3.881580
# logit      70.882133  0.676678  53.235677  2.149435   9.3  2.311805
# original   77.549431       NaN  90.928818       NaN  86.0       NaN
# retrain    77.180229       NaN  79.600693       NaN   2.0       NaN
# unlearned  79.702637       NaN  92.165802       NaN  18.0       NaN

# batch size: 128
#                   dr             dr_class              df
#                 mean       std       mean       std  mean       std
# name
# incorrect  69.273511  0.551090  63.355035  2.521540  11.5  2.013841
# logit      70.768903  0.429791  47.035590  3.243747   6.6  1.074968
# original   76.962906       NaN  88.823784       NaN  81.0       NaN
# retrain    77.460915       NaN  87.868927       NaN   4.0       NaN
# unlearned  79.331116       NaN  88.433159       NaN  13.0       NaN


# epochs 20 early stopping patience = 3. batch 128
#                   dr             dr_class               df
#                 mean       std       mean        std  mean       std
# name
# incorrect  57.836563  3.804787  45.635851  10.572314   3.5  2.013841
# logit      72.076907  0.669879  44.670139   2.409868   6.1  0.994429
# original   78.009003       NaN  87.868927        NaN  71.0       NaN
# retrain    77.451134       NaN  56.358505        NaN   1.0       NaN
# unlearned  80.607811       NaN  86.046005        NaN   5.0       NaN
#
#                   dr             dr_class              df
#                 mean       std       mean       std  mean       std
# name
# incorrect  68.403049  1.043599  56.512586  6.804734   2.6  1.577621
# original   78.470901       NaN  89.149307       NaN  85.0       NaN
# retrain    75.232925       NaN  61.436630       NaN   0.0       NaN
# unlearned  80.588013       NaN  90.625000       NaN  50.0       NaN


# ROCKET
#                   dr             dr_class              df
#                 mean       std       mean       std  mean       std
# name
# original   77.017876       NaN  91.514755       NaN  74.0       NaN
# retrain    77.887993  1.007365  82.760416  3.757458   3.8  2.616189
# unlearned  79.237569  0.191871  91.668836  1.174127   7.0  3.559026
# incorrect  57.965233  1.220379  56.957465  6.953287   3.4  2.590581
# logit      68.964155  0.613142  50.442709  2.050401   4.2  1.032796

# 1. verify recovery
