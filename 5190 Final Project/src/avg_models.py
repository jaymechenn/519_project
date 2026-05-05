import torch

sd1 = torch.load("model_1.pt", map_location="cpu")
sd2 = torch.load("model_2.pt", map_location="cpu")

avg_sd = {}

for k in sd1:
    avg_sd[k] = (sd1[k] + sd2[k]) / 2

torch.save(avg_sd, "model.pt")

print("Saved averaged model.pt")