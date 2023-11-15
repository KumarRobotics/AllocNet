import torch
import yaml

from utils.learning.minsnap_network_conv_mlp_as_ts import ConvMLPMinimalSnapNetwork4AblationStudy



models_dir = "models/minsnap_conv_mlp_as"

model_idx = 2114



model_dir = models_dir + "/checkpoint" + str(model_idx) + ".pt"

config_dir = models_dir + "/config.yaml"
config = yaml.load(open(config_dir), Loader=yaml.FullLoader)

seg = config['planning']['seg']
max_hpoly_length = 50

print("Load model from: ", model_dir)
model = ConvMLPMinimalSnapNetwork4AblationStudy(seg=seg, max_poly_length=max_hpoly_length, hidden_size=256)

checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint["model_state_dict"])

print("Convert to TorchScript")
scripted_model = torch.jit.script(model).cuda().eval()

save_dir = models_dir + "/scripted" + str(model_idx) + ".pt"

torch.jit.save(scripted_model, save_dir)
print("Save converted model to: ", save_dir)