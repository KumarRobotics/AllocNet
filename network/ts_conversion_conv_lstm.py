import torch
import yaml

from utils.learning.minsnap_network_conv_lstm_ts import ConvLSTMMinimalSnapNetwork

seq_len = 5

models_dir = "models/minsnap_dlstm_10_len5_tokenthresh0_42"

model_idx = 2008



model_dir = models_dir + "/checkpoint" + str(model_idx) + ".pt"

config_dir = models_dir + "/config.yaml"
config = yaml.load(open(config_dir), Loader=yaml.FullLoader)

seg = config['planning']['seg']
max_hpoly_length = 50

print("Load model from: ", model_dir)
model = ConvLSTMMinimalSnapNetwork(seg=seg, max_poly_length=max_hpoly_length, hidden_size=256, seq_len=seq_len)

checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint["model_state_dict"])

print("Convert to TorchScript")
scripted_model = torch.jit.script(model).cuda().eval()

save_dir = models_dir + "/scripted" + str(model_idx) + ".pt"

torch.jit.save(scripted_model, save_dir)
print("Save converted model to: ", save_dir)