import torch

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    for name, para in net.named_parameters():
        if para.requires_grad:
            print(name, ": ", para.shape)
    return {"Total": total_num, "Trainable": trainable_num}


def transfer_model(pretrained_file, model):
    pretrained_dict = torch.load(pretrained_file).state_dict()  # Get pretrained dict
    model_dict = model.state_dict()  # get model dict
    model_keys = model.basemodel.state_dict().keys()
    pretrained_dict = transfer_state_dict(pretrained_dict, model_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_keys):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_keys:
            state_dict["basemodel." + k] = v
        else:
            print("Missing key(s) in state_dict: {}".format(k))
    return state_dict
