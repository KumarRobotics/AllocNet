import torch

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params



def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params
def test():


    # Load the model
    model = torch.load('seq10_rest2rest.pt')

    # Set the model to evaluation
    print(model)  
    # Print the parameters
    for name, param in model.named_parameters():
     print(f"Parameter name: {name}, shape: {param.shape}")

    # Evaluate the model
    count = count_parameters(model)
    print(f"Number of parameters: {count}")

    trainable_count = count_trainable_parameters(model)
    print(f"Number of trainable parameters: {trainable_count}")



if __name__ == '__main__':
    test()