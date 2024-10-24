import torch
def inf_time(model):
    iterations = 300
    device = torch.device("cuda:0")
    model.to(device)

    random_input = torch.randn(1, 1, 1024).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Preheat GPU
    for _ in range(50):
        _ = model(random_input)

    # Measure inference time
    times = torch.zeros(iterations)     # Save the time of each iteration
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # Synchronize GPU time
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # Calculate time
            times[iter] = curr_time
    mean_time = times.mean().item()/len(random_input)
    return mean_time