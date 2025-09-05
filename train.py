import torch

def train_model(model, dataloader, optimizer, loss_fn, epochs=50, freeze_first_n_lstm=0):
    loss_list = []

    if freeze_first_n_lstm > 0:
        for lstm in model.lstm_layers[:freeze_first_n_lstm]:
            for param in lstm.parameters():
                param.requires_grad = False

    for epoch in range(epochs):
        total_loss = 0
        for x, h in dataloader:
            optimizer.zero_grad()
            hat_h = model(x.float())
            loss = loss_fn(hat_h, h.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        loss_list.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    return loss_list
