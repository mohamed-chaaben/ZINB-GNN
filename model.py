from torch.autograd import Variable
import torch.optim as optim
from model import Multi_STGAC, ZINB, ZINB_GNN
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


main_df = pd.read_pickle("/Users/mohamedchaaben/Downloads/STGAC-master/inputs/main_matrix_5_23.pkl")
A = np.load("A.npy")
train_loader, valid_loader, test_loader, max_load = PrepareDataset(main_df)


n_input = 410
n_hidden = 410
p_dropout = 0.5
lr = 1e-5
n_epochs = 10


zinb_gnn_model = ZINB_GNN(Multi_STGAC(A), ZINB, n_input, n_hidden, p_dropout)
optimizer = optim.Adam(zinb_gnn_model.parameters(), lr=lr)

# define data loading and preprocessing steps
train_loader = train_loader
val_loader = valid_loader

# training loop
for epoch in range(n_epochs):
    # set model to training mode
    zinb_gnn_model.train()

    # iterate over batches of training data
    for data in train_loader:
        inputs_R, labels_R = data[0]
        inputs_D, labels_D = data[1]
        inputs_W, labels_W = data[2]

        BNP_R, BNP_D, BNP_W = create_BNP(inputs_R), create_BNP(inputs_D), create_BNP(inputs_W)
        BNP_R, BNP_D, BNP_W = Variable(BNP_R), Variable(BNP_D), Variable(BNP_W)


        inputs_R, labels_R = Variable(inputs_R[:,:,1:]), Variable(labels_R[:,:,1:])
        inputs_D, labels_D = Variable(inputs_D[:,:,1:]), Variable(labels_D[:,:,1:])
        inputs_W, labels_W = Variable(inputs_W[:,:,1:]), Variable(labels_W[:,:,1:])

        inputs_R = inputs_R.to(device)
        inputs_D = inputs_D.to(device)
        inputs_W = inputs_W.to(device)
        BNP_R = BNP_R.to(device)
        BNP_D = BNP_D.to(device)
        BNP_W = BNP_W.to(device)


        optimizer.zero_grad()

        n, p, pi = zinb_gnn_model(inputs_R, inputs_D, inputs_W, BNP_R, BNP_D, BNP_W)
        loss = zinb_gnn_model.zinb_loss(labels_R, n, p, pi)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: training loss = {loss:.3f}")

    # set model to evaluation mode and compute validation loss
    zinb_gnn_model.eval()
    with torch.no_grad():
        val_loss = 0
        for val_data in val_loader:
            inputs_val_R, labels_val_R = val_data[0]
            inputs_val_D, labels_val_D = val_data[1]
            inputs_val_W, labels_val_W = val_data[2]


            BNP_val_R, BNP_val_D, BNP_val_W = create_BNP(inputs_val_R), create_BNP(inputs_val_D), create_BNP(inputs_val_W)
            BNP_val_R, BNP_val_D, BNP_val_W = Variable(BNP_val_R), Variable(BNP_val_D), Variable(BNP_val_W)

            inputs_val_R, labels_val_R = Variable(inputs_val_R[:,:,1:]), Variable(labels_val_R[:,:,1:])
            inputs_val_D, labels_val_D = Variable(inputs_val_D[:,:,1:]), Variable(labels_val_D[:,:,1:])
            inputs_val_W, labels_val_W = Variable(inputs_val_W[:,:,1:]), Variable(labels_val_W[:,:,1:])

            inputs_val_R = inputs_val_R.to(device)
            inputs_val_D = inputs_val_D.to(device)
            inputs_val_W = inputs_val_W.to(device)
            BNP_val_R = BNP_val_R.to(device)
            BNP_val_D = BNP_val_D.to(device)
            BNP_val_W = BNP_val_W.to(device)

            n, p, pi = zinb_gnn_model(inputs_val_R, inputs_val_D, inputs_val_W, BNP_val_R, BNP_val_D, BNP_val_W)
            val_loss += zinb_gnn_model.zinb_loss(labels_val_R, n, p, pi)

            valid_pred_flow = n * p * (1-pi)
            print(valid_pred_flow.shape)
            print(labels_val_R.shape)

        val_loss /= len(val_loader)

    # print epoch number and validation loss
    print(f"Epoch {epoch+1}: validation loss = {val_loss:.3f}")
