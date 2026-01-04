import torch, os
import torch.nn as nn
import streamlit as st

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.gestures = st.session_state.gestures

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output_layer(x)
        return x


@st.cache_resource
def load_model(device, weight_dir):
    input_size = 258
    hidden_size = 64
    num_classes = 30
    model = CustomLSTM(input_size, hidden_size, num_classes)

    # load weights into model
    if os.path.exists(weight_dir):
        try:
            loaded_model_state_dict = torch.load(weight_dir, map_location=device)
            model.load_state_dict(loaded_model_state_dict)
            print(f"Model loaded from {weight_dir}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: {weight_dir} not found.")

    # set model to evaluation mode
    model.to(device)
    model.eval()
    return model